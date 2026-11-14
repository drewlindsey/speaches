from collections.abc import Generator
import logging
import math
from pathlib import Path
from typing import NamedTuple, TypedDict
import zlib

import huggingface_hub
import numpy as np
import numpy.typing as npt
import onnx_asr
from onnx_asr.adapters import TextResultsAsrAdapter
from onnx_asr.asr import TimestampedResult
from onnx_asr.models.nemo import NemoConformerTdt
import openai.types.audio
from opentelemetry import trace

from speaches.api_types import Model
from speaches.config import OrtOptions
from speaches.executors.shared.base_model_manager import BaseModelManager, get_ort_providers_with_options
from speaches.executors.shared.errors import UnsupportedResponseFormatError
from speaches.executors.shared.handler_protocol import (
    NonStreamingTranscriptionResponse,
    StreamingTranscriptionEvent,
    TranscriptionRequest,
)
from speaches.executors.silero_vad_v5 import MergedSegment, merge_segments
from speaches.hf_utils import (
    HfModelFilter,
    extract_language_list,
    get_cached_model_repos_info,
    get_model_card_data_from_cached_repo_info,
    list_model_files,
    load_repo_model_card_data,
)
from speaches.model_registry import ModelRegistry
from speaches.tracing import traced, traced_generator

# TODO: support model quants

# LIBRARY_NAME = "onnx" # NOTE: library name is derived and not stored in the README
TASK_NAME_TAG = "automatic-speech-recognition"
# TAGS = {"nemo-conformer-tdt"} # NOTE: I've tried to use this tag however it seems to be derived (likely from config.json) and isn't present when parsing the local model card

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

hf_model_filter = HfModelFilter(
    model_name="istupakov/parakeet-tdt",
    # library_name=LIBRARY_NAME,
    task=TASK_NAME_TAG,
    # tags=TAGS,
)


class NemoConformerTdtModelFiles(TypedDict):
    encoder: Path
    decoder_joint: Path
    vocab: Path
    config: Path


class NemoConformerTdtModelRegistry(ModelRegistry[Model, NemoConformerTdtModelFiles]):
    def list_remote_models(self) -> Generator[Model]:
        models = huggingface_hub.list_models(**self.hf_model_filter.list_model_kwargs(), cardData=True)
        for model in models:
            assert model.created_at is not None and model.card_data is not None, model
            yield Model(
                id=model.id,
                created=int(model.created_at.timestamp()),
                owned_by=model.id.split("/")[0],
                language=extract_language_list(model.card_data),
                task=TASK_NAME_TAG,
            )

    def list_local_models(self) -> Generator[Model]:
        cached_model_repos_info = get_cached_model_repos_info()
        for cached_repo_info in cached_model_repos_info:
            model_card_data = get_model_card_data_from_cached_repo_info(cached_repo_info)
            if model_card_data is None:
                continue
            if self.hf_model_filter.passes_filter(cached_repo_info.repo_id, model_card_data):
                yield Model(
                    id=cached_repo_info.repo_id,
                    created=int(cached_repo_info.last_modified),
                    owned_by=cached_repo_info.repo_id.split("/")[0],
                    language=extract_language_list(model_card_data),
                    task=TASK_NAME_TAG,
                )

    def get_model_files(self, model_id: str) -> NemoConformerTdtModelFiles:
        model_files = list(list_model_files(model_id))

        encoder_file_path = next(file_path for file_path in model_files if file_path.name == "encoder-model.onnx")
        decoder_joint_file_path = next(
            file_path for file_path in model_files if file_path.name == "decoder_joint-model.onnx"
        )
        vocab_file_path = next(file_path for file_path in model_files if file_path.name == "vocab.txt")
        config_file_path = next(file_path for file_path in model_files if file_path.name == "config.json")

        return NemoConformerTdtModelFiles(
            encoder=encoder_file_path,
            decoder_joint=decoder_joint_file_path,
            vocab=vocab_file_path,
            config=config_file_path,
        )

    def download_model_files(self, model_id: str) -> None:
        model_files = list(NemoConformerTdt._get_model_files(quantization=None).values())  # noqa: SLF001
        # mirrors onnx-asr's own resolver: `config.json` is required by `get_model_files` and by the model loader,
        # and an encoder over 2 GB keeps its weights in an external `<name>.onnx.data` file
        data_files = [f"{file}?data" for file in model_files if file.endswith(".onnx")]
        allow_patterns = ["config.json", *model_files, *data_files]

        _model_repo_path_str = huggingface_hub.snapshot_download(
            repo_id=model_id, repo_type="model", allow_patterns=[*allow_patterns, "README.md"]
        )


parakeet_model_registry = NemoConformerTdtModelRegistry(hf_model_filter=hf_model_filter)


class ParakeetModelManager(BaseModelManager[TextResultsAsrAdapter]):
    def __init__(self, ttl: int, ort_opts: OrtOptions) -> None:
        super().__init__(ttl)
        self.ort_opts = ort_opts

    def _load_fn(self, model_id: str) -> TextResultsAsrAdapter:
        providers = get_ort_providers_with_options(self.ort_opts)
        return onnx_asr.load_model(model_id, providers=providers)

    @traced()
    def handle_non_streaming_transcription_request(
        self,
        request: TranscriptionRequest,
        **_kwargs,
    ) -> NonStreamingTranscriptionResponse:
        if request.response_format not in ("text", "json", "verbose_json"):
            raise UnsupportedResponseFormatError(request.response_format, request.model)
        with self.load_model(request.model) as parakeet:
            # TODO: issue warnings when client specifies unsupported parameters like `prompt`, `temperature`, `hotwords`, etc.
            windows = speech_windows(request)
            waveforms = [request.audio.data[window["start"] : window["end"]] for window in windows]

            match request.response_format:
                case "text":
                    return recognize_text(parakeet, waveforms), "text/plain"
                case "json":
                    return openai.types.audio.Transcription(text=recognize_text(parakeet, waveforms))
                case "verbose_json":
                    timestamped = parakeet.with_timestamps()
                    return to_verbose_transcription(
                        windows,
                        [timestamped.recognize(waveform) for waveform in waveforms],
                        sample_rate=request.audio.sample_rate,
                        duration=request.audio.duration,
                        language=request.language or model_language(request.model),
                        word_timestamps="word" in request.timestamp_granularities,
                    )

    @traced_generator()
    def handle_streaming_transcription_request(
        self,
        request: TranscriptionRequest,
        **_kwargs,
    ) -> Generator[StreamingTranscriptionEvent]:
        raise NotImplementedError(f"'{request.model}' model doesn't support streaming transcription.")

    def handle_transcription_request(
        self, request: TranscriptionRequest, **kwargs
    ) -> NonStreamingTranscriptionResponse | Generator[StreamingTranscriptionEvent]:
        if request.stream:
            return self.handle_streaming_transcription_request(request, **kwargs)
        else:
            return self.handle_non_streaming_transcription_request(request, **kwargs)


def speech_windows(request: TranscriptionRequest) -> list[MergedSegment]:
    # VAD speech segments merged into windows of up to `max_speech_duration_s`, as whisper is given through
    # `clip_timestamps`. A window rather than a single segment keeps the context the model needs for punctuation.
    # `merge_segments` modifies the segments it is given, so it gets copies.
    return merge_segments([segment.model_copy() for segment in request.speech_segments], request.vad_options)


def recognize_text(parakeet: TextResultsAsrAdapter, waveforms: list[npt.NDArray[np.float32]]) -> str:
    # plain `recognize`: since onnx-asr 0.12 `with_timestamps()` also computes per-token log-probabilities
    return " ".join(text for waveform in waveforms if (text := parakeet.recognize(waveform)))


def model_language(model_id: str) -> str:
    # Parakeet doesn't detect the language; report the model's when its card lists exactly one
    readme = next((path for path in list_model_files(model_id) if path.name == "README.md"), None)
    languages = extract_language_list(load_repo_model_card_data(readme)) if readme is not None else []
    return languages[0] if len(languages) == 1 else ""


class Word(NamedTuple):
    start: float
    end: float
    tokens: list[int]


# when a word starts, and the indices of its tokens
type WordStart = tuple[float, list[int]]


def group_words(result: TimestampedResult, offset: float) -> list[WordStart]:
    # a token that starts with a space begins a new word; punctuation stays with the word before it
    words: list[WordStart] = []
    for i, (token, start) in enumerate(zip(result.tokens or [], result.timestamps or [], strict=True)):
        if token.startswith(" ") or not words:
            words.append((offset + start, [i]))
        else:
            words[-1][1].append(i)
    return words


def words_by_region(words: list[WordStart], regions: list[tuple[float, float]], duration: float) -> list[list[Word]]:
    # each word goes to the last speech region that started at or before it; both lists are in time order
    by_region: list[list[WordStart]] = [[] for _ in regions]
    r = 0
    for start, tokens in words:
        while r + 1 < len(regions) and regions[r + 1][0] <= start:
            r += 1
        by_region[r].append((start, tokens))
    return [
        with_word_ends(region_words, min(region_end, duration))
        for (_, region_end), region_words in zip(regions, by_region, strict=True)
    ]


def with_word_ends(words: list[WordStart], region_end: float) -> list[Word]:
    # a word ends where the next one starts and the last one where its speech region ends,
    # so the pause after a region isn't counted as part of its last word
    ends = [next_start for next_start, _ in words[1:]] + [region_end]
    return [
        Word(start, max(start, min(end, region_end)), tokens) for (start, tokens), end in zip(words, ends, strict=True)
    ]


def to_verbose_transcription(
    windows: list[MergedSegment],
    results: list[TimestampedResult],
    *,
    sample_rate: int,
    duration: float,
    language: str,
    word_timestamps: bool,
) -> openai.types.audio.TranscriptionVerbose:
    segments: list[openai.types.audio.TranscriptionSegment] = []
    words: list[openai.types.audio.TranscriptionWord] = []
    for window, result in zip(windows, results, strict=True):
        offset = window["start"] / sample_rate
        regions = [(start / sample_rate, end / sample_rate) for start, end in window["segments"]]
        tokens = result.tokens or []
        logprobs = result.logprobs or []
        for region_words in words_by_region(group_words(result, offset), regions, duration):
            if not region_words:
                continue
            segment_words = [
                openai.types.audio.TranscriptionWord.model_validate(
                    {
                        "start": word.start,
                        "end": word.end,
                        "word": "".join(tokens[i] for i in word.tokens),
                        # the mean of the tokens' probabilities, as faster-whisper reports a word's `probability`
                        **(
                            {"probability": sum(math.exp(logprobs[i]) for i in word.tokens) / len(word.tokens)}
                            if logprobs
                            else {}
                        ),
                    }
                )
                for word in region_words
            ]
            text = "".join(word.word for word in segment_words)
            segment_logprobs = [logprobs[i] for word in region_words for i in word.tokens] if logprobs else []
            segments.append(
                openai.types.audio.TranscriptionSegment(
                    id=len(segments),
                    # the window's offset in 10 ms frames, as whisper reports it
                    seek=window["start"] * 100 // sample_rate,
                    start=segment_words[0].start,
                    end=segment_words[-1].end,
                    text=text,
                    # onnx-asr returns token strings, not ids
                    tokens=[],
                    temperature=0.0,
                    avg_logprob=sum(segment_logprobs) / len(segment_logprobs) if segment_logprobs else 0.0,
                    compression_ratio=compression_ratio(text.strip()),
                    # Parakeet has no no-speech head
                    no_speech_prob=0.0,
                )
            )
            words.extend(segment_words)
    return openai.types.audio.TranscriptionVerbose(
        language=language,
        duration=duration,
        text=" ".join(result.text for result in results if result.text),
        segments=segments,
        words=words if word_timestamps else None,
    )


def compression_ratio(text: str) -> float:
    encoded = text.encode()
    return len(encoded) / len(zlib.compress(encoded))
