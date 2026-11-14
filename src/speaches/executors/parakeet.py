from collections.abc import Generator
import logging
import math
from pathlib import Path
from typing import TypedDict
import zlib

from fastapi import HTTPException
import huggingface_hub
import onnx_asr
from onnx_asr.adapters import TextResultsAsrAdapter
from onnx_asr.asr import TimestampedResult
from onnx_asr.models.nemo import NemoConformerTdt
import openai.types.audio
from opentelemetry import trace

from speaches.api_types import Model
from speaches.config import OrtOptions
from speaches.executors.shared.base_model_manager import BaseModelManager, get_ort_providers_with_options
from speaches.executors.shared.handler_protocol import (
    NonStreamingTranscriptionResponse,
    StreamingTranscriptionEvent,
    TranscriptionRequest,
)
from speaches.hf_utils import (
    HfModelFilter,
    extract_language_list,
    get_cached_model_repos_info,
    get_model_card_data_from_cached_repo_info,
    list_model_files,
)
from speaches.model_registry import ModelRegistry
from speaches.tracing import traced, traced_generator

# TODO: support model quants

# LIBRARY_NAME = "onnx" # NOTE: library name is derived and not stored in the README
TASK_NAME_TAG = "automatic-speech-recognition"
# One encoder frame of a NeMo FastConformer (10 ms hop, 8x subsampling). onnx-asr reports when each token
# starts but not how long it lasts, so the last word is given one frame.
FRAME_SECONDS = 0.08
SUPPORTED_RESPONSE_FORMATS = ("text", "json", "verbose_json")
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
        # an encoder over 2 GB keeps its weights in an external `<name>.onnx.data` file
        data_files = [f"{file}?data" for file in model_files if file.endswith(".onnx")]
        allow_patterns = [*model_files, *data_files]

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
        if request.response_format not in SUPPORTED_RESPONSE_FORMATS:
            raise unsupported_response_format(request)
        with self.load_model(request.model) as parakeet:
            # TODO: issue warnings when client specifies unsupported parameters like `prompt`, `temperature`, `hotwords`, etc.
            # TODO: Use request.speech_segments for audio chunking

            results = parakeet.with_timestamps().recognize(request.audio.data)

            match request.response_format:
                case "text":
                    return results.text, "text/plain"
                case "json":
                    return openai.types.audio.Transcription(text=results.text)
                case "verbose_json":
                    return result_to_verbose_transcription(
                        results,
                        duration=request.audio.duration,
                        speech=speech_without_padding(request),
                        language=request.language,
                        word_timestamps="word" in request.timestamp_granularities,
                    )
                case _:
                    raise unsupported_response_format(request)

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


def unsupported_response_format(request: TranscriptionRequest) -> HTTPException:
    return HTTPException(
        status_code=400,
        detail=f"'{request.response_format}' response format is not supported for '{request.model}' model.",
    )


def speech_without_padding(request: TranscriptionRequest) -> list[tuple[float, float]]:
    # VAD pads each speech segment on both sides; take the padding back off to find where speech stopped
    rate = request.audio.sample_rate
    pad = request.vad_options.speech_pad_ms / 1000
    return [
        (segment.start / rate + pad, max(segment.start / rate + pad, segment.end / rate - pad))
        for segment in request.speech_segments
    ]


def speech_end_after(time: float, speech: list[tuple[float, float]]) -> float | None:
    # the segment a token starts in, or failing that the nearest segment that began before it
    started = [(start, end) for start, end in speech if start <= time]
    if not started:
        return None
    _, end = started[-1]
    return end if time <= end or time - end < FRAME_SECONDS else None


def result_to_words(
    result: TimestampedResult, duration: float, speech: list[tuple[float, float]] | None = None
) -> list[openai.types.audio.TranscriptionWord]:
    tokens = result.tokens or []
    starts = result.timestamps or []
    if len(starts) != len(tokens):
        return []
    # a token that starts with a space begins a new word; punctuation stays with the word before it
    groups: list[list[int]] = []
    for i, token in enumerate(tokens):
        if token.startswith(" ") or not groups:
            groups.append([i])
        else:
            groups[-1].append(i)
    words: list[openai.types.audio.TranscriptionWord] = []
    for n, group in enumerate(groups):
        start = starts[group[0]]
        # punctuation is stamped late, often inside the pause that follows the word
        spoken = [i for i in group if any(c.isalnum() for c in tokens[i])] or group
        last_token_start = starts[spoken[-1]]
        end = last_token_start + FRAME_SECONDS if n + 1 == len(groups) else starts[groups[n + 1][0]]
        # a word ends where the next one starts, unless speech stopped first (so a pause isn't counted as the word)
        speech_end = speech_end_after(last_token_start, speech or [])
        if speech_end is not None:
            end = min(end, max(speech_end, last_token_start + FRAME_SECONDS))
        end = min(end, duration)
        word: dict[str, str | float] = {
            "start": start,
            "end": max(start, end),
            "word": "".join(tokens[i] for i in group),
        }
        if result.logprobs is not None and len(result.logprobs) == len(tokens):
            # the mean of the tokens' probabilities, as faster-whisper reports a word's `probability`
            word["probability"] = sum(math.exp(result.logprobs[i]) for i in group) / len(group)
        words.append(openai.types.audio.TranscriptionWord.model_validate(word))
    return words


def result_to_verbose_transcription(
    result: TimestampedResult,
    *,
    duration: float,
    speech: list[tuple[float, float]] | None = None,
    language: str | None,
    word_timestamps: bool,
) -> openai.types.audio.TranscriptionVerbose:
    text = result.text.strip()
    words = result_to_words(result, duration, speech)
    segments = []
    if text:
        logprobs = result.logprobs or []
        segments.append(
            openai.types.audio.TranscriptionSegment(
                id=0,
                seek=0,
                start=words[0].start if words else 0.0,
                end=words[-1].end if words else duration,
                text=result.text,
                # onnx-asr returns token strings, not ids
                tokens=[],
                temperature=0.0,
                avg_logprob=sum(logprobs) / len(logprobs) if logprobs else 0.0,
                compression_ratio=len(text.encode()) / len(zlib.compress(text.encode())),
                # Parakeet has no no-speech head
                no_speech_prob=0.0,
            )
        )
    return openai.types.audio.TranscriptionVerbose(
        # the model doesn't report a language; echo the request's
        language=language or "",
        duration=duration,
        text=text,
        segments=segments,
        words=words if word_timestamps else None,
    )
