import math
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
from onnx_asr.asr import TimestampedResult
import openai.types.audio
import pytest
from pytest_mock import MockerFixture

from speaches.audio import Audio
from speaches.config import OrtOptions
from speaches.executors.parakeet import ParakeetModelManager, model_language, to_verbose_transcription
from speaches.executors.shared.errors import UnsupportedResponseFormatError
from speaches.executors.shared.handler_protocol import TranscriptionRequest
from speaches.executors.silero_vad_v5 import MergedSegment, SpeechTimestamp
from speaches.routers.stt import DEFAULT_VAD_OPTIONS

SAMPLE_RATE = 16000


def transcription_request(
    response_format: str, speech_segments: list[SpeechTimestamp] | None = None
) -> TranscriptionRequest:
    return TranscriptionRequest(
        audio=Audio(np.arange(45 * SAMPLE_RATE, dtype=np.float32), sample_rate=SAMPLE_RATE),
        model="istupakov/parakeet-tdt-0.6b-v2-onnx",
        response_format=response_format,  # pyrefly: ignore[bad-argument-type]
        timestamp_granularities=["segment"],
        speech_segments=[SpeechTimestamp(start=0, end=SAMPLE_RATE)] if speech_segments is None else speech_segments,
        vad_options=DEFAULT_VAD_OPTIONS,
    )


@pytest.fixture
def parakeet(mocker: MockerFixture) -> MagicMock:
    model = MagicMock()
    model.recognize.return_value = "Hello there."
    mocker.patch.object(ParakeetModelManager, "_load_fn", return_value=model)
    return model


@pytest.mark.parametrize("response_format", ["text", "json"])
def test_text_and_json_skip_timestamps_and_logprobs(parakeet: MagicMock, response_format: str) -> None:
    manager = ParakeetModelManager(ttl=-1, ort_opts=OrtOptions())

    manager.handle_non_streaming_transcription_request(transcription_request(response_format))

    parakeet.with_timestamps.assert_not_called()
    parakeet.recognize.assert_called_once()


@pytest.mark.parametrize("response_format", ["srt", "vtt", "diarized_json"])
def test_an_unsupported_format_is_rejected_before_the_model_loads(parakeet: MagicMock, response_format: str) -> None:
    manager = ParakeetModelManager(ttl=-1, ort_opts=OrtOptions())

    with pytest.raises(UnsupportedResponseFormatError):
        manager.handle_non_streaming_transcription_request(transcription_request(response_format))

    parakeet.recognize.assert_not_called()


def test_nearby_speech_segments_are_transcribed_together(parakeet: MagicMock) -> None:
    manager = ParakeetModelManager(ttl=-1, ort_opts=OrtOptions())
    speech = [SpeechTimestamp(start=0, end=8000), SpeechTimestamp(start=32000, end=48000)]

    manager.handle_non_streaming_transcription_request(transcription_request("text", speech))

    (waveform,) = (call.args[0] for call in parakeet.recognize.call_args_list)
    assert (waveform[0], len(waveform)) == (0, 48000)


def test_speech_further_apart_than_a_window_is_transcribed_in_windows_and_joined(parakeet: MagicMock) -> None:
    parakeet.recognize.side_effect = ["Hello there.", "How are you?"]
    manager = ParakeetModelManager(ttl=-1, ort_opts=OrtOptions())
    speech = [SpeechTimestamp(start=0, end=8000), SpeechTimestamp(start=40 * SAMPLE_RATE, end=41 * SAMPLE_RATE)]

    text, _ = manager.handle_non_streaming_transcription_request(transcription_request("text", speech))

    assert text == "Hello there. How are you?"
    waveforms = [call.args[0] for call in parakeet.recognize.call_args_list]
    assert [(w[0], len(w)) for w in waveforms] == [(0, 8000), (40 * SAMPLE_RATE, SAMPLE_RATE)]


@pytest.mark.usefixtures("parakeet")
def test_the_requests_speech_segments_are_left_unchanged() -> None:
    manager = ParakeetModelManager(ttl=-1, ort_opts=OrtOptions())
    speech = [SpeechTimestamp(start=0, end=20000), SpeechTimestamp(start=16000, end=40000)]
    request = transcription_request("text", speech)

    manager.handle_non_streaming_transcription_request(request)

    assert request.speech_segments == [SpeechTimestamp(start=0, end=20000), SpeechTimestamp(start=16000, end=40000)]


def test_audio_without_speech_gives_empty_text(parakeet: MagicMock) -> None:
    manager = ParakeetModelManager(ttl=-1, ort_opts=OrtOptions())

    transcription = manager.handle_non_streaming_transcription_request(transcription_request("json", []))

    assert transcription == openai.types.audio.Transcription(text="")
    parakeet.recognize.assert_not_called()


# tokens, start times and log-probabilities as `istupakov/parakeet-tdt-0.6b-v2-onnx` returned them for a 2.18 s clip
RESULT = TimestampedResult(
    text="Okay, what time does the game start tonight?",
    tokens=[" Okay", ",", " what", " time", " does", " the", " g", "ame", " start", " t", "on", "ight", "?"],
    timestamps=[0.0, 0.32, 0.56, 0.72, 0.96, 1.12, 1.2, 1.36, 1.52, 1.76, 1.84, 1.92, 2.08],
    logprobs=[-0.002, -0.005, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.001, -0.0, 0.0, -0.0],
)
DURATION = 2.18


def one_window(start: float, end: float, regions: list[tuple[float, float]] | None = None) -> MergedSegment:
    regions = regions or [(start, end)]
    return MergedSegment(
        start=int(start * SAMPLE_RATE),
        end=int(end * SAMPLE_RATE),
        segments=[(int(s * SAMPLE_RATE), int(e * SAMPLE_RATE)) for s, e in regions],
    )


def verbose(
    windows: list[MergedSegment], results: list[TimestampedResult], duration: float = DURATION, language: str = "en"
) -> openai.types.audio.TranscriptionVerbose:
    return to_verbose_transcription(
        windows, results, sample_rate=SAMPLE_RATE, duration=duration, language=language, word_timestamps=True
    )


def test_tokens_are_grouped_into_words_with_punctuation_attached() -> None:
    transcription = verbose([one_window(0.0, DURATION)], [RESULT])

    assert transcription.words is not None
    assert [w.word for w in transcription.words] == [
        " Okay,",
        " what",
        " time",
        " does",
        " the",
        " game",
        " start",
        " tonight?",
    ]


def test_a_word_ends_where_the_next_begins_and_the_last_where_its_speech_region_ends() -> None:
    transcription = verbose([one_window(0.0, 2.15)], [RESULT])

    assert transcription.words is not None
    assert [(w.start, w.end) for w in transcription.words[:2]] == [(0.0, 0.56), (0.56, 0.72)]
    assert (transcription.words[-1].start, transcription.words[-1].end) == (1.76, 2.15)


def test_the_last_word_never_ends_after_the_audio() -> None:
    transcription = verbose([one_window(0.0, DURATION)], [RESULT], duration=1.95)

    assert transcription.words is not None
    assert transcription.words[-1].end == 1.95


def test_a_word_before_a_pause_ends_with_its_speech_region_not_at_the_next_word() -> None:
    result = TimestampedResult(
        text="Hey there. Can",
        # the full stop is stamped inside the pause, as Parakeet does
        tokens=[" Hey", " there", ".", " Can"],
        timestamps=[0.0, 0.24, 0.96, 1.28],
        logprobs=[0.0, 0.0, 0.0, 0.0],
    )

    transcription = verbose([one_window(0.0, 1.6, [(0.0, 1.1), (1.1, 1.6)])], [result], duration=1.6)

    assert transcription.words is not None
    assert [(w.word, w.start, w.end) for w in transcription.words] == [
        (" Hey", 0.0, 0.24),
        (" there.", 0.24, 1.1),
        (" Can", 1.28, 1.6),
    ]


@pytest.mark.parametrize(
    ("regions", "timestamps", "segments"),
    [
        # the first region has no words: the model heard both in the second
        ([(0.0, 0.5), (0.5, 1.6)], [0.6, 0.8], [" Hey there"]),
        # a region between two with words
        ([(0.0, 0.5), (0.5, 1.0), (1.0, 1.6)], [0.0, 1.2], [" Hey", " there"]),
        # the last region has no words
        ([(0.0, 1.0), (1.0, 1.6)], [0.0, 0.2], [" Hey there"]),
    ],
)
def test_a_speech_region_with_no_words_in_it_is_left_out(
    regions: list[tuple[float, float]], timestamps: list[float], segments: list[str]
) -> None:
    result = TimestampedResult(
        text="Hey there", tokens=[" Hey", " there"], timestamps=timestamps, logprobs=[-0.1, -0.2]
    )

    transcription = verbose([one_window(0.0, 1.6, regions)], [result], duration=1.6)

    assert transcription.words is not None
    assert [w.word for w in transcription.words] == [" Hey", " there"]
    assert transcription.segments is not None
    assert [s.text for s in transcription.segments] == segments


def test_each_speech_region_is_a_segment_with_times_offset_by_its_window() -> None:
    second = TimestampedResult(text="Bye.", tokens=[" Bye", "."], timestamps=[0.4, 0.56], logprobs=[-0.1, -0.3])
    windows = [one_window(0.0, 2.15, [(0.0, 1.0), (1.0, 2.15)]), one_window(40.0, 41.0)]

    transcription = verbose(windows, [RESULT, second], duration=41.0)

    assert transcription.segments is not None
    assert [(s.id, s.start, s.end, s.text) for s in transcription.segments] == [
        (0, 0.0, 1.0, " Okay, what time does"),
        (1, 1.12, 2.15, " the game start tonight?"),
        (2, 40.4, 41.0, " Bye."),
    ]
    assert transcription.segments[2].seek == 4000
    assert math.isclose(transcription.segments[2].avg_logprob, -0.2)
    assert transcription.text == "Okay, what time does the game start tonight? Bye."


def test_a_word_carries_the_mean_probability_of_its_tokens() -> None:
    transcription = verbose([one_window(0.0, DURATION)], [RESULT])

    assert transcription.words is not None
    expected = sum(math.exp(lp) for lp in [-0.002, -0.005]) / 2
    assert math.isclose(transcription.words[0].probability, expected)  # pyrefly: ignore[missing-attribute]


def test_without_logprobs_a_word_has_no_probability() -> None:
    result = TimestampedResult(text=RESULT.text, tokens=RESULT.tokens, timestamps=RESULT.timestamps)

    transcription = verbose([one_window(0.0, DURATION)], [result])

    assert transcription.words is not None
    assert len(transcription.words) == 8
    assert "probability" not in transcription.words[0].model_dump()


def test_words_are_only_returned_when_asked_for() -> None:
    transcription = to_verbose_transcription(
        [one_window(0.0, DURATION)],
        [RESULT],
        sample_rate=SAMPLE_RATE,
        duration=DURATION,
        language="en",
        word_timestamps=False,
    )

    assert transcription.words is None
    assert transcription.segments is not None
    assert len(transcription.segments) == 1


def test_silence_gives_no_segments_and_no_words() -> None:
    transcription = verbose([], [], duration=1.0)

    assert (transcription.text, transcription.segments, transcription.words) == ("", [], [])


def test_verbose_json_text_matches_json_text(parakeet: MagicMock) -> None:
    parakeet.recognize.return_value = RESULT.text
    parakeet.with_timestamps.return_value.recognize.return_value = RESULT
    manager = ParakeetModelManager(ttl=-1, ort_opts=OrtOptions())
    speech = [SpeechTimestamp(start=0, end=int(DURATION * SAMPLE_RATE))]

    json = manager.handle_non_streaming_transcription_request(transcription_request("json", speech))
    verbose_json = manager.handle_non_streaming_transcription_request(transcription_request("verbose_json", speech))

    assert isinstance(json, openai.types.audio.Transcription)
    assert isinstance(verbose_json, openai.types.audio.TranscriptionVerbose)
    assert verbose_json.text == json.text


def test_without_a_requested_language_the_model_cards_only_language_is_reported(
    mocker: MockerFixture, tmp_path: Path
) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("---\nlanguage:\n- en\n---\n")
    mocker.patch("speaches.executors.parakeet.list_model_files", return_value=iter([readme]))

    assert model_language("istupakov/parakeet-tdt-0.6b-v2-onnx") == "en"


def test_a_model_card_with_several_languages_reports_none(mocker: MockerFixture, tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("---\nlanguage:\n- en\n- de\n---\n")
    mocker.patch("speaches.executors.parakeet.list_model_files", return_value=iter([readme]))

    assert model_language("istupakov/parakeet-tdt-0.6b-v3-onnx") == ""
