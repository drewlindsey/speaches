from unittest.mock import MagicMock

import numpy as np
import openai.types.audio
import pytest
from pytest_mock import MockerFixture

from speaches.audio import Audio
from speaches.config import OrtOptions
from speaches.executors.parakeet import ParakeetModelManager
from speaches.executors.shared.errors import UnsupportedResponseFormatError
from speaches.executors.shared.handler_protocol import TranscriptionRequest
from speaches.executors.silero_vad_v5 import SpeechTimestamp
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
