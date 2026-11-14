from unittest.mock import MagicMock

import numpy as np
import pytest
from pytest_mock import MockerFixture

from speaches.audio import Audio
from speaches.config import OrtOptions
from speaches.executors.parakeet import ParakeetModelManager
from speaches.executors.shared.errors import UnsupportedResponseFormatError
from speaches.executors.shared.handler_protocol import TranscriptionRequest
from speaches.executors.silero_vad_v5 import SpeechTimestamp, VadOptions

SAMPLE_RATE = 16000


def transcription_request(response_format: str) -> TranscriptionRequest:
    return TranscriptionRequest(
        audio=Audio(np.zeros(SAMPLE_RATE, dtype=np.float32), sample_rate=SAMPLE_RATE),
        model="istupakov/parakeet-tdt-0.6b-v2-onnx",
        response_format=response_format,  # pyrefly: ignore[bad-argument-type]
        timestamp_granularities=["segment"],
        speech_segments=[SpeechTimestamp(start=0, end=SAMPLE_RATE)],
        vad_options=VadOptions(),
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
