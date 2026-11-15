import io
import subprocess
import sys

import av.error
from faster_whisper.audio import decode_audio as faster_whisper_decode_audio
import numpy as np
import pytest
import soundfile as sf

from speaches.audio import decode_audio


def encoded(sample_rate: int, channels: int = 1, **kwargs: str) -> bytes:
    # two seconds of a chirp, so resampling has something to change
    t = np.arange(2 * sample_rate) / sample_rate
    signal = 0.5 * np.sin(2 * np.pi * (200 + 400 * t) * t)
    data = np.stack([signal, 0.5 * signal], axis=1) if channels == 2 else signal
    buffer = io.BytesIO()
    sf.write(buffer, data, sample_rate, **kwargs)
    return buffer.getvalue()


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(encoded(16000, format="WAV", subtype="PCM_16"), id="wav-16k"),
        pytest.param(encoded(48000, channels=2, format="WAV", subtype="PCM_16"), id="wav-48k-stereo"),
        pytest.param(encoded(48000, format="OGG", subtype="OPUS"), id="opus-48k"),
        pytest.param(encoded(44100, format="FLAC"), id="flac-44.1k"),
    ],
)
def test_decodes_the_same_samples_as_faster_whisper(data: bytes) -> None:
    expected = faster_whisper_decode_audio(io.BytesIO(data), sampling_rate=16000)

    decoded = decode_audio(io.BytesIO(data), sampling_rate=16000)

    assert decoded.dtype == np.float32
    np.testing.assert_array_equal(decoded, expected)


def test_a_file_without_samples_decodes_to_nothing() -> None:
    header_only = encoded(16000, format="WAV", subtype="PCM_16")[:44]

    decoded = decode_audio(io.BytesIO(header_only))

    assert (decoded.dtype, len(decoded)) == (np.float32, 0)


def test_data_that_isnt_audio_is_rejected() -> None:
    with pytest.raises(av.error.InvalidDataError):
        decode_audio(io.BytesIO(b"not audio" * 100))


def test_loading_the_app_doesnt_import_pyannote() -> None:
    # pyannote pulls in lightning; loaded at startup, it doubles the objects every full garbage collection walks
    code = (
        "import sys, speaches.main; print(sorted({m.split('.')[0] for m in sys.modules} & {'pyannote', 'lightning'}))"
    )

    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)

    assert result.stdout.strip() == "[]"
