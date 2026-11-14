import math

from onnx_asr.asr import TimestampedResult

from speaches.executors.parakeet import result_to_verbose_transcription, result_to_words

# tokens, start times and log-probabilities as `istupakov/parakeet-tdt-0.6b-v2-onnx` returned them for a 2.28 s clip
RESULT = TimestampedResult(
    text="Chloe, what time does the game start tonight?",
    tokens=[
        " Ch",
        "l",
        "o",
        "e",
        ",",
        " what",
        " time",
        " does",
        " the",
        " g",
        "ame",
        " start",
        " t",
        "on",
        "ight",
        "?",
    ],
    timestamps=[0.0, 0.08, 0.24, 0.4, 0.56, 0.64, 0.8, 1.04, 1.2, 1.28, 1.44, 1.6, 1.84, 1.92, 2.0, 2.16],
    logprobs=[-0.034, -0.001, 0.0, 0.0, -0.002, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.001, 0.0, 0.0, 0.0],
)
DURATION = 2.28


def test_tokens_are_grouped_into_words_with_punctuation_attached() -> None:
    words = result_to_words(RESULT, DURATION)

    assert [w.word for w in words] == [
        " Chloe,",
        " what",
        " time",
        " does",
        " the",
        " game",
        " start",
        " tonight?",
    ]


def test_a_word_ends_where_the_next_begins_and_the_last_one_frame_after_its_last_spoken_token() -> None:
    words = result_to_words(RESULT, DURATION)

    assert (words[0].start, words[0].end) == (0.0, 0.64)
    assert (words[5].start, words[5].end) == (1.28, 1.6)
    assert words[-1].start == 1.84
    # 'ight' starts at 2.0; the '?' (2.16) is punctuation and doesn't extend the word
    assert math.isclose(words[-1].end, 2.08)


def test_the_last_word_never_ends_after_the_audio() -> None:
    words = result_to_words(RESULT, duration=2.05)

    assert words[-1].end == 2.05


def test_a_word_carries_the_mean_probability_of_its_tokens() -> None:
    words = result_to_words(RESULT, DURATION)

    expected = sum(math.exp(lp) for lp in [-0.034, -0.001, 0.0, 0.0, -0.002]) / 5
    assert math.isclose(words[0].probability, expected)  # pyrefly: ignore[missing-attribute]


def test_without_logprobs_a_word_has_no_probability() -> None:
    result = TimestampedResult(text=RESULT.text, tokens=RESULT.tokens, timestamps=RESULT.timestamps)

    words = result_to_words(result, DURATION)

    assert len(words) == 8
    assert "probability" not in words[0].model_dump()


def test_verbose_json_has_one_segment_and_words_only_when_asked() -> None:
    with_words = result_to_verbose_transcription(RESULT, duration=DURATION, language=None, word_timestamps=True)
    without_words = result_to_verbose_transcription(RESULT, duration=DURATION, language="en", word_timestamps=False)

    assert with_words.text == RESULT.text
    assert with_words.duration == DURATION
    assert with_words.words is not None
    assert len(with_words.words) == 8
    assert with_words.segments is not None
    assert len(with_words.segments) == 1
    assert (with_words.segments[0].start, with_words.segments[0].end) == (0.0, with_words.words[-1].end)
    assert without_words.words is None
    assert without_words.language == "en"


def test_silence_gives_no_segments_and_no_words() -> None:
    result = TimestampedResult(text="", tokens=[], timestamps=[], logprobs=[])

    transcription = result_to_verbose_transcription(result, duration=1.0, language=None, word_timestamps=True)

    assert transcription.text == ""
    assert transcription.segments == []
    assert transcription.words == []


def test_a_word_before_a_pause_ends_where_speech_stops() -> None:
    result = TimestampedResult(
        text="Hey Chloe. Can",
        # the full stop is stamped inside the pause, as Parakeet does
        tokens=[" Hey", " Chloe", ".", " Can"],
        timestamps=[0.0, 0.24, 0.96, 1.28],
        logprobs=[0.0, 0.0, 0.0, 0.0],
    )

    words = result_to_words(result, duration=1.6, speech=[(0.0, 0.7), (1.2, 1.6)])

    assert [(w.word, w.start, w.end) for w in words] == [
        (" Hey", 0.0, 0.24),
        (" Chloe.", 0.24, 0.7),
        (" Can", 1.28, 1.36),
    ]
