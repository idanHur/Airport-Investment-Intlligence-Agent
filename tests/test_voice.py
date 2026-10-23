from types import SimpleNamespace

import pytest

from airport_agent import voice


class FakeClient:
    """Stands in for openai.OpenAI so no network or key is needed."""

    def __init__(self, text=None, error=None):
        self.calls = []
        self.audio = SimpleNamespace(transcriptions=SimpleNamespace(create=self._create))
        self._text, self._error = text, error

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        return SimpleNamespace(text=self._text)


def test_transcribe_returns_stripped_text_and_sends_vocabulary_hint():
    client = FakeClient(text="  Compare LAX and SNA  ")
    assert voice.transcribe(b"audio", client=client) == "Compare LAX and SNA"
    call = client.calls[0]
    assert call["prompt"] == voice.VOCABULARY_HINT and "SNA" in call["prompt"]
    assert call["file"] == ("question.wav", b"audio")


def test_empty_transcript_is_an_error():
    with pytest.raises(RuntimeError, match="No speech"):
        voice.transcribe(b"audio", client=FakeClient(text="   "))


def test_api_failure_becomes_readable_error():
    with pytest.raises(RuntimeError, match="Transcription failed: boom"):
        voice.transcribe(b"audio", client=FakeClient(error=ValueError("boom")))
