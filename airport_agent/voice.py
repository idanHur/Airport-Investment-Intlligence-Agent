"""Voice input: speech -> text. The text then goes through the same agent as typed questions,
so voice adds no scoring or agent logic."""
import os

from dotenv import load_dotenv
from openai import OpenAI

from airport_agent.config import TRANSCRIBE_MODEL

# Words the recogniser tends to mishear. Passed as a hint, not a constraint.
VOCABULARY_HINT = (
    "Questions about US airports and airlines. Airport codes: SFO, LAX, SNA, BUR, LGB, ONT, BOS, PVD, BDL, "
    "BGR, PWM, MHT, BTV, ANC, JFK, ORD, ATL. Terms: terminal expansion, load factor, congestion, long haul, "
    "unmet demand, New England, Santa Ana, Anchorage."
)


def transcribe(audio_bytes: bytes, filename: str = "question.wav", client: OpenAI | None = None) -> str:
    """Return the transcript of a short recording. Raises RuntimeError with a readable message on failure."""
    if client is None:
        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set, so voice input is unavailable.")
        client = OpenAI()
    try:
        result = client.audio.transcriptions.create(
            model=TRANSCRIBE_MODEL, file=(filename, audio_bytes), prompt=VOCABULARY_HINT, language="en",
        )
    except Exception as error:
        raise RuntimeError(f"Transcription failed: {error}") from error
    text = result.text.strip()
    if not text:
        raise RuntimeError("No speech was recognised. Please try again.")
    return text
