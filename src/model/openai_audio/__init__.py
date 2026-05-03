"""OpenAI audio model integration for PRiSM."""

from src.model.openai_audio.client import OpenAIAudioClient
from src.model.openai_audio.transcribe import GptAudioInference

__all__ = ["GptAudioInference", "OpenAIAudioClient"]
