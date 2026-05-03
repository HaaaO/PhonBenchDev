"""OpenAI audio client for transcription-style benchmark calls."""

from __future__ import annotations

import base64
import json
import os
import random
import time
import wave
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torchaudio


class OpenAIAudioClient:
    """Small wrapper around OpenAI Chat Completions audio input."""

    TOOL_NAME = "submit_transcription"

    def __init__(
        self,
        model_name: str = "gpt-audio-1.5",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 256,
        timeout: float = 600.0,
        response_schema: Optional[dict[str, Any]] = None,
        retry_config: Optional[dict[str, Any]] = None,
        client: Optional[Any] = None,
    ) -> None:
        key = api_key or os.getenv("OPENAI_API_KEY")
        if client is None and not key:
            raise ValueError(
                "OpenAI API key is not configured. Provide api_key or set OPENAI_API_KEY."
            )

        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.response_schema = self._build_transcription_schema(response_schema)
        default_retry = {"max_retries": 3, "initial_delay": 1.0, "backoff_factor": 2.0}
        self.retry_config = {**default_retry, **(retry_config or {})}

        if client is not None:
            self.client = client
        else:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ImportError(
                    "The OpenAI Python package is required for GPT-audio inference. "
                    "Install dependencies from requirements.txt."
                ) from e
            self.client = OpenAI(api_key=key, timeout=timeout)

    @staticmethod
    def _to_mono_float32(audio: Any) -> np.ndarray:
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 2:
            if audio.shape[0] <= audio.shape[1]:
                audio = audio.mean(axis=0)
            else:
                audio = audio.mean(axis=1)
        if audio.ndim != 1:
            raise ValueError(f"Expected mono audio with shape (T,), got {audio.shape}")
        return np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    @classmethod
    def _audio_to_base64_wav(cls, audio: Any, sample_rate: int) -> str:
        audio = cls._to_mono_float32(audio)
        audio = np.clip(audio, -1.0, 1.0)
        pcm16 = (audio * 32767.0).astype("<i2")

        buffer = BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(int(sample_rate))
            wav.writeframes(pcm16.tobytes())
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    @staticmethod
    def _load_audio(path: str | Path) -> tuple[torch.Tensor, int]:
        waveform, sr = torchaudio.load(str(path))
        if waveform.ndim == 2 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.squeeze(0)
        return waveform.to(torch.float32), int(sr)

    @classmethod
    def audio_input_from_sample(
        cls,
        *,
        speech: Optional[Any] = None,
        sample_rate: int = 16000,
        audio_path: Optional[str | Path] = None,
        wavpath: Optional[str | Path] = None,
    ) -> dict[str, str]:
        if speech is not None:
            return {
                "data": cls._audio_to_base64_wav(speech, sample_rate),
                "format": "wav",
            }

        path = audio_path or wavpath
        if path is None:
            raise ValueError("OpenAI audio inference requires speech, audio_path, or wavpath.")

        waveform, sr = cls._load_audio(path)
        return {
            "data": cls._audio_to_base64_wav(waveform, sr),
            "format": "wav",
        }

    @staticmethod
    def _default_transcription_schema() -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["transcription"],
            "properties": {
                "transcription": {
                    "type": "string",
                    "description": "Exactly one IPA transcription string.",
                }
            },
        }

    @classmethod
    def _normalise_schema_types(cls, schema: Any) -> Any:
        if isinstance(schema, dict):
            normalised = {}
            for key, value in schema.items():
                if key == "type" and isinstance(value, str):
                    normalised[key] = value.lower()
                else:
                    normalised[key] = cls._normalise_schema_types(value)
            return normalised
        if isinstance(schema, list):
            return [cls._normalise_schema_types(item) for item in schema]
        return schema

    @classmethod
    def _build_transcription_schema(
        cls, response_schema: Optional[dict[str, Any]]
    ) -> dict[str, Any]:
        if not response_schema:
            return cls._default_transcription_schema()

        schema = cls._normalise_schema_types(response_schema)
        properties = schema.get("properties") if isinstance(schema, dict) else None
        if not isinstance(properties, dict) or "transcription" not in properties:
            return cls._default_transcription_schema()

        required = schema.get("required") or []
        if "transcription" not in required:
            required = [*required, "transcription"]
        schema["required"] = required
        return schema

    def transcription_tool(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": OpenAIAudioClient.TOOL_NAME,
                    "description": "Submit the IPA transcription for the audio clip.",
                    "parameters": self.response_schema,
                },
            }
        ]

    @staticmethod
    def _extract_text_response(message: Any) -> str:
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    parts.append(str(part.get("text", "")))
                else:
                    parts.append(str(getattr(part, "text", "")))
            return "".join(parts).strip()
        return str(content or "").strip()

    @classmethod
    def _extract_transcription(cls, message: Any) -> tuple[str, str]:
        tool_calls = getattr(message, "tool_calls", None) or []
        for call in tool_calls:
            fn = getattr(call, "function", None)
            name = getattr(fn, "name", None)
            if name != cls.TOOL_NAME:
                continue
            args = getattr(fn, "arguments", "{}") or "{}"
            parsed = json.loads(args)
            return str(parsed.get("transcription", "")), json.dumps(parsed, ensure_ascii=False)

        raw_text = cls._extract_text_response(message)
        if "{" in raw_text and "}" in raw_text:
            start = raw_text.find("{")
            end = raw_text.rfind("}") + 1
            try:
                parsed = json.loads(raw_text[start:end])
                if isinstance(parsed, dict) and "transcription" in parsed:
                    return str(parsed["transcription"]), raw_text
            except json.JSONDecodeError:
                pass
        return raw_text, raw_text

    @staticmethod
    def _is_retryable_error(e: Exception) -> bool:
        status = getattr(e, "status_code", None) or getattr(e, "status", None)
        if status in {408, 409, 429, 500, 502, 503, 504}:
            return True
        return isinstance(e, (TimeoutError, ConnectionError, OSError))

    def generate_transcription(
        self,
        *,
        audio_input: dict[str, str],
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> tuple[str, str]:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "input_audio", "input_audio": audio_input},
                    {"type": "text", "text": prompt},
                ],
            }
        )

        last_error: Optional[Exception] = None
        delay = float(self.retry_config["initial_delay"])
        max_retries = int(self.retry_config["max_retries"])
        backoff = float(self.retry_config["backoff_factor"])

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    modalities=["text"],
                    tools=self.transcription_tool(),
                    tool_choice={
                        "type": "function",
                        "function": {"name": self.TOOL_NAME},
                    },
                    parallel_tool_calls=False,
                    n=1,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                )
                message = response.choices[0].message
                return self._extract_transcription(message)
            except Exception as e:
                last_error = e
                if not self._is_retryable_error(e) or attempt >= max_retries - 1:
                    raise
                jitter = random.uniform(0.0, min(0.25, delay * 0.1))
                time.sleep(delay + jitter)
                delay *= backoff

        raise RuntimeError(f"Failed after {max_retries} attempts") from last_error
