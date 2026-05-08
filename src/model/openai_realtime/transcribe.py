"""GPT Realtime inference wrapper for distributed phone transcription."""

from __future__ import annotations

import json
import os
import string
import unicodedata
from pathlib import Path
from typing import Any, Optional

from src.model.openai_realtime.client import OpenAIRealtimeTranscriptionClient


class GptRealtimeInference:
    """Distributed-inference-compatible GPT Realtime transcription wrapper."""

    def __init__(
        self,
        client_config: dict[str, Any],
        prompt_config: dict[str, Any],
        clean_response: bool = False,
        output_key: Optional[str] = None,
        device: Optional[str] = None,
        cache_path: Optional[str | Path] = None,
        resume: bool = True,
        cache_key_field: str = "utt_id",
        error_log_path: Optional[str | Path] = None,
    ) -> None:
        self.client = OpenAIRealtimeTranscriptionClient(**client_config)
        self.system_prompt = prompt_config.get("system_prompt", "")
        self.user_prompt = prompt_config.get("user_prompt", "")
        self.clean_response = clean_response
        self.output_key = output_key
        self.cache_path = Path(cache_path) if cache_path else None
        self.error_log_path = Path(error_log_path) if error_log_path else None
        self.resume = resume
        self.cache_key_field = cache_key_field
        self._cache: dict[str, Any] = {}
        if self.cache_path and self.resume:
            self._load_cache()

    def _load_cache(self) -> None:
        assert self.cache_path is not None
        if not self.cache_path.exists():
            return
        try:
            with self.cache_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    key = rec.get("key")
                    pred = rec.get("pred")
                    if key is not None and pred is not None:
                        self._cache[str(key)] = pred
        except Exception:
            return

    @staticmethod
    def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, default=str)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            try:
                f.flush()
                os.fsync(f.fileno())
            except OSError:
                pass

    @staticmethod
    def _clean_response(text: str) -> str:
        text = "".join(str(text).split())
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = unicodedata.normalize("NFD", text)
        text = text.replace("g", "ɡ")
        text = text.replace("ɚ", "ə˞").replace("ɝ", "ɜ˞")
        return text.strip()

    @staticmethod
    def _is_error_pred(pred: Any) -> bool:
        if not isinstance(pred, list) or not pred:
            return True
        first = pred[0]
        return isinstance(first, dict) and "error" in first

    @staticmethod
    def _render_prompt(template: str, values: dict[str, Any]) -> str:
        if not template:
            return template
        rendered_values = {
            key: "" if value is None else value for key, value in values.items()
        }
        try:
            return template.format_map(rendered_values)
        except KeyError as e:
            missing_key = e.args[0]
            raise ValueError(
                f"Prompt template requires missing field: {missing_key}"
            ) from e

    def _cache_key(self, kwargs: dict[str, Any]) -> str:
        if self.cache_key_field in kwargs:
            return str(kwargs[self.cache_key_field])
        return str(
            kwargs.get("utt_id")
            or kwargs.get("audio_path")
            or kwargs.get("wavpath")
            or kwargs.get("metadata_idx")
        )

    def __call__(self, **kwargs: Any) -> Any:
        cache_key = self._cache_key(kwargs)
        if self.cache_path and self.resume and cache_key in self._cache:
            return self._cache[cache_key]

        try:
            user_prompt = self._render_prompt(self.user_prompt, kwargs)
            system_prompt = self._render_prompt(self.system_prompt, kwargs)
            sample_rate = int(kwargs.get("sample_rate") or kwargs.get("sr") or 16000)
            audio_input = self.client.audio_input_from_sample(
                speech=kwargs.get("speech"),
                sample_rate=sample_rate,
                audio_path=kwargs.get("audio_path"),
                wavpath=kwargs.get("wavpath"),
            )
            raw_transcript, raw_model_response = self.client.generate_transcription(
                audio_input=audio_input,
                prompt=user_prompt,
                system_prompt=system_prompt if system_prompt else None,
            )
        except Exception as e:
            err = {
                "type": type(e).__name__,
                "code": getattr(e, "code", None),
                "status": getattr(e, "status", None) or getattr(e, "status_code", None),
                "message": str(e),
            }
            pred = [
                {
                    "processed_transcript": "",
                    "predicted_transcript": "",
                    "raw_model_response": "",
                    "error": err,
                }
            ]
            if self.error_log_path:
                self._append_jsonl(
                    self.error_log_path,
                    {
                        "key": cache_key,
                        "audio_path": str(
                            kwargs.get("audio_path") or kwargs.get("wavpath") or ""
                        ),
                        "error": err,
                    },
                )
            return pred

        processed_transcript = (
            self._clean_response(raw_transcript) if self.clean_response else raw_transcript
        )
        pred = [
            {
                "processed_transcript": processed_transcript,
                "predicted_transcript": raw_transcript,
                "raw_model_response": raw_model_response,
            }
        ]
        if self.cache_path and cache_key and not self._is_error_pred(pred):
            self._cache[cache_key] = pred
            self._append_jsonl(self.cache_path, {"key": cache_key, "pred": pred})
        return pred
