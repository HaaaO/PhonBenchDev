"""OpenAI Realtime WebSocket client for IPA transcription benchmarks."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import quote

import numpy as np
import torch
import torchaudio

from src.model.openai_audio.client import OpenAIAudioClient


class RealtimeAPIError(RuntimeError):
    """Error reported by the Realtime API."""

    def __init__(self, message: str, *, code: Any = None, status: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.status = status
        self.status_code = status


class OpenAIRealtimeTranscriptionClient:
    """Small wrapper around the OpenAI Realtime WebSocket API.

    The client intentionally opens a fresh Realtime session for each call to
    ``generate_transcription``. That keeps benchmark utterances independent.
    """

    TOOL_NAME = OpenAIAudioClient.TOOL_NAME
    CONVERSATION_ITEM_ACK_EVENTS = {
        "conversation.item.created",
        "conversation.item.added",
        "conversation.item.done",
    }

    def __init__(
        self,
        model_name: str = "gpt-realtime-2",
        api_key: Optional[str] = None,
        realtime_url: str = "wss://api.openai.com/v1/realtime",
        input_sample_rate: int = 24000,
        max_tokens: int | str = "inf",
        timeout: float = 600.0,
        connect_timeout: float = 30.0,
        response_schema: Optional[dict[str, Any]] = None,
        retry_config: Optional[dict[str, Any]] = None,
        extra_headers: Optional[dict[str, str]] = None,
        use_tool: bool = False,
        tool_choice: str | dict[str, Any] = "auto",
        audio_delivery_mode: str = "conversation_item",
        retry_no_audio_response: bool = True,
        retry_empty_response: bool = False,
        websocket_connect: Optional[Callable[..., Any]] = None,
    ) -> None:
        key = api_key or os.getenv("OPENAI_API_KEY")
        if websocket_connect is None and not key:
            raise ValueError(
                "OpenAI API key is not configured. Provide api_key or set OPENAI_API_KEY."
            )

        self.model_name = model_name
        self.api_key = key
        self.realtime_url = realtime_url.rstrip("?")
        self.input_sample_rate = int(input_sample_rate)
        self.max_tokens = max_tokens
        self.timeout = float(timeout)
        self.connect_timeout = float(connect_timeout)
        self.response_schema = OpenAIAudioClient._build_transcription_schema(
            response_schema
        )
        default_retry = {"max_retries": 3, "initial_delay": 1.0, "backoff_factor": 2.0}
        self.retry_config = {**default_retry, **(retry_config or {})}
        self.extra_headers = dict(extra_headers or {})
        self.use_tool = bool(use_tool)
        self.tool_choice = tool_choice
        self.audio_delivery_mode = str(audio_delivery_mode)
        if self.audio_delivery_mode not in {
            "conversation_item",
            "buffer",
            "response_input_tool",
        }:
            raise ValueError(
                "audio_delivery_mode must be 'conversation_item', 'buffer', "
                "or 'response_input_tool', "
                f"got {audio_delivery_mode!r}."
            )
        self.retry_no_audio_response = bool(retry_no_audio_response)
        self.retry_empty_response = bool(retry_empty_response)

        if websocket_connect is not None:
            self.websocket_connect = websocket_connect
        else:
            try:
                import websockets
            except ImportError as e:
                raise ImportError(
                    "The websockets package is required for OpenAI Realtime inference. "
                    "Install dependencies from requirements.txt."
                ) from e
            self.websocket_connect = websockets.connect

    @staticmethod
    def _to_mono_float32(audio: Any) -> np.ndarray:
        return OpenAIAudioClient._to_mono_float32(audio)

    @staticmethod
    def _load_audio(path: str | Path) -> tuple[torch.Tensor, int]:
        return OpenAIAudioClient._load_audio(path)

    @classmethod
    def _normalise_schema_types(cls, schema: Any) -> Any:
        return OpenAIAudioClient._normalise_schema_types(schema)

    def _audio_to_base64_pcm16(self, audio: Any, sample_rate: int) -> str:
        audio_np = self._to_mono_float32(audio)
        if int(sample_rate) != self.input_sample_rate:
            audio_tensor = torch.from_numpy(audio_np).to(torch.float32)
            audio_tensor = torchaudio.functional.resample(
                audio_tensor,
                orig_freq=int(sample_rate),
                new_freq=self.input_sample_rate,
            )
            audio_np = audio_tensor.detach().cpu().numpy()
        audio_np = np.clip(audio_np, -1.0, 1.0)
        pcm16 = (audio_np * 32767.0).astype("<i2")
        return base64.b64encode(pcm16.tobytes()).decode("ascii")

    def audio_input_from_sample(
        self,
        *,
        speech: Optional[Any] = None,
        sample_rate: int = 16000,
        audio_path: Optional[str | Path] = None,
        wavpath: Optional[str | Path] = None,
    ) -> dict[str, Any]:
        if speech is not None:
            return {
                "data": self._audio_to_base64_pcm16(speech, sample_rate),
                "format": {"type": "audio/pcm", "rate": self.input_sample_rate},
            }

        path = audio_path or wavpath
        if path is None:
            raise ValueError(
                "OpenAI Realtime inference requires speech, audio_path, or wavpath."
            )

        waveform, sr = self._load_audio(path)
        return {
            "data": self._audio_to_base64_pcm16(waveform, sr),
            "format": {"type": "audio/pcm", "rate": self.input_sample_rate},
        }

    def transcription_tool(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": self.TOOL_NAME,
                "description": "Submit exactly one IPA transcription string for the audio clip.",
                "parameters": self.response_schema,
            }
        ]

    def _websocket_url(self) -> str:
        return f"{self.realtime_url}?model={quote(self.model_name, safe='')}"

    def _headers(self) -> dict[str, str]:
        headers = dict(self.extra_headers)
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _connect(self) -> Any:
        return self.websocket_connect(
            self._websocket_url(),
            additional_headers=self._headers(),
            open_timeout=self.connect_timeout,
            max_size=None,
        )

    @staticmethod
    async def _send_json(ws: Any, payload: dict[str, Any]) -> None:
        await ws.send(json.dumps(payload, ensure_ascii=False))

    @staticmethod
    async def _recv_json(ws: Any, timeout: float) -> dict[str, Any]:
        raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw)

    @staticmethod
    def _is_error_event(event: dict[str, Any]) -> bool:
        event_type = str(event.get("type") or "")
        if event_type == "error" or event_type.endswith("_error"):
            return True
        return "message" in event and ("code" in event or "param" in event)

    @staticmethod
    def _raise_for_error_event(event: dict[str, Any]) -> None:
        err = event.get("error") if isinstance(event.get("error"), dict) else event
        message = str(err.get("message") or event)
        raise RealtimeAPIError(
            message,
            code=err.get("code") or err.get("type"),
            status=err.get("status") or err.get("status_code"),
        )

    async def _recv_until(
        self, ws: Any, wanted_types: set[str], events: list[dict[str, Any]]
    ) -> dict[str, Any]:
        while True:
            event = await self._recv_json(ws, self.timeout)
            events.append(event)
            event_type = event.get("type")
            if self._is_error_event(event):
                self._raise_for_error_event(event)
            if event_type in wanted_types:
                return event

    async def _recv_until_all(
        self, ws: Any, wanted_types: set[str], events: list[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        seen: dict[str, dict[str, Any]] = {}
        while wanted_types - set(seen):
            event = await self._recv_json(ws, self.timeout)
            events.append(event)
            event_type = event.get("type")
            if self._is_error_event(event):
                self._raise_for_error_event(event)
            if event_type in wanted_types:
                seen[str(event_type)] = event
        return seen

    def _session_update_event(
        self,
        system_prompt: Optional[str],
        *,
        include_tools: bool = True,
    ) -> dict[str, Any]:
        session: dict[str, Any] = {
            "type": "realtime",
            "instructions": system_prompt or "",
            "output_modalities": ["text"],
            "audio": {
                "input": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": self.input_sample_rate,
                    },
                    "turn_detection": None,
                }
            },
        }
        if include_tools and self.use_tool:
            session["tools"] = self.transcription_tool()
            session["tool_choice"] = self.tool_choice
        else:
            session["tool_choice"] = "none"
        return {"type": "session.update", "session": session}

    @staticmethod
    def _response_instructions(prompt: str, system_prompt: Optional[str]) -> str:
        parts = []
        if system_prompt:
            parts.append(system_prompt.strip())
        if prompt:
            parts.append(f"User request:\n{prompt.strip()}")
        return "\n\n".join(part for part in parts if part)

    def _response_create_event(
        self,
        *,
        prompt: str,
        system_prompt: Optional[str],
    ) -> dict[str, Any]:
        response: dict[str, Any] = {"output_modalities": ["text"]}
        if self.max_tokens is not None:
            response["max_output_tokens"] = self.max_tokens
        instructions = self._response_instructions(prompt, system_prompt)
        if instructions:
            response["instructions"] = instructions
        return {"type": "response.create", "response": response}

    def _response_input_tool_create_event(
        self,
        *,
        audio_input: dict[str, Any],
        prompt: str,
        system_prompt: Optional[str],
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        if prompt:
            content.append({"type": "input_text", "text": prompt})
        content.append({"type": "input_audio", "audio": audio_input["data"]})

        response: dict[str, Any] = {
            "conversation": "none",
            "output_modalities": ["text"],
            "tools": self.transcription_tool(),
            "tool_choice": {"type": "function", "name": self.TOOL_NAME},
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": content,
                }
            ],
        }
        if self.max_tokens is not None:
            response["max_output_tokens"] = self.max_tokens
        if system_prompt:
            response["instructions"] = system_prompt.strip()
        return {"type": "response.create", "response": response}

    @staticmethod
    def _conversation_item_create_event(
        *,
        audio_input: dict[str, Any],
        prompt: str,
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = [
            {
                "type": "input_audio",
                "audio": audio_input["data"],
            }
        ]
        if prompt:
            content.append({"type": "input_text", "text": prompt})
        return {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": content,
            },
        }

    async def _send_audio_and_prompt(
        self,
        ws: Any,
        *,
        audio_input: dict[str, Any],
        prompt: str,
        events: list[dict[str, Any]],
    ) -> None:
        if self.audio_delivery_mode == "conversation_item":
            await self._send_json(
                ws,
                self._conversation_item_create_event(
                    audio_input=audio_input,
                    prompt=prompt,
                ),
            )
            await self._recv_until(ws, self.CONVERSATION_ITEM_ACK_EVENTS, events)
            return

        await self._send_json(ws, {"type": "input_audio_buffer.clear"})
        await self._recv_until(ws, {"input_audio_buffer.cleared"}, events)
        await self._send_json(
            ws,
            {
                "type": "input_audio_buffer.append",
                "audio": audio_input["data"],
            },
        )
        await self._send_json(ws, {"type": "input_audio_buffer.commit"})
        await self._recv_until_all(
            ws,
            {"input_audio_buffer.committed"},
            events,
        )
        await self._recv_until(ws, self.CONVERSATION_ITEM_ACK_EVENTS, events)

    @staticmethod
    def _text_from_response_item(item: dict[str, Any]) -> str:
        content = item.get("content") or []
        parts = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("text"):
                parts.append(str(part["text"]))
            elif part.get("transcript"):
                parts.append(str(part["transcript"]))
        return "".join(parts).strip()

    @classmethod
    def _extract_text_transcription(cls, raw_text: str) -> tuple[str, str]:
        raw_text = str(raw_text or "").strip()
        if "{" in raw_text and "}" in raw_text:
            start = raw_text.find("{")
            end = raw_text.rfind("}") + 1
            try:
                parsed = json.loads(raw_text[start:end])
                if isinstance(parsed, dict) and "transcription" in parsed:
                    return str(parsed["transcription"]), raw_text
            except json.JSONDecodeError:
                pass

        field_match = re.search(
            r'^\s*"?transcription"?\s*:\s*(?P<value>"[^"]*"|\'[^\']*\'|.*)\s*$',
            raw_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if field_match:
            value = field_match.group("value").strip()
            if (value.startswith('"') and value.endswith('"')) or (
                value.startswith("'") and value.endswith("'")
            ):
                value = value[1:-1]
            return value.strip(), raw_text
        return raw_text, raw_text

    def _extract_transcription_from_events(
        self,
        events: list[dict[str, Any]],
        response_done: Optional[dict[str, Any]],
    ) -> tuple[str, str]:
        function_args: Optional[str] = None
        output_text_done: Optional[str] = None
        output_text_delta: list[str] = []
        item_texts: list[str] = []

        for event in events:
            event_type = event.get("type")
            if (
                event_type == "response.function_call_arguments.done"
                and event.get("name") == self.TOOL_NAME
            ):
                function_args = str(event.get("arguments") or "")
            elif event_type == "response.output_text.done":
                output_text_done = str(event.get("text") or "")
            elif event_type == "response.output_text.delta":
                output_text_delta.append(str(event.get("delta") or ""))
            elif event_type == "response.output_item.done":
                item = event.get("item") if isinstance(event.get("item"), dict) else {}
                if item.get("type") == "function_call" and item.get("name") == self.TOOL_NAME:
                    function_args = str(item.get("arguments") or function_args or "")
                elif item.get("type") == "message":
                    item_text = self._text_from_response_item(item)
                    if item_text:
                        item_texts.append(item_text)

        response = response_done.get("response", {}) if response_done else {}
        response_output = response.get("output") or []
        for item in response_output:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "function_call" and item.get("name") == self.TOOL_NAME:
                function_args = str(item.get("arguments") or function_args or "")
            elif item.get("type") == "message":
                item_text = self._text_from_response_item(item)
                if item_text:
                    item_texts.append(item_text)

        source = "empty"
        raw_value = ""
        transcript = ""

        if function_args:
            source = "function_call"
            raw_value = function_args
            try:
                parsed = json.loads(function_args)
            except json.JSONDecodeError:
                parsed = {}
            if isinstance(parsed, dict):
                transcript = str(parsed.get("transcription", ""))
            else:
                transcript = str(parsed)
        else:
            source = "output_text"
            raw_value = (
                output_text_done
                if output_text_done is not None
                else "".join(output_text_delta) or "".join(item_texts)
            )
            transcript, raw_value = self._extract_text_transcription(raw_value)

        raw_payload = {
            "transcription": transcript,
            "source": source,
            "raw": raw_value,
            "model": self.model_name,
            "transport": "websocket",
            "session_id": self._first_event_id(events, "session.created"),
            "response_id": response.get("id"),
            "response_status": response.get("status"),
        }
        return transcript, json.dumps(raw_payload, ensure_ascii=False, default=str)

    @staticmethod
    def _first_event_id(events: list[dict[str, Any]], event_type: str) -> Optional[str]:
        for event in events:
            if event.get("type") == event_type:
                session = event.get("session") if isinstance(event.get("session"), dict) else {}
                return session.get("id") or event.get("event_id")
        return None

    async def _generate_transcription_once(
        self,
        *,
        audio_input: dict[str, Any],
        prompt: str,
        system_prompt: Optional[str],
    ) -> tuple[str, str]:
        events: list[dict[str, Any]] = []
        async with self._connect() as ws:
            await self._recv_until(ws, {"session.created"}, events)
            is_response_input_tool = self.audio_delivery_mode == "response_input_tool"
            await self._send_json(
                ws,
                self._session_update_event(
                    None if is_response_input_tool else system_prompt,
                    include_tools=not is_response_input_tool,
                ),
            )
            await self._recv_until(ws, {"session.updated"}, events)
            if is_response_input_tool:
                await self._send_json(
                    ws,
                    self._response_input_tool_create_event(
                        audio_input=audio_input,
                        prompt=prompt,
                        system_prompt=system_prompt,
                    ),
                )
            else:
                await self._send_audio_and_prompt(
                    ws,
                    audio_input=audio_input,
                    prompt=prompt,
                    events=events,
                )
                await self._send_json(
                    ws,
                    self._response_create_event(
                        prompt="" if self.audio_delivery_mode == "conversation_item" else prompt,
                        system_prompt=system_prompt,
                    ),
                )

            response_done: Optional[dict[str, Any]] = None
            while True:
                event = await self._recv_json(ws, self.timeout)
                events.append(event)
                if self._is_error_event(event):
                    self._raise_for_error_event(event)
                if event.get("type") == "response.done":
                    response_done = event
                    break

            response = response_done.get("response", {}) if response_done else {}
            status = response.get("status")
            if status and status != "completed":
                raise RealtimeAPIError(
                    f"Realtime response ended with status={status}: "
                    f"{response.get('status_details')}",
                    code=status,
                    status=status,
                )
            return self._extract_transcription_from_events(events, response_done)

    @staticmethod
    def _is_retryable_error(e: Exception) -> bool:
        status = getattr(e, "status_code", None) or getattr(e, "status", None)
        code = getattr(e, "code", None)
        if code in {"missing_audio_response", "empty_transcription"}:
            return True
        if status in {408, 409, 429, 500, 502, 503, 504}:
            return True
        return isinstance(e, (TimeoutError, asyncio.TimeoutError, ConnectionError, OSError))

    @staticmethod
    def _looks_like_missing_audio_response(transcript: str, raw_response: str) -> bool:
        text = f"{transcript}\n{raw_response}".lower()
        text = re.sub(r"\s+", " ", text)
        patterns = (
            r"\bno audio\b",
            r"\baudio (?:clip|file|input )?(?:was )?not provided\b",
            r"\bno playable audio\b",
            r"\bprovide (?:the |an )?audio\b",
            r"\battach (?:the |an )?audio\b",
            r"\bwithout hearing it\b",
            r"\bcannot transcribe provided content\b",
            r"\bcan't transcribe (?:it|the audio).*(?:without|because no)",
            r"\bcan’t transcribe (?:it|the audio).*(?:without|because no)",
        )
        return any(re.search(pattern, text) for pattern in patterns)

    def generate_transcription(
        self,
        *,
        audio_input: dict[str, Any],
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> tuple[str, str]:
        last_error: Optional[Exception] = None
        delay = float(self.retry_config["initial_delay"])
        max_retries = int(self.retry_config["max_retries"])
        backoff = float(self.retry_config["backoff_factor"])

        for attempt in range(max_retries):
            try:
                transcript, raw_response = asyncio.run(
                    self._generate_transcription_once(
                        audio_input=audio_input,
                        prompt=prompt,
                        system_prompt=system_prompt,
                    )
                )
                if self.retry_no_audio_response and self._looks_like_missing_audio_response(
                    transcript, raw_response
                ):
                    raise RealtimeAPIError(
                        "Realtime model responded as if no audio was provided.",
                        code="missing_audio_response",
                        status="missing_audio_response",
                    )
                if self.retry_empty_response and not str(transcript).strip():
                    raise RealtimeAPIError(
                        "Realtime model returned an empty transcription.",
                        code="empty_transcription",
                        status="empty_transcription",
                    )
                return transcript, raw_response
            except Exception as e:
                last_error = e
                if not self._is_retryable_error(e) or attempt >= max_retries - 1:
                    raise
                jitter = random.uniform(0.0, min(0.25, delay * 0.1))
                time.sleep(delay + jitter)
                delay *= backoff

        raise RuntimeError(f"Failed after {max_retries} attempts") from last_error
