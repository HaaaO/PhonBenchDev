import base64
import json
from types import SimpleNamespace

import pytest
import torch

from src.model.openai_realtime.client import OpenAIRealtimeTranscriptionClient
from src.model.openai_realtime.transcribe import GptRealtimeInference


class FakeWebSocket:
    def __init__(self, events):
        self.events = list(events)
        self.sent = []
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.closed = True

    async def send(self, payload):
        self.sent.append(json.loads(payload))

    async def recv(self):
        if not self.events:
            raise TimeoutError("no fake events left")
        return json.dumps(self.events.pop(0))


class FakeWebSocketFactory:
    def __init__(self, event_batches):
        self.event_batches = [list(batch) for batch in event_batches]
        self.calls = []
        self.websockets = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        ws = FakeWebSocket(self.event_batches.pop(0))
        self.websockets.append(ws)
        return ws


def _session_events(*response_events):
    return [
        {
            "type": "session.created",
            "event_id": "event_session",
            "session": {"id": "sess_123"},
        },
        {"type": "session.updated", "session": {"id": "sess_123"}},
        {
            "type": "conversation.item.added",
            "item": {"id": "item_audio", "type": "message", "role": "user"},
        },
        *response_events,
    ]


def _done(status="completed"):
    return {"type": "response.done", "response": {"id": "resp_1", "status": status}}


def _prompt_schema():
    return {
        "type": "OBJECT",
        "required": ["transcription"],
        "properties": {"transcription": {"type": "STRING"}},
    }


def _client(factory):
    return OpenAIRealtimeTranscriptionClient(
        api_key="test-key",
        model_name="gpt-realtime-2",
        websocket_connect=factory,
        response_schema=_prompt_schema(),
        use_tool=False,
        retry_config={"max_retries": 1},
    )


def test_client_uses_conversation_item_audio_and_prompt():
    factory = FakeWebSocketFactory(
        [
            _session_events(
                {
                    "type": "response.output_text.done",
                    "text": "g a!",
                },
                _done(),
            )
        ]
    )
    client = _client(factory)
    audio_input = client.audio_input_from_sample(speech=torch.zeros(160), sample_rate=16000)

    transcript, raw = client.generate_transcription(
        audio_input=audio_input,
        prompt="transcribe",
        system_prompt="system",
    )

    assert transcript == "g a!"
    assert json.loads(raw)["source"] == "output_text"
    assert factory.calls[0][0][0].endswith("?model=gpt-realtime-2")
    assert factory.calls[0][1]["additional_headers"]["Authorization"] == "Bearer test-key"

    sent = factory.websockets[0].sent
    session_update = sent[0]["session"]
    assert session_update["output_modalities"] == ["text"]
    assert session_update["audio"]["input"]["format"] == {
        "type": "audio/pcm",
        "rate": 24000,
    }
    assert session_update["audio"]["input"]["turn_detection"] is None
    assert session_update["tool_choice"] == "none"
    assert "tools" not in session_update

    assert sent[1]["type"] == "conversation.item.create"
    item = sent[1]["item"]
    assert item["role"] == "user"
    assert item["content"][0]["type"] == "input_audio"
    assert base64.b64decode(item["content"][0]["audio"])
    assert item["content"][1] == {"type": "input_text", "text": "transcribe"}
    assert sent[2]["type"] == "response.create"
    assert "system" in sent[2]["response"]["instructions"]
    assert "transcribe" not in sent[2]["response"]["instructions"]
    assert factory.websockets[0].closed


def test_client_buffer_mode_waits_for_audio_commit_before_response():
    factory = FakeWebSocketFactory(
        [
            [
                {
                    "type": "session.created",
                    "event_id": "event_session",
                    "session": {"id": "sess_123"},
                },
                {"type": "session.updated", "session": {"id": "sess_123"}},
                {"type": "input_audio_buffer.cleared"},
                {"type": "input_audio_buffer.committed", "item_id": "item_audio"},
                {
                    "type": "conversation.item.added",
                    "item": {"id": "item_audio", "type": "message", "role": "user"},
                },
                {
                    "type": "response.output_text.done",
                    "text": "t u",
                },
                _done(),
            ]
        ]
    )
    client = OpenAIRealtimeTranscriptionClient(
        api_key="test-key",
        model_name="gpt-realtime-2",
        websocket_connect=factory,
        response_schema=_prompt_schema(),
        use_tool=False,
        audio_delivery_mode="buffer",
        retry_config={"max_retries": 1},
    )
    audio_input = client.audio_input_from_sample(speech=torch.zeros(160), sample_rate=16000)

    transcript, _ = client.generate_transcription(
        audio_input=audio_input,
        prompt="transcribe",
        system_prompt="system",
    )

    sent = factory.websockets[0].sent
    assert transcript == "t u"
    assert sent[1] == {"type": "input_audio_buffer.clear"}
    assert sent[2]["type"] == "input_audio_buffer.append"
    assert base64.b64decode(sent[2]["audio"])
    assert sent[3] == {"type": "input_audio_buffer.commit"}
    assert sent[4]["type"] == "response.create"
    assert "system" in sent[4]["response"]["instructions"]
    assert "transcribe" in sent[4]["response"]["instructions"]


def test_client_response_input_tool_mode_forces_function_with_audio_input():
    factory = FakeWebSocketFactory(
        [
            [
                {
                    "type": "session.created",
                    "event_id": "event_session",
                    "session": {"id": "sess_123"},
                },
                {"type": "session.updated", "session": {"id": "sess_123"}},
                {
                    "type": "response.done",
                    "response": {
                        "id": "resp_1",
                        "status": "completed",
                        "output": [
                            {
                                "type": "function_call",
                                "name": OpenAIRealtimeTranscriptionClient.TOOL_NAME,
                                "arguments": json.dumps({"transcription": "t u"}),
                            }
                        ],
                    },
                },
            ]
        ]
    )
    client = OpenAIRealtimeTranscriptionClient(
        api_key="test-key",
        model_name="gpt-realtime-2",
        websocket_connect=factory,
        response_schema=_prompt_schema(),
        audio_delivery_mode="response_input_tool",
        retry_config={"max_retries": 1},
    )
    audio_input = client.audio_input_from_sample(speech=torch.zeros(160), sample_rate=16000)

    transcript, raw = client.generate_transcription(
        audio_input=audio_input,
        prompt="transcribe",
        system_prompt="system",
    )

    sent = factory.websockets[0].sent
    assert transcript == "t u"
    assert json.loads(raw)["source"] == "function_call"
    assert len(sent) == 2
    session_update = sent[0]["session"]
    assert session_update["instructions"] == ""
    assert session_update["tool_choice"] == "none"

    response = sent[1]["response"]
    assert sent[1]["type"] == "response.create"
    assert response["conversation"] == "none"
    assert response["instructions"] == "system"
    assert response["tool_choice"] == {
        "type": "function",
        "name": OpenAIRealtimeTranscriptionClient.TOOL_NAME,
    }
    assert response["tools"][0]["name"] == OpenAIRealtimeTranscriptionClient.TOOL_NAME
    user_item = response["input"][0]
    assert user_item["role"] == "user"
    assert user_item["content"][0] == {"type": "input_text", "text": "transcribe"}
    assert user_item["content"][1]["type"] == "input_audio"
    assert base64.b64decode(user_item["content"][1]["audio"])


def test_client_parses_function_call_from_response_done():
    factory = FakeWebSocketFactory(
        [
            _session_events(
                {
                    "type": "response.done",
                    "response": {
                        "id": "resp_1",
                        "status": "completed",
                        "output": [
                            {
                                "type": "function_call",
                                "name": OpenAIRealtimeTranscriptionClient.TOOL_NAME,
                                "arguments": json.dumps({"transcription": "d ɔ g"}),
                            }
                        ],
                    },
                }
            )
        ]
    )
    client = _client(factory)
    audio_input = client.audio_input_from_sample(speech=torch.zeros(160), sample_rate=16000)

    transcript, raw = client.generate_transcription(
        audio_input=audio_input,
        prompt="transcribe",
    )

    assert transcript == "d ɔ g"
    assert json.loads(raw)["source"] == "function_call"


def test_client_text_response_fallback_is_parsed():
    factory = FakeWebSocketFactory(
        [
            _session_events(
                {
                    "type": "response.output_text.done",
                    "text": 'prefix {"transcription": "k ae t"} suffix',
                },
                _done(),
            )
        ]
    )
    client = _client(factory)
    audio_input = client.audio_input_from_sample(speech=torch.zeros(160), sample_rate=16000)

    transcript, raw = client.generate_transcription(
        audio_input=audio_input,
        prompt="transcribe",
    )

    assert transcript == "k ae t"
    parsed = json.loads(raw)
    assert parsed["source"] == "output_text"
    assert parsed["raw"] == 'prefix {"transcription": "k ae t"} suffix'


def test_client_bare_transcription_field_fallback_is_parsed():
    factory = FakeWebSocketFactory(
        [
            _session_events(
                {
                    "type": "response.output_text.done",
                    "text": '"transcription":"θɪŋk əv teɪst æz tʌtʃ"',
                },
                _done(),
            )
        ]
    )
    client = _client(factory)
    audio_input = client.audio_input_from_sample(speech=torch.zeros(160), sample_rate=16000)

    transcript, raw = client.generate_transcription(
        audio_input=audio_input,
        prompt="transcribe",
    )

    assert transcript == "θɪŋk əv teɪst æz tʌtʃ"
    parsed = json.loads(raw)
    assert parsed["source"] == "output_text"
    assert parsed["raw"] == '"transcription":"θɪŋk əv teɪst æz tʌtʃ"'


def test_client_retries_no_audio_text_response():
    factory = FakeWebSocketFactory(
        [
            _session_events(
                {
                    "type": "response.output_text.done",
                    "text": "No audio provided",
                },
                _done(),
            ),
            _session_events(
                {
                    "type": "response.output_text.done",
                    "text": "t u",
                },
                _done(),
            ),
        ]
    )
    client = OpenAIRealtimeTranscriptionClient(
        api_key="test-key",
        model_name="gpt-realtime-2",
        websocket_connect=factory,
        response_schema=_prompt_schema(),
        use_tool=False,
        retry_config={"max_retries": 2, "initial_delay": 0.0, "backoff_factor": 1.0},
    )
    audio_input = client.audio_input_from_sample(speech=torch.zeros(160), sample_rate=16000)

    transcript, _ = client.generate_transcription(
        audio_input=audio_input,
        prompt="transcribe",
    )

    assert transcript == "t u"
    assert len(factory.calls) == 2


def test_client_retries_empty_text_response_when_enabled():
    factory = FakeWebSocketFactory(
        [
            _session_events(
                {
                    "type": "response.output_text.done",
                    "text": 'transcription: ""',
                },
                _done(),
            ),
            _session_events(
                {
                    "type": "response.output_text.done",
                    "text": "t u",
                },
                _done(),
            ),
        ]
    )
    client = OpenAIRealtimeTranscriptionClient(
        api_key="test-key",
        model_name="gpt-realtime-2",
        websocket_connect=factory,
        response_schema=_prompt_schema(),
        use_tool=False,
        retry_empty_response=True,
        retry_config={"max_retries": 2, "initial_delay": 0.0, "backoff_factor": 1.0},
    )
    audio_input = client.audio_input_from_sample(speech=torch.zeros(160), sample_rate=16000)

    transcript, _ = client.generate_transcription(
        audio_input=audio_input,
        prompt="transcribe",
    )

    assert transcript == "t u"
    assert len(factory.calls) == 2


def test_client_opens_fresh_connection_for_each_utterance():
    factory = FakeWebSocketFactory(
        [
            _session_events(
                {
                    "type": "response.function_call_arguments.done",
                    "name": OpenAIRealtimeTranscriptionClient.TOOL_NAME,
                    "arguments": json.dumps({"transcription": "a"}),
                },
                _done(),
            ),
            _session_events(
                {
                    "type": "response.function_call_arguments.done",
                    "name": OpenAIRealtimeTranscriptionClient.TOOL_NAME,
                    "arguments": json.dumps({"transcription": "b"}),
                },
                _done(),
            ),
        ]
    )
    client = _client(factory)
    audio_input = client.audio_input_from_sample(speech=torch.zeros(160), sample_rate=16000)

    first, _ = client.generate_transcription(audio_input=audio_input, prompt="utt one")
    second, _ = client.generate_transcription(audio_input=audio_input, prompt="utt two")

    assert (first, second) == ("a", "b")
    assert len(factory.calls) == 2
    assert len(factory.websockets) == 2
    assert factory.websockets[0].sent[1]["item"]["content"][1]["text"] == "utt one"
    assert factory.websockets[1].sent[1]["item"]["content"][1]["text"] == "utt two"


def test_inference_returns_existing_prediction_schema(tmp_path):
    factory = FakeWebSocketFactory(
        [
            _session_events(
                {
                    "type": "response.function_call_arguments.done",
                    "name": OpenAIRealtimeTranscriptionClient.TOOL_NAME,
                    "arguments": json.dumps({"transcription": "g a!"}),
                },
                _done(),
            )
        ]
    )
    inf = GptRealtimeInference(
        client_config={
            "api_key": "test-key",
            "websocket_connect": factory,
            "response_schema": _prompt_schema(),
            "use_tool": False,
            "retry_config": {"max_retries": 1},
        },
        prompt_config={"system_prompt": "system", "user_prompt": "transcribe"},
        clean_response=True,
        output_key="transcription",
        cache_path=tmp_path / "cache.jsonl",
        error_log_path=tmp_path / "errors.jsonl",
    )

    pred = inf(speech=torch.zeros(160), metadata_idx=7, utt_id="utt7")

    assert pred[0]["processed_transcript"] == "ɡa"
    assert pred[0]["predicted_transcript"] == "g a!"
    assert json.loads(pred[0]["raw_model_response"])["transport"] == "websocket"
    assert (tmp_path / "cache.jsonl").exists()


def test_inference_reports_missing_prompt_field(tmp_path):
    factory = FakeWebSocketFactory([])
    inf = GptRealtimeInference(
        client_config={
            "api_key": "test-key",
            "websocket_connect": factory,
            "use_tool": False,
            "retry_config": {"max_retries": 1},
        },
        prompt_config={
            "system_prompt": "Canonical IPA: {canonical_ipa}",
            "user_prompt": "Transcribe",
        },
        clean_response=False,
        error_log_path=tmp_path / "errors.jsonl",
    )

    pred = inf(speech=torch.zeros(160), metadata_idx=9, utt_id="utt9")

    assert pred[0]["processed_transcript"] == ""
    assert pred[0]["error"]["type"] == "ValueError"
    assert "canonical_ipa" in pred[0]["error"]["message"]
    assert factory.calls == []


def test_inference_uses_cache_on_resume(tmp_path):
    factory = FakeWebSocketFactory(
        [
            _session_events(
                {
                    "type": "response.function_call_arguments.done",
                    "name": OpenAIRealtimeTranscriptionClient.TOOL_NAME,
                    "arguments": json.dumps({"transcription": "abc"}),
                },
                _done(),
            )
        ]
    )
    inf = GptRealtimeInference(
        client_config={
            "api_key": "test-key",
            "websocket_connect": factory,
            "use_tool": False,
            "retry_config": {"max_retries": 1},
        },
        prompt_config={"system_prompt": "system", "user_prompt": "transcribe"},
        cache_path=tmp_path / "cache.jsonl",
    )

    first = inf(speech=torch.zeros(160), metadata_idx=1)
    second = inf(speech=torch.ones(160), metadata_idx=1)

    assert second == first
    assert len(factory.calls) == 1


def test_inference_logs_api_error(tmp_path):
    factory = FakeWebSocketFactory(
        [
            _session_events(
                {
                    "type": "error",
                    "error": {
                        "type": "server_error",
                        "code": "server_error",
                        "message": "api down",
                    },
                }
            )
        ]
    )
    inf = GptRealtimeInference(
        client_config={
            "api_key": "test-key",
            "websocket_connect": factory,
            "use_tool": False,
            "retry_config": {"max_retries": 1},
        },
        prompt_config={"system_prompt": "system", "user_prompt": "transcribe"},
        error_log_path=tmp_path / "errors.jsonl",
    )

    pred = inf(speech=torch.zeros(160), metadata_idx=4, utt_id="utt4")

    assert pred[0]["processed_transcript"] == ""
    assert pred[0]["error"]["type"] == "RealtimeAPIError"
    assert "api down" in pred[0]["error"]["message"]
    lines = (tmp_path / "errors.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["key"] == "utt4"


def test_inference_logs_top_level_realtime_error(tmp_path):
    factory = FakeWebSocketFactory(
        [
            _session_events(
                {
                    "type": "invalid_request_error",
                    "code": "invalid_value",
                    "message": "bad event",
                    "param": "type",
                }
            )
        ]
    )
    inf = GptRealtimeInference(
        client_config={
            "api_key": "test-key",
            "websocket_connect": factory,
            "use_tool": False,
            "retry_config": {"max_retries": 1},
        },
        prompt_config={"system_prompt": "system", "user_prompt": "transcribe"},
        error_log_path=tmp_path / "errors.jsonl",
    )

    pred = inf(speech=torch.zeros(160), metadata_idx=4, utt_id="utt4")

    assert pred[0]["processed_transcript"] == ""
    assert pred[0]["error"]["type"] == "RealtimeAPIError"
    assert "bad event" in pred[0]["error"]["message"]


def test_missing_audio_returns_error_prediction(tmp_path):
    factory = FakeWebSocketFactory([])
    inf = GptRealtimeInference(
        client_config={
            "api_key": "test-key",
            "websocket_connect": factory,
            "use_tool": False,
            "retry_config": {"max_retries": 1},
        },
        prompt_config={"system_prompt": "system", "user_prompt": "transcribe"},
        error_log_path=tmp_path / "errors.jsonl",
    )

    pred = inf(metadata_idx=5)

    assert pred[0]["processed_transcript"] == ""
    assert pred[0]["error"]["type"] == "ValueError"
    assert not factory.calls
