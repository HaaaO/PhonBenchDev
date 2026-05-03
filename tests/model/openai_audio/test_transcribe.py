import base64
import json
import wave
from io import BytesIO
from types import SimpleNamespace

import pytest
import torch

from src.model.openai_audio.client import OpenAIAudioClient
from src.model.openai_audio.transcribe import GptAudioInference


class FakeCompletions:
    def __init__(self, message=None, error=None):
        self.message = message
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(choices=[SimpleNamespace(message=self.message)])


class FakeOpenAIClient:
    def __init__(self, message=None, error=None):
        self.completions = FakeCompletions(message=message, error=error)
        self.chat = SimpleNamespace(completions=self.completions)


def _tool_message(transcription="həˈloʊ"):
    return SimpleNamespace(
        content=None,
        tool_calls=[
            SimpleNamespace(
                function=SimpleNamespace(
                    name=OpenAIAudioClient.TOOL_NAME,
                    arguments=json.dumps({"transcription": transcription}),
                )
            )
        ],
    )


def _text_message(content):
    return SimpleNamespace(content=content, tool_calls=[])


def _prompt_schema():
    return {
        "type": "OBJECT",
        "required": ["transcription"],
        "properties": {"transcription": {"type": "STRING"}},
    }


def _inference(fake_client, tmp_path):
    return GptAudioInference(
        client_config={
            "model_name": "gpt-audio-1.5",
            "api_key": "test-key",
            "client": fake_client,
            "temperature": 0.0,
            "response_schema": _prompt_schema(),
            "retry_config": {"max_retries": 1},
        },
        prompt_config={
            "system_prompt": "system",
            "user_prompt": "transcribe",
        },
        clean_response=True,
        output_key="transcription",
        cache_path=tmp_path / "cache.jsonl",
        error_log_path=tmp_path / "errors.jsonl",
    )


def _write_wav(path, sample_rate=16000):
    samples = torch.zeros(160, dtype=torch.float32)
    pcm16 = (samples.numpy() * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm16.tobytes())


def test_client_accepts_prompt_response_schema_and_encodes_tensor_audio():
    fake = FakeOpenAIClient(message=_tool_message("abc"))
    client = OpenAIAudioClient(
        api_key="test-key",
        client=fake,
        response_schema=_prompt_schema(),
    )
    audio_input = client.audio_input_from_sample(speech=torch.zeros(160), sample_rate=16000)

    transcript, raw = client.generate_transcription(
        audio_input=audio_input,
        prompt="prompt",
        system_prompt="system",
    )

    assert transcript == "abc"
    assert json.loads(raw) == {"transcription": "abc"}

    call = fake.completions.calls[0]
    assert call["model"] == "gpt-audio-1.5"
    assert call["modalities"] == ["text"]
    assert call["tool_choice"]["function"]["name"] == OpenAIAudioClient.TOOL_NAME
    assert call["parallel_tool_calls"] is False

    tool_schema = call["tools"][0]["function"]["parameters"]
    assert tool_schema["type"] == "object"
    assert tool_schema["properties"]["transcription"]["type"] == "string"

    user_content = call["messages"][1]["content"]
    assert user_content[0]["type"] == "input_audio"
    assert user_content[0]["input_audio"]["format"] == "wav"
    decoded = base64.b64decode(user_content[0]["input_audio"]["data"])
    with wave.open(BytesIO(decoded), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getframerate() == 16000


def test_inference_returns_gemini_compatible_transcription(tmp_path):
    fake = FakeOpenAIClient(message=_tool_message("g a!"))
    inf = _inference(fake, tmp_path)

    pred = inf(speech=torch.zeros(160), metadata_idx=7, utt_id="utt7")

    assert pred == [
        {
            "processed_transcript": "ɡa",
            "predicted_transcript": "g a!",
            "raw_model_response": '{"transcription": "g a!"}',
        }
    ]
    assert (tmp_path / "cache.jsonl").exists()


def test_inference_renders_canonical_prompt(tmp_path):
    fake = FakeOpenAIClient(message=_tool_message("h a"))
    inf = GptAudioInference(
        client_config={
            "model_name": "gpt-audio-1.5",
            "api_key": "test-key",
            "client": fake,
            "temperature": 0.0,
            "response_schema": _prompt_schema(),
            "retry_config": {"max_retries": 1},
        },
        prompt_config={
            "system_prompt": "Canonical IPA: {canonical_ipa}",
            "user_prompt": "Utterance: {utt_id}",
        },
        clean_response=False,
        output_key="transcription",
        cache_path=tmp_path / "cache.jsonl",
        error_log_path=tmp_path / "errors.jsonl",
    )

    pred = inf(
        speech=torch.zeros(160),
        metadata_idx=8,
        utt_id="utt8",
        canonical_ipa="h aʊ s",
    )

    assert pred[0]["predicted_transcript"] == "h a"
    messages = fake.completions.calls[0]["messages"]
    assert messages[0]["content"] == "Canonical IPA: h aʊ s"
    assert messages[1]["content"][1]["text"] == "Utterance: utt8"


def test_inference_reports_missing_prompt_field(tmp_path):
    fake = FakeOpenAIClient(message=_tool_message("h a"))
    inf = GptAudioInference(
        client_config={
            "model_name": "gpt-audio-1.5",
            "api_key": "test-key",
            "client": fake,
            "temperature": 0.0,
            "response_schema": _prompt_schema(),
            "retry_config": {"max_retries": 1},
        },
        prompt_config={
            "system_prompt": "Canonical IPA: {canonical_ipa}",
            "user_prompt": "Transcribe",
        },
        clean_response=False,
        output_key="transcription",
        error_log_path=tmp_path / "errors.jsonl",
    )

    pred = inf(speech=torch.zeros(160), metadata_idx=9, utt_id="utt9")

    assert pred[0]["processed_transcript"] == ""
    assert pred[0]["error"]["type"] == "ValueError"
    assert "canonical_ipa" in pred[0]["error"]["message"]
    assert fake.completions.calls == []


def test_inference_uses_cache_on_resume(tmp_path):
    fake = FakeOpenAIClient(message=_tool_message("abc"))
    inf = _inference(fake, tmp_path)

    first = inf(speech=torch.zeros(160), metadata_idx=1)
    second = inf(speech=torch.ones(160), metadata_idx=1)

    assert second == first
    assert len(fake.completions.calls) == 1


def test_inference_loads_audio_path_fallback(tmp_path):
    audio_path = tmp_path / "sample.wav"
    _write_wav(audio_path)
    fake = FakeOpenAIClient(message=_tool_message("abc"))
    inf = _inference(fake, tmp_path)

    pred = inf(audio_path=str(audio_path), metadata_idx=2)

    assert pred[0]["predicted_transcript"] == "abc"
    user_content = fake.completions.calls[0]["messages"][1]["content"]
    assert user_content[0]["input_audio"]["format"] == "wav"


def test_text_json_fallback_is_parsed(tmp_path):
    fake = FakeOpenAIClient(message=_text_message('{"transcription": "k ae t"}'))
    inf = _inference(fake, tmp_path)

    pred = inf(speech=torch.zeros(160), metadata_idx=3)

    assert pred[0]["predicted_transcript"] == "k ae t"
    assert pred[0]["processed_transcript"] == "kaet"


def test_error_prediction_and_error_log(tmp_path):
    fake = FakeOpenAIClient(error=RuntimeError("api down"))
    inf = _inference(fake, tmp_path)

    pred = inf(speech=torch.zeros(160), metadata_idx=4, audio_path="/tmp/a.wav")

    assert pred[0]["processed_transcript"] == ""
    assert pred[0]["error"]["type"] == "RuntimeError"
    assert "api down" in pred[0]["error"]["message"]
    lines = (tmp_path / "errors.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["key"] == "4"


def test_missing_audio_returns_error_prediction(tmp_path):
    fake = FakeOpenAIClient(message=_tool_message("abc"))
    inf = _inference(fake, tmp_path)

    pred = inf(metadata_idx=5)

    assert pred[0]["processed_transcript"] == ""
    assert pred[0]["error"]["type"] == "ValueError"
    assert not fake.completions.calls


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("plain ipa", "plain ipa"),
        ('prefix {"transcription": "json ipa"} suffix', "json ipa"),
    ],
)
def test_client_text_response_fallback(content, expected):
    message = _text_message(content)
    transcript, raw = OpenAIAudioClient._extract_transcription(message)
    assert transcript == expected
    assert raw == content
