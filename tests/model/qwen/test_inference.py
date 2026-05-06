import json
from types import SimpleNamespace

import torch

from src.model.qwen.inference import VllmInference


class FakeCompletions:
    def __init__(self, content='{"transcription": "h a!"}', error=None):
        self.content = content
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeOpenAIClient:
    def __init__(self, content='{"transcription": "h a!"}', error=None):
        self.completions = FakeCompletions(content=content, error=error)
        self.chat = SimpleNamespace(completions=self.completions)


def _inference(fake_client, tmp_path, prompt_config=None):
    return VllmInference(
        client_config={
            "base_url": "http://127.0.0.1:8000/v1",
            "model_name": "Qwen/Qwen2.5-Omni-3B",
            "api_key": "EMPTY",
            "client": fake_client,
            "temperature": 0.0,
            "top_p": 0.95,
            "max_tokens": 128,
            "modalities": ["text"],
            "extra_body": {"test": True},
        },
        prompt_config=prompt_config
        or {
            "system_prompt": "system",
            "user_prompt": "transcribe",
        },
        clean_response=True,
        output_key="transcription",
        cache_path=tmp_path / "cache.jsonl",
        error_log_path=tmp_path / "errors.jsonl",
    )


def test_vllm_inference_renders_shared_prompt_and_requests_text_only(tmp_path):
    fake = FakeOpenAIClient()
    inference = _inference(
        fake,
        tmp_path,
        prompt_config={
            "system_prompt": "Canonical IPA: {canonical_ipa}",
            "user_prompt": "Utterance: {utt_id}",
        },
    )

    pred = inference(
        speech=torch.zeros(160),
        utt_id="utt1",
        canonical_ipa="h aʊ s",
    )

    assert pred[0]["predicted_transcript"] == "h a!"
    assert pred[0]["processed_transcript"] == "ha"
    call = fake.completions.calls[0]
    assert call["model"] == "Qwen/Qwen2.5-Omni-3B"
    assert call["modalities"] == ["text"]
    assert call["extra_body"] == {"test": True}
    assert call["messages"][0]["content"] == "Canonical IPA: h aʊ s"
    user_content = call["messages"][1]["content"]
    assert user_content[0]["type"] == "audio_url"
    assert user_content[0]["audio_url"]["url"].startswith("data:audio/wav;base64,")
    assert user_content[1]["text"] == "Utterance: utt1"


def test_vllm_inference_reports_missing_prompt_field(tmp_path):
    fake = FakeOpenAIClient()
    inference = _inference(
        fake,
        tmp_path,
        prompt_config={
            "system_prompt": "Canonical IPA: {canonical_ipa}",
            "user_prompt": "Transcribe",
        },
    )

    pred = inference(speech=torch.zeros(160), utt_id="utt2")

    assert pred[0]["processed_transcript"] == ""
    assert pred[0]["error"]["type"] == "ValueError"
    assert "canonical_ipa" in pred[0]["error"]["message"]
    assert fake.completions.calls == []


def test_vllm_inference_supports_legacy_user_prompt_template(tmp_path):
    fake = FakeOpenAIClient(content=json.dumps({"transcription": "k ae t"}))
    inference = _inference(
        fake,
        tmp_path,
        prompt_config={
            "system_prompt": "",
            "user_prompt_template": "Legacy template for {utt_id}",
        },
    )

    pred = inference(speech=torch.zeros(160), utt_id="utt3")

    assert pred[0]["processed_transcript"] == "kaet"
    call = fake.completions.calls[0]
    assert call["messages"][0]["content"][1]["text"] == "Legacy template for utt3"


def test_vllm_inference_loads_sibling_distributed_cache_shards(tmp_path):
    cached_pred = [
        {
            "processed_transcript": "cached",
            "predicted_transcript": "cached",
            "raw_model_response": "cached",
        }
    ]
    sibling_cache = tmp_path / "task.cache.0.0.jsonl"
    current_worker_cache = tmp_path / "task.cache.0.1.jsonl"
    sibling_cache.write_text(
        json.dumps({"key": "utt4", "pred": cached_pred}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    fake = FakeOpenAIClient(content=json.dumps({"transcription": "fresh"}))
    inference = VllmInference(
        client_config={
            "base_url": "http://127.0.0.1:8000/v1",
            "model_name": "Qwen/Qwen2.5-Omni-3B",
            "api_key": "EMPTY",
            "client": fake,
        },
        prompt_config={"system_prompt": "", "user_prompt": "transcribe"},
        clean_response=True,
        output_key="transcription",
        cache_path=current_worker_cache,
    )

    pred = inference(speech=torch.zeros(160), utt_id="utt4")

    assert pred == cached_pred
    assert fake.completions.calls == []
