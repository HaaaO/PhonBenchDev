import json

import pytest

from src.model.gemini.transcribe import GeminiInference


class FakeGeminiClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return json.dumps({"transcription": "h a"})


def test_gemini_inference_renders_canonical_prompt(monkeypatch, tmp_path):
    monkeypatch.setattr("src.model.gemini.transcribe.GeminiClient", FakeGeminiClient)
    inference = GeminiInference(
        client_config={"model_name": "fake", "api_key": "test-key"},
        prompt_config={
            "system_prompt": "Canonical IPA: {canonical_ipa}",
            "user_prompt": "Utterance: {utt_id}",
        },
        output_key="transcription",
        clean_response=False,
        cache_path=tmp_path / "cache.jsonl",
        error_log_path=tmp_path / "errors.jsonl",
    )

    pred = inference(
        wavpath="audio.wav",
        canonical_ipa="h aʊ s",
        utt_id="utt1",
        metadata_idx=1,
    )

    assert pred[0]["predicted_transcript"] == "h a"
    call = inference.client.calls[0]
    assert call["system_prompt"] == "Canonical IPA: h aʊ s"
    assert call["prompt"] == "Utterance: utt1"


def test_gemini_inference_reports_missing_prompt_field(monkeypatch, tmp_path):
    monkeypatch.setattr("src.model.gemini.transcribe.GeminiClient", FakeGeminiClient)
    inference = GeminiInference(
        client_config={"model_name": "fake", "api_key": "test-key"},
        prompt_config={
            "system_prompt": "Canonical IPA: {canonical_ipa}",
            "user_prompt": "Transcribe",
        },
        output_key="transcription",
        clean_response=False,
        error_log_path=tmp_path / "errors.jsonl",
    )

    pred = inference(wavpath="audio.wav", utt_id="utt1", metadata_idx=1)

    assert pred[0]["processed_transcript"] == ""
    assert pred[0]["error"]["type"] == "ValueError"
    assert "canonical_ipa" in pred[0]["error"]["message"]
    assert inference.client.calls == []


def test_gemini_inference_prefers_utt_id_over_metadata_idx_cache_key(
    monkeypatch, tmp_path
):
    monkeypatch.setattr("src.model.gemini.transcribe.GeminiClient", FakeGeminiClient)
    cache_path = tmp_path / "cache.jsonl"
    inference = GeminiInference(
        client_config={"model_name": "fake", "api_key": "test-key"},
        prompt_config={"system_prompt": "", "user_prompt": "Transcribe"},
        output_key="transcription",
        clean_response=False,
        cache_path=cache_path,
        cache_key_field="metadata_idx",
    )

    with pytest.warns(RuntimeWarning, match="metadata_idx"):
        first = inference(wavpath="audio-a.wav", utt_id="utt1", metadata_idx=1)
    second = inference(wavpath="audio-b.wav", utt_id="utt1", metadata_idx=2)

    assert second == first
    assert len(inference.client.calls) == 1
    record = json.loads(cache_path.read_text(encoding="utf-8").strip())
    assert record["key"] == "utt1"
