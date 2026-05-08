import json
from types import SimpleNamespace

from scripts import gemini_batch


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_collect_merges_only_transcription_shards(tmp_path):
    manifest_path = tmp_path / "batch_manifest.jsonl"
    results_path = tmp_path / "batch_results.jsonl"
    _write_jsonl(
        manifest_path,
        [
            {
                "key": "utt1",
                "idx": 0,
                "passthrough": {
                    "target": "a",
                    "utt_id": "utt1",
                    "lang_sym": "eng",
                    "canonical_ipa": "a",
                },
            },
            {
                "key": "utt2",
                "idx": 1,
                "passthrough": {
                    "target": "b",
                    "utt_id": "utt2",
                    "lang_sym": "eng",
                    "canonical_ipa": "b",
                },
            },
        ],
    )
    _write_jsonl(
        results_path,
        [
            {
                "key": "utt1",
                "response": {
                    "candidates": [
                        {
                            "content": {
                                "parts": [{"text": '{"transcription":"a"}'}]
                            }
                        }
                    ]
                },
            },
            {
                "key": "utt2",
                "response": {"text": '{"transcription":"b"}'},
            },
        ],
    )
    _write_jsonl(tmp_path / "transcription.9.jsonl", [{"stale": {"pred": []}}])
    _write_jsonl(
        tmp_path / "transcription.errors.jsonl",
        [{"key": "utt1", "error": {"type": "StaleError"}}],
    )

    gemini_batch.convert_results_to_transcription_shard(
        run_dir=tmp_path,
        manifest_path=manifest_path,
        results_path=results_path,
        output_key="transcription",
        clean=True,
    )
    pred_path = gemini_batch.merge_transcription_outputs(tmp_path)

    merged = json.loads(pred_path.read_text(encoding="utf-8"))
    assert sorted(merged) == ["0", "1"]
    assert merged["0"]["pred"][0]["processed_transcript"] == "a"
    assert merged["1"]["pred"][0]["processed_transcript"] == "b"
    assert "key" not in merged
    assert "response" not in merged
    assert not (tmp_path / "transcription.9.jsonl").exists()
    assert not (tmp_path / "transcription.errors.jsonl").exists()


def test_collect_records_missing_results_as_empty_predictions(tmp_path):
    manifest_path = tmp_path / "batch_manifest.jsonl"
    results_path = tmp_path / "batch_results.jsonl"
    _write_jsonl(
        manifest_path,
        [
            {
                "key": "utt1",
                "idx": 0,
                "passthrough": {"target": "a", "utt_id": "utt1"},
            }
        ],
    )
    _write_jsonl(results_path, [])

    gemini_batch.convert_results_to_transcription_shard(
        run_dir=tmp_path,
        manifest_path=manifest_path,
        results_path=results_path,
        output_key="transcription",
        clean=True,
    )
    pred_path = gemini_batch.merge_transcription_outputs(tmp_path)

    merged = json.loads(pred_path.read_text(encoding="utf-8"))
    pred = merged["0"]["pred"][0]
    assert pred["processed_transcript"] == ""
    assert pred["predicted_transcript"] == ""
    assert pred["error"]["type"] == "MissingGeminiBatchResult"


def test_build_generate_content_request_uses_audio_prompt_and_schema():
    request = gemini_batch.build_generate_content_request(
        user_prompt="Transcribe this.",
        system_prompt="You are a phonetician.",
        file_uri="files/audio123",
        mime_type="audio/wav",
        generation_config={
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "required": ["transcription"],
                "properties": {"transcription": {"type": "STRING"}},
            },
        },
    )

    parts = request["contents"][0]["parts"]
    assert parts[0]["fileData"] == {
        "fileUri": "files/audio123",
        "mimeType": "audio/wav",
    }
    assert parts[1] == {"text": "Transcribe this."}
    assert request["systemInstruction"]["parts"][0]["text"] == "You are a phonetician."
    assert request["generationConfig"]["responseMimeType"] == "application/json"


def test_status_writes_batch_status_without_network(tmp_path, monkeypatch, capsys):
    job_path = tmp_path / "batch_job.json"
    job_path.write_text(
        json.dumps({"name": "batches/test-job"}, ensure_ascii=False),
        encoding="utf-8",
    )

    fake_job = SimpleNamespace(
        name="batches/test-job",
        state=SimpleNamespace(name="JOB_STATE_SUCCEEDED"),
        dest=SimpleNamespace(file_name="files/result-jsonl"),
    )
    fake_client = SimpleNamespace(
        batches=SimpleNamespace(get=lambda name: fake_job)
    )
    monkeypatch.setattr(gemini_batch, "make_client", lambda: fake_client)

    args = SimpleNamespace(
        run_dir=str(tmp_path),
        job_name=None,
        wait=False,
        poll_interval=0.01,
        download_results=False,
    )
    gemini_batch.status(args)

    output = capsys.readouterr().out
    assert "JOB_STATE_SUCCEEDED" in output
    assert (tmp_path / "batch_status.json").exists()


def test_submit_overrides_apply_defaults_and_allow_lightweight_overrides():
    overrides = gemini_batch.submit_overrides(["inference.limit_samples=1"])

    assert "experiment=inference/transcribe_gemini31pro_batch" in overrides
    assert "data=powsmeval" in overrides
    assert "data.dataset_name=authentic_kids_kaldi" in overrides
    assert "inference.limit_samples=1" in overrides
    assert any(
        item.startswith("task_name=inf_authentic_kids_kaldi_gemini31pro_batch_")
        for item in overrides
    )


def test_load_default_env_reads_gemini_key_without_overriding_existing_env(
    tmp_path, monkeypatch
):
    env_path = tmp_path / ".env"
    env_path.write_text("GEMINI_API_KEY=from_file\n", encoding="utf-8")
    monkeypatch.setattr(gemini_batch, "DEFAULT_ENV_FILE", env_path)

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    gemini_batch.load_default_env()
    assert gemini_batch.resolve_env_api_key(None) == "from_file"

    monkeypatch.setenv("GEMINI_API_KEY", "from_shell")
    gemini_batch.load_default_env()
    assert gemini_batch.resolve_env_api_key(None) == "from_shell"
