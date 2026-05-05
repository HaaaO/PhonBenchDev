import csv
import importlib.util
import json
from pathlib import Path


def _load_jsonl2json_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "jsonl2json.py"
    spec = importlib.util.spec_from_file_location("jsonl2json", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_merge_jsonl_dir_writes_raw_normalized_alias_and_report(tmp_path):
    module = _load_jsonl2json_module()
    raw_record = {
        "0": {
            "pred": [
                {
                    "processed_transcript": "ˈt ʃ ˌpʰ r g",
                    "predicted_transcript": "ˈt ʃ ˌpʰ r g",
                }
            ],
            "passthrough": {
                "utt_id": "utt-0",
                "target": "ˈt ʃ ˌpʰ r g",
                "canonical_ipa": "ˈd ʒ ˌkʰ ɫ",
            },
        }
    }
    ignored_cache_record = {
        "cache": {
            "pred": [{"processed_transcript": "ɕ"}],
            "passthrough": {"utt_id": "cache"},
        }
    }
    (tmp_path / "transcription.0.jsonl").write_text(
        json.dumps(raw_record, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "transcription.cache.0.jsonl").write_text(
        json.dumps(ignored_cache_record, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    outputs = module.merge_jsonl_dir(tmp_path)

    assert outputs["raw"] == tmp_path / "transcription_raw.json"
    assert outputs["normalized"] == tmp_path / "transcription_normalized.json"
    assert outputs["compat"] == tmp_path / "transcription.json"
    assert outputs["report"] == tmp_path / "normalization_report.csv"

    raw = json.loads((tmp_path / "transcription_raw.json").read_text(encoding="utf-8"))
    normalized = json.loads(
        (tmp_path / "transcription_normalized.json").read_text(encoding="utf-8")
    )
    compat = json.loads((tmp_path / "transcription.json").read_text(encoding="utf-8"))

    assert raw == raw_record
    assert compat == normalized
    assert "cache" not in normalized

    normalized_item = normalized["0"]
    assert normalized_item["normalization_profile"] == "ipa_eng_broad_v1"
    assert normalized_item["pred"][0]["processed_transcript"] == "t͡ʃ p ɹ ɡ"
    assert normalized_item["pred"][0]["processed_transcript_raw"] == "ˈt ʃ ˌpʰ r g"
    assert normalized_item["passthrough"]["target"] == "t͡ʃ p ɹ ɡ"
    assert normalized_item["passthrough"]["target_raw"] == "ˈt ʃ ˌpʰ r g"
    assert normalized_item["passthrough"]["canonical_ipa"] == "d͡ʒ k l"

    with (tmp_path / "normalization_report.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert {row["field"] for row in rows} == {
        "pred[0].processed_transcript",
        "pred[0].predicted_transcript",
        "passthrough.target",
        "passthrough.canonical_ipa",
    }
    assert all(row["utt_id"] == "utt-0" for row in rows)
    assert all(row["profile"] == "ipa_eng_broad_v1" for row in rows)
    assert all(row["changed"] == "True" for row in rows)
