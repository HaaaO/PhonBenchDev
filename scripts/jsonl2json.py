import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.ipa_normalization import (
    DEFAULT_IPA_NORMALIZATION_PROFILE,
    normalize_transcription_data,
)


REPORT_COLUMNS = [
    "utt_id",
    "field",
    "raw",
    "normalized",
    "profile",
    "changed",
    "rule_ids",
]


def _is_inference_jsonl(path: Path) -> bool:
    name = path.name
    return ".cache" not in name and ".error" not in name and ".errors" not in name


def load_jsonl_shards(dirpath: Path) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for path in sorted(dirpath.glob("*jsonl")):
        if not _is_inference_jsonl(path):
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    merged.update(json.loads(line))
    return merged


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_normalization_report(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in REPORT_COLUMNS})


def merge_jsonl_dir(
    dirpath: Path,
    *,
    normalization_profile: Optional[str] = DEFAULT_IPA_NORMALIZATION_PROFILE,
    raw_filename: str = "transcription_raw.json",
    normalized_filename: str = "transcription_normalized.json",
    compat_filename: str = "transcription.json",
    report_filename: str = "normalization_report.csv",
) -> Dict[str, Path]:
    dirpath = Path(dirpath)
    merged = load_jsonl_shards(dirpath)

    raw_path = dirpath / raw_filename
    compat_path = dirpath / compat_filename
    write_json(raw_path, merged)

    outputs = {"raw": raw_path, "compat": compat_path}

    if normalization_profile:
        normalized, report_rows = normalize_transcription_data(
            merged,
            profile=normalization_profile,
        )
        normalized_path = dirpath / normalized_filename
        report_path = dirpath / report_filename
        write_json(normalized_path, normalized)
        write_json(compat_path, normalized)
        write_normalization_report(report_path, report_rows)
        outputs.update({"normalized": normalized_path, "report": report_path})
        print(
            "Merged "
            f"{len(merged)} entries into {raw_path}, {normalized_path}, and {compat_path}"
        )
        print(f"Wrote {len(report_rows)} normalization audit rows to {report_path}")
    else:
        write_json(compat_path, merged)
        print(f"Merged {len(merged)} entries into {raw_path} and {compat_path}")

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirname", required=True)
    parser.add_argument(
        "--normalization-profile",
        default=DEFAULT_IPA_NORMALIZATION_PROFILE,
        help=(
            "IPA normalization profile to apply during merge. "
            "Use --no-normalize to keep transcription.json raw."
        ),
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Write transcription.json as the raw merged output.",
    )
    args = parser.parse_args()

    merge_jsonl_dir(
        Path(args.dirname),
        normalization_profile=None if args.no_normalize else args.normalization_profile,
    )


if __name__ == "__main__":
    main()
