#!/usr/bin/env python3
"""Audit IPA symbols that CMU39 projection would skip.

The scanner mirrors ``src.core.cmu39_projection.segment_ipa_for_cmu39``:
normalize IPA text, remove whitespace/ASCII punctuation, greedily match the
CMU39 inventory plus aliases, and report spans that do not match anything.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.cmu39_projection import _SCAN_TOKENS  # noqa: E402
from src.core.cmu39_projection import _clean_for_scan as _project_clean_for_scan  # noqa: E402
from src.core.ipa_normalization import normalize_ipa_text  # noqa: E402


def _clean_for_scan(text: str) -> str:
    return _project_clean_for_scan(text)


def skipped_spans(text: str) -> list[str]:
    """Return normalized spans that the CMU39 scanner would skip."""
    compact = _clean_for_scan(text)
    spans: list[str] = []
    idx = 0
    while idx < len(compact):
        matched = False
        for token in _SCAN_TOKENS:
            if compact.startswith(token, idx):
                idx += len(token)
                matched = True
                break
        if matched:
            continue

        start = idx
        idx += 1
        while idx < len(compact):
            if any(compact.startswith(token, idx) for token in _SCAN_TOKENS):
                break
            idx += 1
        spans.append(compact[start:idx])
    return spans


def codepoints(text: str) -> str:
    return " ".join(f"U+{ord(ch):04X}" for ch in text)


def unicode_names(text: str) -> str:
    return "; ".join(
        f"{ch}: {unicodedata.name(ch, 'UNNAMED')}" for ch in text
    )


def is_transcription_shard(path: Path) -> bool:
    name = path.name
    if not name.startswith("transcription.") or not name.endswith(".jsonl"):
        return False
    lowered = name.lower()
    return "cache" not in lowered and "error" not in lowered


def _under_path(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _excluded_old_dir(root: Path, include_old: bool) -> Optional[Path]:
    old_dir = REPO_ROOT / "exp" / "runs" / "Old"
    if include_old or not _under_path(old_dir, root):
        return None
    return old_dir


def _is_excluded(path: Path, excluded_dir: Optional[Path]) -> bool:
    return excluded_dir is not None and _under_path(path, excluded_dir)


def discover_files(
    root: Path,
    include_all_shards: bool,
    *,
    include_old: bool,
) -> list[Path]:
    excluded_dir = _excluded_old_dir(root, include_old)
    files = {
        path
        for path in root.rglob("transcription.json")
        if not _is_excluded(path, excluded_dir)
    }
    for path in root.rglob("transcription.*.jsonl"):
        if _is_excluded(path, excluded_dir):
            continue
        if not is_transcription_shard(path):
            continue
        if include_all_shards or not (path.parent / "transcription.json").exists():
            files.add(path)
    return sorted(files)


def read_json_or_jsonl(path: Path) -> Iterator[Any]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"WARNING: invalid JSON in {path}:{line_no}: {e}", file=sys.stderr)
        return

    try:
        with path.open("r", encoding="utf-8") as f:
            yield json.load(f)
    except json.JSONDecodeError as e:
        print(f"WARNING: invalid JSON in {path}: {e}", file=sys.stderr)


def _utt_from_mapping(mapping: dict[str, Any], fallback: Optional[str]) -> Optional[str]:
    for key in ("utt_id", "key", "request_key", "batch_key", "metadata_idx"):
        if mapping.get(key) is not None:
            return str(mapping[key])
    passthrough = mapping.get("passthrough")
    if isinstance(passthrough, dict):
        for key in ("utt_id", "key", "metadata_idx"):
            if passthrough.get(key) is not None:
                return str(passthrough[key])
    return fallback


def iter_field_values(
    obj: Any,
    *,
    field: str,
    utt_id: Optional[str] = None,
    json_path: str = "$",
) -> Iterator[tuple[str, Optional[str], str]]:
    if isinstance(obj, dict):
        local_utt = _utt_from_mapping(obj, utt_id)
        value = obj.get(field)
        if isinstance(value, str):
            yield value, local_utt, f"{json_path}.{field}"
        for key, child in obj.items():
            child_path = f"{json_path}.{key}"
            yield from iter_field_values(
                child,
                field=field,
                utt_id=local_utt,
                json_path=child_path,
            )
    elif isinstance(obj, list):
        for idx, child in enumerate(obj):
            yield from iter_field_values(
                child,
                field=field,
                utt_id=utt_id,
                json_path=f"{json_path}[{idx}]",
            )


def iter_transcription_items(
    obj: Any,
    *,
    utt_id: Optional[str] = None,
    json_path: str = "$",
) -> Iterator[tuple[dict[str, Any], Optional[str], str]]:
    if isinstance(obj, dict):
        local_utt = _utt_from_mapping(obj, utt_id)
        if "pred" in obj or isinstance(obj.get("passthrough"), dict):
            yield obj, local_utt, json_path
            return
        for key, child in obj.items():
            yield from iter_transcription_items(
                child,
                utt_id=local_utt,
                json_path=f"{json_path}.{key}",
            )
    elif isinstance(obj, list):
        for idx, child in enumerate(obj):
            yield from iter_transcription_items(
                child,
                utt_id=utt_id,
                json_path=f"{json_path}[{idx}]",
            )


def iter_eval_field_values(
    obj: Any,
) -> Iterator[tuple[str, str, str, Optional[str], str]]:
    """Yield the exact transcript fields used by CMU39 MDD evaluation.

    Returns ``(value, role, field, utt_id, json_path)``. The prediction role is
    intentionally limited to ``pred[0].processed_transcript`` because that is
    the default scoring input.
    """
    for item, utt_id, item_path in iter_transcription_items(obj):
        pred = item.get("pred")
        if isinstance(pred, list) and pred and isinstance(pred[0], dict):
            value = pred[0].get("processed_transcript")
            if isinstance(value, str):
                yield (
                    value,
                    "predicted",
                    "processed_transcript",
                    utt_id,
                    f"{item_path}.pred[0].processed_transcript",
                )
        elif isinstance(pred, dict):
            value = pred.get("processed_transcript")
            if isinstance(value, str):
                yield (
                    value,
                    "predicted",
                    "processed_transcript",
                    utt_id,
                    f"{item_path}.pred.processed_transcript",
                )

        passthrough = item.get("passthrough")
        if not isinstance(passthrough, dict):
            continue
        for role, field in (
            ("uttered", "target"),
            ("canonical", "canonical_ipa"),
        ):
            value = passthrough.get(field)
            if isinstance(value, str):
                yield value, role, field, utt_id, f"{item_path}.passthrough.{field}"


def _audit_rows_for_value(
    *,
    file: Path,
    utt_id: Optional[str],
    json_path: str,
    field: str,
    role: str,
    value: str,
) -> Iterator[dict[str, Any]]:
    normalized = normalize_ipa_text(value).normalized
    spans = skipped_spans(value)
    for span in spans:
        yield {
            "file": str(file),
            "utt_id": utt_id or "",
            "json_path": json_path,
            "role": role,
            "field": field,
            "skipped_span": span,
            "codepoints": codepoints(span),
            "unicode_names": unicode_names(span),
            "raw_transcript": value,
            "normalized_transcript": normalized,
        }


def iter_audit_rows(
    files: Iterable[Path],
    *,
    field: str,
    max_records: Optional[int],
) -> Iterator[dict[str, Any]]:
    seen_records = 0
    for path in files:
        for obj in read_json_or_jsonl(path):
            for value, utt_id, json_path in iter_field_values(obj, field=field):
                seen_records += 1
                yield from _audit_rows_for_value(
                    file=path,
                    utt_id=utt_id,
                    json_path=json_path,
                    role=field,
                    field=field,
                    value=value,
                )
                if max_records is not None and seen_records >= max_records:
                    return


def iter_eval_audit_rows(
    files: Iterable[Path],
    *,
    max_records: Optional[int],
) -> Iterator[dict[str, Any]]:
    seen_records = 0
    for path in files:
        for obj in read_json_or_jsonl(path):
            for value, role, field, utt_id, json_path in iter_eval_field_values(obj):
                seen_records += 1
                yield from _audit_rows_for_value(
                    file=path,
                    utt_id=utt_id,
                    json_path=json_path,
                    role=role,
                    field=field,
                    value=value,
                )
                if max_records is not None and seen_records >= max_records:
                    return


def iter_canonical_file_audit_rows(
    files: Iterable[Path],
    *,
    max_records: Optional[int],
) -> Iterator[dict[str, Any]]:
    seen_records = 0
    for path in files:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                parts = line.split(maxsplit=1)
                utt_id = parts[0]
                value = parts[1] if len(parts) == 2 else ""
                seen_records += 1
                yield from _audit_rows_for_value(
                    file=path,
                    utt_id=utt_id,
                    json_path=f"line:{line_no}",
                    role="canonical_file",
                    field="text.canonical",
                    value=value,
                )
                if max_records is not None and seen_records >= max_records:
                    return


def discover_canonical_files(
    roots: Iterable[Path],
    files: Iterable[Path],
) -> list[Path]:
    discovered = {path.expanduser().resolve() for path in files}
    for root in roots:
        discovered.update(
            path
            for path in root.expanduser().resolve().rglob("text.canonical")
            if path.is_file()
        )
    return sorted(discovered)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file",
        "utt_id",
        "json_path",
        "role",
        "field",
        "skipped_span",
        "codepoints",
        "unicode_names",
        "raw_transcript",
        "normalized_transcript",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_py(path: Path, span_counts: Counter[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    spans = sorted(span_counts)
    rows = span_counts.most_common()
    with path.open("w", encoding="utf-8") as f:
        f.write('"""Unique IPA spans skipped by CMU39 projection audit."""\n\n')
        f.write("SKIPPED_CMU39_SPANS = [\n")
        for span in spans:
            f.write(f"    {span!r},\n")
        f.write("]\n\n")
        f.write("SKIPPED_CMU39_SPAN_COUNTS = {\n")
        for span, count in rows:
            f.write(f"    {span!r}: {count},\n")
        f.write("}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT / "exp" / "runs",
        help="Root directory to scan. Defaults to exp/runs.",
    )
    parser.add_argument(
        "--field",
        default="processed_transcript",
        help="Transcript field to audit.",
    )
    parser.add_argument(
        "--eval-fields",
        action="store_true",
        help=(
            "Audit the fields used by scoring together: "
            "pred[0].processed_transcript, passthrough.target, and "
            "passthrough.canonical_ipa."
        ),
    )
    parser.add_argument(
        "--canonical-root",
        action="append",
        type=Path,
        default=[],
        help="Directory to search recursively for Kaldi text.canonical files.",
    )
    parser.add_argument(
        "--canonical-file",
        action="append",
        type=Path,
        default=[],
        help="Specific Kaldi text.canonical file to include in the audit.",
    )
    parser.add_argument(
        "--include-all-shards",
        action="store_true",
        help=(
            "Also scan transcription.*.jsonl shards when transcription.json "
            "exists in the same run directory. By default shards are used only "
            "when the merged file is absent."
        ),
    )
    parser.add_argument(
        "--include-old",
        action="store_true",
        help="Include exp/runs/Old. By default, Old runs are skipped.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Optional path for detailed skipped-span rows.",
    )
    parser.add_argument(
        "--output-py",
        type=Path,
        help=(
            "Optional .py file containing SKIPPED_CMU39_SPANS and "
            "SKIPPED_CMU39_SPAN_COUNTS."
        ),
    )
    parser.add_argument(
        "--top",
        type=int,
        default=50,
        help="Number of skipped spans to show in the terminal summary.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        help="Debug limit on number of processed transcript values inspected.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=5,
        help="Number of example utterances to show per skipped span.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    files = discover_files(
        root,
        args.include_all_shards,
        include_old=args.include_old,
    )
    if args.eval_fields:
        rows = list(
            iter_eval_audit_rows(
                files,
                max_records=args.max_records,
            )
        )
    else:
        rows = list(
            iter_audit_rows(
                files,
                field=args.field,
                max_records=args.max_records,
            )
        )

    canonical_files = discover_canonical_files(
        args.canonical_root,
        args.canonical_file,
    )
    if canonical_files:
        rows.extend(
            iter_canonical_file_audit_rows(
                canonical_files,
                max_records=args.max_records,
            )
        )

    span_counts = Counter(row["skipped_span"] for row in rows)
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        bucket = examples[row["skipped_span"]]
        if len(bucket) < args.max_examples:
            bucket.append(row)

    print(f"Scanned root: {root}")
    if not args.include_old:
        print("Excluded: exp/runs/Old")
    print(f"Files inspected: {len(files)}")
    if canonical_files:
        print(f"Canonical files inspected: {len(canonical_files)}")
    print(f"Skipped-span instances: {len(rows)}")
    print(f"Unique skipped spans: {len(span_counts)}")

    if span_counts:
        print("")
        print(f"Top skipped spans (up to {args.top}):")
        for span, count in span_counts.most_common(args.top):
            print(f"  {span!r}  count={count}  {codepoints(span)}")
            for row in examples[span]:
                print(
                    "    "
                    f"utt_id={row['utt_id'] or '?'} "
                    f"role={row['role']} "
                    f"file={Path(row['file']).parent.name}/{Path(row['file']).name} "
                    f"raw={row['raw_transcript']!r}"
                )

    if args.output_csv:
        out_path = args.output_csv.expanduser().resolve()
        write_csv(out_path, rows)
        print(f"\nWrote detailed rows to {out_path}")

    if args.output_py:
        out_path = args.output_py.expanduser().resolve()
        write_py(out_path, span_counts)
        print(f"Wrote skipped span list to {out_path}")


if __name__ == "__main__":
    main()
