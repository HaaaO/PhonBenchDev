#!/usr/bin/env python
"""Slot-level MDD error analysis.

This script expands the aggregate MDD counts from ``phone_recognition.py`` into
one row per canonical phone slot or insertion gap.  It is intended for auditing
questions such as:

* why True_Diagnostic_Accuracy has a particular value;
* which actual pronunciation errors were missed (FA) or diagnosed as the wrong
  phone (DE);
* whether a missed error has a matching predicted error nearby, which can point
  to a possible alignment/slot-boundary artifact.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.metrics.phone_recognition import (
    MDD_BLANK,
    PhoneRecognitionEvaluator,
    _align_to_prompted_slots,
    _levenshtein_align,
)


def load_canonical(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split(maxsplit=1)
            out[parts[0]] = parts[1] if len(parts) == 2 else ""
    return out


def load_prediction_items(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return list(data.values())


def pred_value(item: dict[str, Any], pred_field: str) -> str:
    pred = item.get("pred")
    if not isinstance(pred, list) or not pred or not isinstance(pred[0], dict):
        return ""
    value = pred[0].get(pred_field, "")
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def correctness(prompted: str, other: str, blank: str = MDD_BLANK) -> str:
    if prompted == blank:
        return "C" if other == blank else "E"
    if other == blank:
        return "E"
    return "C" if prompted == other else "E"


def bucket(prompted: str, uttered: str, predicted: str) -> str:
    corr_u = correctness(prompted, uttered)
    corr_p = correctness(prompted, predicted)
    if corr_u == "E" and corr_p == "E":
        return "CD" if uttered == predicted else "DE"
    if corr_u == "C" and corr_p == "C":
        return "TA"
    if corr_u == "C" and corr_p == "E":
        return "FR"
    return "FA"


def actual_error_type(prompted: str, uttered: str) -> str:
    if prompted == MDD_BLANK and uttered != MDD_BLANK:
        return "insertion"
    if prompted != MDD_BLANK and uttered == MDD_BLANK:
        return "deletion"
    if prompted != MDD_BLANK and uttered != MDD_BLANK and prompted != uttered:
        return "substitution"
    return "none"


def predicted_error_type(prompted: str, predicted: str) -> str:
    if prompted == MDD_BLANK and predicted != MDD_BLANK:
        return "insertion"
    if prompted != MDD_BLANK and predicted == MDD_BLANK:
        return "deletion"
    if prompted != MDD_BLANK and predicted != MDD_BLANK and prompted != predicted:
        return "substitution"
    return "none"


def expand_slots(
    evaluator: PhoneRecognitionEvaluator,
    utt_id: str,
    lang: str,
    canonical: str,
    uttered: str,
    predicted: str,
) -> list[dict[str, Any]]:
    prompted_segs = evaluator._mdd_segments(canonical)
    uttered_segs = evaluator._mdd_segments(uttered)
    predicted_segs = evaluator._mdd_segments(predicted)

    u_phones, u_gaps = _align_to_prompted_slots(prompted_segs, uttered_segs)
    h_phones, h_gaps = _align_to_prompted_slots(prompted_segs, predicted_segs)

    rows: list[dict[str, Any]] = []
    row_order = 0
    for slot_idx in range(len(prompted_segs) + 1):
        u_gap_aligned, h_gap_aligned = _levenshtein_align(
            u_gaps[slot_idx], h_gaps[slot_idx]
        )
        for gap_pos, (u_sym, h_sym) in enumerate(zip(u_gap_aligned, h_gap_aligned)):
            b = bucket(MDD_BLANK, u_sym, h_sym)
            rows.append(
                {
                    "utt_id": utt_id,
                    "language": lang,
                    "row_order": row_order,
                    "slot_type": "gap",
                    "slot_index": slot_idx,
                    "gap_pos": gap_pos,
                    "prompted": MDD_BLANK,
                    "uttered": u_sym,
                    "predicted": h_sym,
                    "corr_U": correctness(MDD_BLANK, u_sym),
                    "corr_P": correctness(MDD_BLANK, h_sym),
                    "bucket": b,
                    "actual_error_type": actual_error_type(MDD_BLANK, u_sym),
                    "predicted_error_type": predicted_error_type(MDD_BLANK, h_sym),
                    "canonical_seq": " ".join(prompted_segs),
                    "uttered_seq": " ".join(uttered_segs),
                    "predicted_seq": " ".join(predicted_segs),
                }
            )
            row_order += 1

        if slot_idx < len(prompted_segs):
            p_sym = prompted_segs[slot_idx]
            u_sym = u_phones[slot_idx]
            h_sym = h_phones[slot_idx]
            b = bucket(p_sym, u_sym, h_sym)
            rows.append(
                {
                    "utt_id": utt_id,
                    "language": lang,
                    "row_order": row_order,
                    "slot_type": "phone",
                    "slot_index": slot_idx,
                    "gap_pos": "",
                    "prompted": p_sym,
                    "uttered": u_sym,
                    "predicted": h_sym,
                    "corr_U": correctness(p_sym, u_sym),
                    "corr_P": correctness(p_sym, h_sym),
                    "bucket": b,
                    "actual_error_type": actual_error_type(p_sym, u_sym),
                    "predicted_error_type": predicted_error_type(p_sym, h_sym),
                    "canonical_seq": " ".join(prompted_segs),
                    "uttered_seq": " ".join(uttered_segs),
                    "predicted_seq": " ".join(predicted_segs),
                }
            )
            row_order += 1

    return rows


def mark_alignment_suspects(rows: list[dict[str, Any]], window: int) -> None:
    by_utt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_utt[row["utt_id"]].append(row)

    for utt_rows in by_utt.values():
        ordered = sorted(utt_rows, key=lambda r: int(r["row_order"]))
        for row in ordered:
            row["nearby_match_window"] = ""
            row["nearby_match_reason"] = ""
            row["global_match_elsewhere"] = ""

            if row["bucket"] not in {"DE", "FA"}:
                continue

            target_sym = row["uttered"]
            if target_sym == MDD_BLANK and row["actual_error_type"] != "deletion":
                continue

            current_order = int(row["row_order"])
            nearby_hits: list[str] = []
            global_hit = False
            for other in ordered:
                if other is row:
                    continue
                same_predicted_error = False
                if row["actual_error_type"] == "deletion":
                    same_predicted_error = (
                        other["slot_type"] == "phone"
                        and other["predicted"] == MDD_BLANK
                        and other["predicted_error_type"] == "deletion"
                    )
                elif target_sym != MDD_BLANK:
                    same_predicted_error = (
                        other["corr_P"] == "E" and other["predicted"] == target_sym
                    )

                if not same_predicted_error:
                    continue

                global_hit = True
                dist = abs(int(other["row_order"]) - current_order)
                if dist <= window:
                    nearby_hits.append(
                        f"{other['slot_type']}:{other['slot_index']}:{other['bucket']}"
                    )

            if global_hit:
                row["global_match_elsewhere"] = "1"
            if nearby_hits:
                row["nearby_match_window"] = str(window)
                row["nearby_match_reason"] = ";".join(nearby_hits[:5])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = [
        "utt_id",
        "language",
        "row_order",
        "slot_type",
        "slot_index",
        "gap_pos",
        "prompted",
        "uttered",
        "predicted",
        "corr_U",
        "corr_P",
        "bucket",
        "actual_error_type",
        "predicted_error_type",
        "nearby_match_window",
        "nearby_match_reason",
        "global_match_elsewhere",
        "canonical_seq",
        "uttered_seq",
        "predicted_seq",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({h: row.get(h, "") for h in headers})


def pair_label(row: dict[str, Any], use_predicted: bool = False) -> str:
    right = row["predicted"] if use_predicted else row["uttered"]
    return f"{row['prompted']}->{right}"


def summarize(rows: list[dict[str, Any]], path: Path) -> None:
    bucket_counts = Counter(row["bucket"] for row in rows)
    actual_error_counts = Counter(
        row["actual_error_type"] for row in rows if row["actual_error_type"] != "none"
    )
    missed_counts = Counter(
        row["actual_error_type"] for row in rows if row["bucket"] in {"DE", "FA"}
    )
    de_pairs = Counter(
        f"{row['prompted']}->{row['uttered']} predicted {row['predicted']}"
        for row in rows
        if row["bucket"] == "DE"
    )
    fa_pairs = Counter(
        pair_label(row) for row in rows if row["bucket"] == "FA"
    )
    fr_pairs = Counter(
        pair_label(row, use_predicted=True) for row in rows if row["bucket"] == "FR"
    )

    tr = bucket_counts["CD"] + bucket_counts["DE"]
    denom = tr + bucket_counts["FA"]
    true_diag = bucket_counts["CD"] / denom if denom else 0.0
    diag_acc = bucket_counts["CD"] / tr if tr else 0.0

    missed = [r for r in rows if r["bucket"] in {"DE", "FA"}]
    nearby = [r for r in missed if r.get("nearby_match_reason")]
    global_elsewhere = [r for r in missed if r.get("global_match_elsewhere")]

    lines: list[str] = []
    lines.append("MDD Error Analysis")
    lines.append("==================")
    lines.append("")
    lines.append("Aggregate counts")
    lines.append(f"  TA={bucket_counts['TA']} TR={tr} FR={bucket_counts['FR']} FA={bucket_counts['FA']}")
    lines.append(f"  CD={bucket_counts['CD']} DE={bucket_counts['DE']}")
    lines.append(f"  Diagnostic_Accuracy = CD / TR = {bucket_counts['CD']} / {tr} = {diag_acc:.4f}")
    lines.append(
        "  True_Diagnostic_Accuracy = CD / (TR + FA) = "
        f"{bucket_counts['CD']} / ({tr} + {bucket_counts['FA']}) = {true_diag:.4f}"
    )
    lines.append("")
    lines.append("Actual pronunciation-error slots")
    for key, value in actual_error_counts.most_common():
        lines.append(f"  {key}: {value}")
    lines.append("")
    lines.append("Missed or wrong-diagnosis actual errors (DE + FA)")
    for key, value in missed_counts.most_common():
        lines.append(f"  {key}: {value}")
    lines.append("")
    lines.append("Possible alignment/slot-shift flags")
    lines.append(f"  DE+FA rows: {len(missed)}")
    lines.append(
        "  Same error symbol predicted as an error elsewhere in utterance: "
        f"{len(global_elsewhere)}"
    )
    lines.append(f"  Same error symbol predicted as an error within local window: {len(nearby)}")
    lines.append("")
    lines.append("Top DE confusions: canonical->uttered predicted predicted")
    for label, count in de_pairs.most_common(15):
        lines.append(f"  {count:>3}  {label}")
    lines.append("")
    lines.append("Top FA missed actual errors: canonical->uttered")
    for label, count in fa_pairs.most_common(15):
        lines.append(f"  {count:>3}  {label}")
    lines.append("")
    lines.append("Top FR false alarms: canonical->predicted")
    for label, count in fr_pairs.most_common(15):
        lines.append(f"  {count:>3}  {label}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-file", type=Path, required=True)
    parser.add_argument("--canonical-file", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--gt-field", default="target")
    parser.add_argument("--pred-field", default="processed_transcript")
    parser.add_argument("--key-field", default="utt_id")
    parser.add_argument("--language-field", default="lang_sym")
    parser.add_argument("--nearby-window", type=int, default=2)
    args = parser.parse_args()

    evaluator = PhoneRecognitionEvaluator(normalize_ipa=True)
    canonical = load_canonical(args.canonical_file)
    rows: list[dict[str, Any]] = []
    missing_canonical = 0

    for item in load_prediction_items(args.prediction_file):
        passthrough = item.get("passthrough", {})
        utt_id = passthrough.get(args.key_field, "")
        if not utt_id:
            continue
        prompted = canonical.get(utt_id, "")
        if not prompted:
            missing_canonical += 1
            continue
        lang = passthrough.get(args.language_field, "combined")
        uttered = passthrough.get(args.gt_field, "")
        predicted = pred_value(item, args.pred_field)
        rows.extend(expand_slots(evaluator, utt_id, lang, prompted, uttered, predicted))

    mark_alignment_suspects(rows, args.nearby_window)
    write_csv(args.out_csv, rows)
    summarize(rows, args.summary)
    print(f"Wrote {len(rows)} slot rows to {args.out_csv}")
    print(f"Wrote summary to {args.summary}")
    if missing_canonical:
        print(f"Skipped {missing_canonical} utterances missing canonical IPA")


if __name__ == "__main__":
    main()
