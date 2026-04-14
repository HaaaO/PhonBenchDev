from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path


def safe_stem(word: str) -> str:
    return re.sub(r"[^\w]+", "_", word, flags=re.UNICODE).strip("_")


def safe_utt_id(pattern_id: str, stem: str) -> str:
    raw = f"{pattern_id}_{stem}"
    return re.sub(r"[^A-Za-z0-9._-]", "_", raw)


def ipa_compact(ipa: str) -> str:
    """Single-token IPA line like the synthetics_kaldi template (no spaces)."""
    return "".join(ipa.split())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="manifest.jsonl from synthesize_phonetic_errors.py",
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        required=True,
        help="Synthesis --output-dir (contains wav/ and manifest.jsonl).",
    )
    parser.add_argument(
        "--export-parent",
        type=Path,
        required=True,
        help="Directory that will contain <dataset_name>/ (use this as data_dir root in Hydra).",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="test_synthetic_phonetic",
        help="Subfolder name and prefix inside wav.scp paths.",
    )
    parser.add_argument(
        "--lang-tag",
        type=str,
        default="<eng><pr>",
        help="Second column in ``text`` (language / task tag).",
    )
    parser.add_argument(
        "--symlink-wavs",
        action="store_true",
        help="Symlink into wavs/ instead of copying (smaller disk use; paths must stay valid).",
    )
    args = parser.parse_args()

    manifest = args.manifest.resolve()
    audio_root = args.audio_root.resolve()
    export_parent = args.export_parent.resolve()
    ds_name = args.dataset_name
    ds_dir = export_parent / ds_name
    wavs_dir = ds_dir / "wavs"

    if not manifest.is_file():
        raise SystemExit(f"Manifest not found: {manifest}")
    if not audio_root.is_dir():
        raise SystemExit(f"audio-root not found: {audio_root}")

    ds_dir.mkdir(parents=True, exist_ok=True)
    wavs_dir.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, str, str, Path]] = []
    seen_utts: set[str] = set()

    with manifest.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("skipped"):
                continue
            ipa = (row.get("ipa_for_tts") or "").strip()
            rel = (row.get("wav_relpath") or "").strip()
            if not ipa or not rel:
                continue
            word = row["word"]
            pid = row["pattern_id"]
            stem = safe_stem(word)
            utt = safe_utt_id(pid, stem)
            if utt in seen_utts:
                raise SystemExit(f"Duplicate utt_id after sanitization: {utt}")
            seen_utts.add(utt)

            src = (audio_root / rel).resolve()
            if not src.is_file():
                raise SystemExit(f"Missing wav for utt={utt}: {src}")

            rows.append((utt, ipa_compact(ipa), rel, src))

    rows.sort(key=lambda x: x[0])

    wav_scp_lines: list[str] = []
    text_good_lines: list[str] = []
    text_lines: list[str] = []

    for utt, ipa, _rel, src in rows:
        dst = wavs_dir / f"{utt}.wav"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if args.symlink_wavs:
            rel_link = os.path.relpath(src, start=dst.parent)
            dst.symlink_to(rel_link)
        else:
            shutil.copy2(src, dst)

        rel_scp = f"{ds_name}/wavs/{utt}.wav"
        wav_scp_lines.append(f"{utt}\t{rel_scp}")
        text_good_lines.append(f"{utt}\t{ipa}")
        text_lines.append(f"{utt}\t{args.lang_tag}")

    (ds_dir / "wav.scp").write_text("\n".join(wav_scp_lines) + "\n", encoding="utf-8")
    (ds_dir / "text.good").write_text("\n".join(text_good_lines) + "\n", encoding="utf-8")
    (ds_dir / "text").write_text("\n".join(text_lines) + "\n", encoding="utf-8")

    print(f"Exported {len(rows)} utterances to {ds_dir}")
    print(f"  wav.scp, text.good, text, wavs/  (export_parent={export_parent})")
    print(
        "Register in configs/data/powsm_evalset_index.yaml under datasets:, e.g.\n"
        f"  {ds_name}:\n"
        f'    wav_scp: "{ds_name}/wav.scp"\n'
        f'    text_phoneme: "{ds_name}/text.good"\n'
        f'    language: "{ds_name}/text"'
    )


if __name__ == "__main__":
    main()
