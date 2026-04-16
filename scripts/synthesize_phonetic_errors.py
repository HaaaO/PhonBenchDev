from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Union

AudioLayout = Literal["flat", "by_pattern"]
TTSBackend = Literal["espeak", "polly", "kokoro"]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.ipa_utils import ARPABET_TO_IPA  # noqa: E402

VOWEL_BASES = frozenset(
    {
        "AA",
        "AE",
        "AH",
        "AO",
        "AW",
        "AY",
        "EH",
        "ER",
        "EY",
        "IH",
        "IY",
        "OW",
        "OY",
        "UH",
        "UW",
        "AX",
        "AXR",
        "IX",
    }
)

VOICING_PAIRS = {
    "P": "B",
    "B": "P",
    "T": "D",
    "D": "T",
    "K": "G",
    "G": "K",
    "F": "V",
    "V": "F",
    "S": "Z",
    "Z": "S",
    "SH": "ZH",
    "ZH": "SH",
    "CH": "JH",
    "JH": "CH",
    "TH": "DH",
    "DH": "TH",
}


def split_stress(token: str) -> tuple[str, Optional[str]]:
    if token and token[-1] in "012":
        return token[:-1], token[-1]
    return token, None


def base_is_vowel(base: str) -> bool:
    return base in VOWEL_BASES


def is_consonant_token(token: str) -> bool:
    base, _ = split_stress(token)
    return not base_is_vowel(base)


def tokenize_g2p(g2p_out: Union[str, List[str]]) -> List[str]:
    """Normalize g2p_en output (some versions return a str, others a list of phones)."""
    if isinstance(g2p_out, list):
        return [str(p).strip() for p in g2p_out if str(p).strip()]
    return [p for p in str(g2p_out).strip().split() if p]


def first_vowel_index(tokens: List[str]) -> Optional[int]:
    for i, t in enumerate(tokens):
        base, _ = split_stress(t)
        if base_is_vowel(base):
            return i
    return None


def arpabet_base_to_ipa(base: str) -> str:
    key = base.lower()
    ipa = ARPABET_TO_IPA.get(key)
    if ipa is not None:
        return ipa
    return ARPABET_TO_IPA.get(base, "")


def tokens_to_ipa(tokens: List[str]) -> str:
    """Concatenate IPA for espeak phoneme attribute (stress on first primary / first secondary)."""
    parts: List[str] = []
    primary_marks = 0
    secondary_marks = 0
    for t in tokens:
        base, stress = split_stress(t)
        ipa = arpabet_base_to_ipa(base)
        if not ipa:
            ipa = base.lower()
        if stress == "1" and base_is_vowel(base) and primary_marks == 0:
            ipa = "\u02C8" + ipa
            primary_marks += 1
        elif stress == "2" and base_is_vowel(base) and secondary_marks == 0:
            ipa = "\u02CC" + ipa
            secondary_marks += 1
        parts.append(ipa)
    return "".join(parts)


def subst_first_map(tokens: List[str], cmap: Dict[str, str]) -> Optional[List[str]]:
    out = list(tokens)
    for i, t in enumerate(out):
        base, stress = split_stress(t)
        if not is_consonant_token(t):
            continue
        if base in cmap:
            nb = cmap[base]
            out[i] = nb + (stress or "")
            return out
    return None


def pat_voicing_flip(tokens: List[str]) -> Optional[List[str]]:
    return subst_first_map(tokens, {k: v for k, v in VOICING_PAIRS.items()})


def pat_stopping(tokens: List[str]) -> Optional[List[str]]:
    cmap = {"S": "T", "Z": "D", "SH": "T", "ZH": "D", "F": "P", "V": "B", "TH": "T", "DH": "D"}
    return subst_first_map(tokens, cmap)


def pat_deaffricate(tokens: List[str]) -> Optional[List[str]]:
    return subst_first_map(tokens, {"CH": "SH", "JH": "ZH"})


def pat_fronting(tokens: List[str]) -> Optional[List[str]]:
    return subst_first_map(tokens, {"K": "T", "G": "D", "NG": "N"})


def pat_glide_r(tokens: List[str]) -> Optional[List[str]]:
    return subst_first_map(tokens, {"R": "W"})


def pat_glide_l(tokens: List[str]) -> Optional[List[str]]:
    return subst_first_map(tokens, {"L": "W"})


def pat_r_to_l(tokens: List[str]) -> Optional[List[str]]:
    return subst_first_map(tokens, {"R": "L"})


def pat_l_to_r(tokens: List[str]) -> Optional[List[str]]:
    return subst_first_map(tokens, {"L": "R"})


def pat_s_sh_confuse(tokens: List[str]) -> Optional[List[str]]:
    out = list(tokens)
    for i, t in enumerate(out):
        base, stress = split_stress(t)
        if base == "S":
            out[i] = "SH" + (stress or "")
            return out
        if base == "SH":
            out[i] = "S" + (stress or "")
            return out
    return None


def pat_nasal_place(tokens: List[str]) -> Optional[List[str]]:
    return subst_first_map(tokens, {"N": "M", "M": "N"})


def pat_denasal_final(tokens: List[str]) -> Optional[List[str]]:
    """Replace word-final N→D or M→B. Does not apply to NG (e.g. -ing)."""
    if not tokens:
        return None
    last = tokens[-1]
    if not is_consonant_token(last):
        return None
    base, stress = split_stress(last)
    if base == "N":
        out = list(tokens)
        out[-1] = "D" + (stress or "")
        return out
    if base == "M":
        out = list(tokens)
        out[-1] = "B" + (stress or "")
        return out
    return None


def pat_omit_final_coda(tokens: List[str]) -> Optional[List[str]]:
    if not tokens:
        return None
    last = tokens[-1]
    if not is_consonant_token(last):
        return None
    return tokens[:-1]


def pat_omit_initial_onset(tokens: List[str]) -> Optional[List[str]]:
    if not tokens:
        return None
    if not is_consonant_token(tokens[0]):
        return None
    return tokens[1:]


def pat_omit_cluster_second(tokens: List[str]) -> Optional[List[str]]:
    j = first_vowel_index(tokens)
    if j is None or j < 2:
        return None
    if not is_consonant_token(tokens[0]) or not is_consonant_token(tokens[1]):
        return None
    out = list(tokens)
    del out[1]
    return out


def pat_epenthesis_cc_onset(tokens: List[str]) -> Optional[List[str]]:
    j = first_vowel_index(tokens)
    if j is None or j < 2:
        return None
    if not is_consonant_token(tokens[0]) or not is_consonant_token(tokens[1]):
        return None
    out = list(tokens)
    out.insert(1, "AH0")
    return out


def pat_affricate_t(tokens: List[str]) -> Optional[List[str]]:
    return subst_first_map(tokens, {"T": "CH"})


def pat_backing_t(tokens: List[str]) -> Optional[List[str]]:
    return subst_first_map(tokens, {"T": "K", "D": "G", "N": "NG"})


PATTERN_FUNCS: Dict[str, Callable[[List[str]], Optional[List[str]]]] = {
    "subst_voicing_flip": pat_voicing_flip,
    "subst_stopping": pat_stopping,
    "subst_deaffricate": pat_deaffricate,
    "subst_fronting": pat_fronting,
    "subst_glide_r_to_w": pat_glide_r,
    "subst_glide_l_to_w": pat_glide_l,
    "subst_r_to_l": pat_r_to_l,
    "subst_l_to_r": pat_l_to_r,
    "subst_s_sh": pat_s_sh_confuse,
    "subst_nasal_nm": pat_nasal_place,
    "subst_denasal_final": pat_denasal_final,
    "omit_final_coda": pat_omit_final_coda,
    "omit_initial_onset": pat_omit_initial_onset,
    "omit_cluster_second": pat_omit_cluster_second,
    "add_epenthesis_cc_onset": pat_epenthesis_cc_onset,
    "subst_affricate_t_to_ch": pat_affricate_t,
    "subst_backing_coronal_to_velar": pat_backing_t,
    "canonical": lambda t: list(t),
}
OPTIONAL_CANONICAL = "canonical"


@dataclass
class ManifestRow:
    word: str
    g2p_input: str
    pattern_id: str
    canonical_arpabet: str
    modified_arpabet: str
    ipa_for_tts: str
    skipped: bool
    reason: str
    wav_relpath: str


def g2p_word(g2p, surface: str) -> List[str]:
    s = re.sub(r"[-–]", " ", surface)
    return tokenize_g2p(g2p(s))


def escape_ipa_for_ssml(ipa: str) -> str:
    return ipa.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&#34;")


def ipa_ssml_body(ipa: str) -> str:
    ipa_esc = escape_ipa_for_ssml(ipa)
    return (
        '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">\n'
        f'  <phoneme alphabet="ipa" ph="{ipa_esc}">word</phoneme>\n'
        "</speak>\n"
    )


def synthesize_espeak_ipa(ipa: str, wav_path: Path, voice: str = "en-us") -> None:
    espeak = shutil.which("espeak-ng") or shutil.which("espeak")
    if not espeak:
        raise RuntimeError("espeak-ng (or espeak) not found on PATH.")
    ssml = '<?xml version="1.0"?>\n' + ipa_ssml_body(ipa)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".ssml", delete=False) as f:
        f.write(ssml)
        tmp = f.name
    try:
        cmd = [espeak, "-m", "-f", tmp, "-w", str(wav_path), "-v", voice]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    finally:
        Path(tmp).unlink(missing_ok=True)


def _pcm_to_wav(pcm: bytes, wav_path: Path, sample_rate: int = 16000) -> None:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


def synthesize_polly_ipa(
    ipa: str,
    wav_path: Path,
    *,
    voice_id: str,
    engine: str,
    region: str,
    sample_rate: str = "16000",
) -> None:
    try:
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError as e:
        raise RuntimeError("Install boto3 for Polly (see requirements.txt).") from e

    polly = boto3.client("polly", region_name=region)
    ssml = ipa_ssml_body(ipa)
    try:
        resp = polly.synthesize_speech(
            Text=ssml,
            TextType="ssml",
            OutputFormat="pcm",
            SampleRate=sample_rate,
            VoiceId=voice_id,
            Engine=engine,
        )
    except (ClientError, BotoCoreError) as e:
        raise RuntimeError(f"Polly synthesize_speech failed: {e}") from e

    stream = resp.get("AudioStream")
    if stream is None:
        raise RuntimeError("Polly returned no AudioStream.")
    pcm = stream.read()
    _pcm_to_wav(pcm, wav_path, sample_rate=int(sample_rate))

def synthesize_kokoro_ipa(
    ipa: str,
    wav_path: Path,
    *,
    voice: str = "af_heart",
    speed: float = 1.0,
    sample_rate: int = 24000,
    pipeline: Optional[Any] = None,
) -> None:
    try:
        from kokoro import KPipeline
        import soundfile as sf
        import numpy as np
    except ImportError as e:
        raise RuntimeError("Install kokoro and soundfile: pip install 'kokoro>=0.9.2' soundfile") from e

    if pipeline is None:
        pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    # Raw IPA phoneme string (Kokoro 0.9+: generate_from_tokens, not __call__(..., ps=...))
    chunks = []
    for result in pipeline.generate_from_tokens(ipa, voice=voice, speed=speed):
        audio = result.audio
        if audio is not None and len(audio) > 0:
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()
            chunks.append(np.asarray(audio, dtype=np.float32))

    if not chunks:
        raise RuntimeError(f"Kokoro produced no audio for IPA: {ipa!r}")

    combined = np.concatenate(chunks)
    sf.write(str(wav_path), combined, sample_rate)

def resolve_wav_path(
    wav_dir: Path, layout: AudioLayout, pattern_id: str, stem: str
) -> tuple[Path, str]:
    if layout == "by_pattern":
        path = wav_dir / pattern_id / f"{stem}.wav"
        rel = f"wav/{pattern_id}/{stem}.wav"
    else:
        path = wav_dir / f"{stem}__{pattern_id}.wav"
        rel = f"wav/{stem}__{pattern_id}.wav"
    return path, rel


def safe_stem(word: str) -> str:
    return re.sub(r"[^\w]+", "_", word, flags=re.UNICODE).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "AWS Polly: configure SSO once (`aws configure sso`, region us-west-2), then "
            "`aws sso login` before running with `--tts-backend polly`. "
            "Default Polly engine is generative with PCM at 16 kHz (AWS allows only 8000/16000 for pcm). "
            "Use --polly-engine neural if a voice rejects generative. "
            "Do not paste temporary access keys into the repo or chat—rotate them if exposed."
        ),
    )
    parser.add_argument(
        "--word-list",
        type=Path,
        default=PROJECT_ROOT / "data/word_lists/merged_decoding_words.json",
        help="JSON with a top-level 'words' array of {\"word\": ..., \"sources\": ...}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Root for manifest.jsonl + wav/ (see --audio-layout). "
        "Default depends on --tts-backend: outputs/synthetic_phonetic (espeak/polly) vs "
        "outputs/synthetic_phonetic_kokoro (kokoro), so backends do not clobber each other.",
    )
    parser.add_argument(
        "--patterns",
        type=str,
        default=",".join(sorted(k for k in PATTERN_FUNCS if k != OPTIONAL_CANONICAL)),
        help="Comma-separated pattern ids (see PATTERN_FUNCS keys; 'canonical' is identity).",
    )
    parser.add_argument(
        "--include-canonical",
        action="store_true",
        help=f'Also emit pattern "{OPTIONAL_CANONICAL}" (identity mapping) if listed in --patterns.',
    )
    parser.add_argument(
        "--synthesize",
        action="store_true",
        help="If set, render WAVs (see --tts-backend). Otherwise manifest only.",
    )
    parser.add_argument(
        "--tts-backend",
        type=str,
        choices=["espeak", "polly", "kokoro"],
        default="espeak",
        help="espeak-ng, Amazon Polly, or Kokoro (pip install kokoro soundfile; Python <3.13).",
    )
    parser.add_argument("--kokoro-voice", type=str, default="af_heart",
        help="Kokoro voice id (e.g. af_heart, af_bella, am_adam).")
    parser.add_argument("--kokoro-speed", type=float, default=1.0,
        help="Kokoro speech rate multiplier.")
    parser.add_argument(
        "--audio-layout",
        type=str,
        choices=["flat", "by_pattern"],
        default="by_pattern",
        help="by_pattern: wav/<pattern_id>/<word_stem>.wav ; flat: wav/<stem>__<pattern>.wav",
    )
    parser.add_argument("--voice", type=str, default="en-us", help="espeak-ng -v voice code.")
    parser.add_argument(
        "--polly-voice-id",
        type=str,
        default="Joanna",
        help="Amazon Polly VoiceId (e.g. Joanna, Matthew, Ruth).",
    )
    parser.add_argument(
        "--polly-engine",
        type=str,
        default="generative",
        help="Polly engine: generative (default) | neural | standard | long-form (voice/region dependent).",
    )
    parser.add_argument(
        "--polly-sample-rate",
        type=str,
        default="16000",
        choices=["8000", "16000"],
        help="PCM sample rate for Polly (OutputFormat=pcm allows only 8000 or 16000 per AWS API).",
    )
    parser.add_argument(
        "--aws-region",
        type=str,
        default="us-west-2",
        help="AWS region for Polly (e.g. us-west-2).",
    )
    parser.add_argument("--limit", type=int, default=0, help="If >0, only first N words.")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = (
            PROJECT_ROOT / "outputs/synthetic_phonetic_kokoro"
            if args.tts_backend == "kokoro"
            else PROJECT_ROOT / "outputs/synthetic_phonetic"
        )

    try:
        from g2p_en import G2p
    except ImportError as e:
        raise SystemExit("Missing dependency: g2p_en (install project requirements.txt).") from e

    g2p = G2p()
    raw = json.loads(args.word_list.read_text(encoding="utf-8"))
    entries = raw["words"]
    if args.limit > 0:
        entries = entries[: args.limit]

    pattern_ids = [p.strip() for p in args.patterns.split(",") if p.strip()]
    if args.include_canonical and OPTIONAL_CANONICAL not in pattern_ids:
        pattern_ids.append(OPTIONAL_CANONICAL)

    unknown = [p for p in pattern_ids if p not in PATTERN_FUNCS]
    if unknown:
        raise SystemExit(f"Unknown pattern ids: {unknown}")

    out_dir = args.output_dir
    wav_dir = out_dir / "wav"
    out_dir.mkdir(parents=True, exist_ok=True)

    layout: AudioLayout = args.audio_layout  # type: ignore[assignment]
    backend: TTSBackend = args.tts_backend  # type: ignore[assignment]

    manifest_path = out_dir / "manifest.jsonl"
    rows_written = 0
    kokoro_pipeline = None
    if args.synthesize and backend == "kokoro":
        try:
            from kokoro import KPipeline
        except ImportError as e:
            raise SystemExit(
                "Kokoro backend requires: pip install 'kokoro>=0.9.2' soundfile "
                "(Python 3.10–3.12; kokoro does not support 3.13+ yet)."
            ) from e
        kokoro_pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")

    with manifest_path.open("w", encoding="utf-8") as mf:
        for entry in entries:
            word = entry["word"]
            g2p_in = word
            canon = g2p_word(g2p, word)
            if not canon:
                row = ManifestRow(
                    word=word,
                    g2p_input=g2p_in,
                    pattern_id="",
                    canonical_arpabet="",
                    modified_arpabet="",
                    ipa_for_tts="",
                    skipped=True,
                    reason="empty_g2p",
                    wav_relpath="",
                )
                mf.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
                rows_written += 1
                continue

            for pid in pattern_ids:
                fn = PATTERN_FUNCS[pid]
                modified = fn(canon)

                if modified is None:
                    row = ManifestRow(
                        word=word,
                        g2p_input=g2p_in,
                        pattern_id=pid,
                        canonical_arpabet=" ".join(canon),
                        modified_arpabet="",
                        ipa_for_tts="",
                        skipped=True,
                        reason="pattern_not_applicable",
                        wav_relpath="",
                    )
                    mf.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
                    rows_written += 1
                    continue

                ipa = tokens_to_ipa(modified)
                stem = safe_stem(word)
                wav_path, rel = resolve_wav_path(wav_dir, layout, pid, stem)

                if args.synthesize:
                    try:
                        if backend == "polly":
                            synthesize_polly_ipa(
                                ipa,
                                wav_path,
                                voice_id=args.polly_voice_id,
                                engine=args.polly_engine,
                                region=args.aws_region,
                                sample_rate=args.polly_sample_rate,
                            )
                        elif backend == "kokoro":
                            synthesize_kokoro_ipa(
                                ipa,
                                wav_path,
                                voice=args.kokoro_voice,
                                speed=args.kokoro_speed,
                                pipeline=kokoro_pipeline,
                            )
                        else:
                            synthesize_espeak_ipa(ipa, wav_path, voice=args.voice)
                    except (RuntimeError, subprocess.CalledProcessError) as e:
                        row = ManifestRow(
                            word=word,
                            g2p_input=g2p_in,
                            pattern_id=pid,
                            canonical_arpabet=" ".join(canon),
                            modified_arpabet=" ".join(modified),
                            ipa_for_tts=ipa,
                            skipped=True,
                            reason=f"synth_failed:{e}",
                            wav_relpath=rel,
                        )
                        mf.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
                        rows_written += 1
                        continue

                row = ManifestRow(
                    word=word,
                    g2p_input=g2p_in,
                    pattern_id=pid,
                    canonical_arpabet=" ".join(canon),
                    modified_arpabet=" ".join(modified),
                    ipa_for_tts=ipa,
                    skipped=False,
                    reason="",
                    wav_relpath=rel,
                )
                mf.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
                rows_written += 1

    print(f"Wrote {rows_written} manifest lines to {manifest_path}")
    if not args.synthesize:
        print(
            "Synthesis skipped (pass --synthesize; use --tts-backend polly for AWS Polly "
            "or default espeak-ng)."
        )
    else:
        print(f"Audio layout: {layout}, backend: {backend}, files under {wav_dir}/")


if __name__ == "__main__":
    main()
