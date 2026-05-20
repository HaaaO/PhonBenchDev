"""IPA normalization helpers for PhonBench evaluation outputs."""

from __future__ import annotations

import copy
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, MutableMapping, Tuple


DEFAULT_IPA_NORMALIZATION_PROFILE = "ipa_eng_broad_v1"
SUPPORTED_IPA_NORMALIZATION_PROFILES = {DEFAULT_IPA_NORMALIZATION_PROFILE}

_TIE_BAR = "\u0361"
_TIE_BAR_BELOW = "\u035c"
_DENTAL_DIACRITIC = "\u032a"
_DOT_BELOW_DIACRITIC = "\u0323"
_CANDRABINDU_DIACRITIC = "\u0310"
_STRESS_MARKS = ("ˈ", "ˌ")

_ASCII_IPA_TOKENS = {
    "a",
    "b",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "z",
}

_IPA_HINT_CHARS = set(
    "ɑæʌɔəɚɝɛɡɪŋɹʃθʊʒɜɫɕʂʑʐʈɖðʰː˞ˈˌʤʧıε"
    + _TIE_BAR
    + _TIE_BAR_BELOW
    + _DENTAL_DIACRITIC
    + _DOT_BELOW_DIACRITIC
    + _CANDRABINDU_DIACRITIC
)


@dataclass(frozen=True)
class IPANormalizationResult:
    raw: str
    normalized: str
    profile: str
    changed: bool
    rule_ids: Tuple[str, ...]


def _mark(rule_ids: List[str], rule_id: str) -> None:
    if rule_id not in rule_ids:
        rule_ids.append(rule_id)


def _replace(text: str, old: str, new: str, rule_id: str, rule_ids: List[str]) -> str:
    updated = text.replace(old, new)
    if updated != text:
        _mark(rule_ids, rule_id)
    return updated


def _sub(text: str, pattern: str, repl: str, rule_id: str, rule_ids: List[str]) -> str:
    updated = re.sub(pattern, repl, text)
    if updated != text:
        _mark(rule_ids, rule_id)
    return updated


def looks_ipa_like(value: Any) -> bool:
    """Heuristic for deciding whether a free-form field is likely IPA text.

    This is intentionally conservative for raw ``predicted_transcript`` fields,
    which may contain natural-language model chatter or JSON for non-phone
    tasks. Named IPA fields such as ``processed_transcript`` and
    ``canonical_ipa`` are normalized by field contract instead of relying on
    this heuristic.
    """
    if not isinstance(value, str):
        return False

    stripped = value.strip()
    if not stripped:
        return True

    ascii_words = re.findall(r"[A-Za-z]+", stripped)
    if any(len(word) > 1 for word in ascii_words):
        return False

    if any(ch in _IPA_HINT_CHARS or unicodedata.category(ch).startswith("M") for ch in stripped):
        return True

    tokens = stripped.split()
    if len(tokens) > 1 and all(token in _ASCII_IPA_TOKENS for token in tokens):
        return True

    return False


def normalize_ipa_text(
    text: str,
    profile: str = DEFAULT_IPA_NORMALIZATION_PROFILE,
) -> IPANormalizationResult:
    """Normalize IPA notation under an English broad-phone profile."""
    if profile not in SUPPORTED_IPA_NORMALIZATION_PROFILES:
        supported = ", ".join(sorted(SUPPORTED_IPA_NORMALIZATION_PROFILES))
        raise ValueError(f"Unsupported IPA normalization profile: {profile}. Supported: {supported}")

    raw = text
    rule_ids: List[str] = []
    normalized = unicodedata.normalize("NFD", text)
    if normalized != text:
        _mark(rule_ids, "unicode_nfd")

    # Representation fixes that improve downstream IPA segmentation.
    normalized = _replace(normalized, "AUDIOGAP", "", "drop_audio_gap_marker", rule_ids)
    normalized = _replace(normalized, "ɚ", "ə˞", "rhotic_schwa_decompose", rule_ids)
    normalized = _replace(normalized, "ɝ", "ɜ˞", "rhotic_schwa_decompose", rule_ids)
    normalized = _replace(normalized, "ı", "ɪ", "dotless_i_to_ipa_small_cap_i", rule_ids)
    normalized = _replace(normalized, "ε", "ɛ", "greek_epsilon_to_open_e", rule_ids)
    normalized = _replace(normalized, "g", "ɡ", "latin_g_to_ipa_g", rule_ids)

    # English-specific affricate inventory normalization.
    normalized = _sub(normalized, rf"t\s*ɕʰ?", f"t{_TIE_BAR}ʃ", "eng_t_alveolopalatal_to_tesh", rule_ids)
    normalized = _sub(normalized, rf"ʈ\s*ʂʰ?", f"t{_TIE_BAR}ʃ", "eng_retroflex_to_tesh", rule_ids)
    normalized = _sub(normalized, rf"d\s*ʑ", f"d{_TIE_BAR}ʒ", "eng_d_alveolopalatal_to_dezh", rule_ids)
    normalized = _sub(normalized, rf"ɖ\s*ʐ", f"d{_TIE_BAR}ʒ", "eng_retroflex_to_dezh", rule_ids)

    # IPA notation variants for the English affricates.
    normalized = _replace(normalized, "ʧ", f"t{_TIE_BAR}ʃ", "tesh_digraph_to_tied", rule_ids)
    normalized = _replace(normalized, "ʤ", f"d{_TIE_BAR}ʒ", "dezh_digraph_to_tied", rule_ids)
    normalized = _replace(normalized, f"t{_TIE_BAR_BELOW}ʃ", f"t{_TIE_BAR}ʃ", "tie_bar_below_to_above", rule_ids)
    normalized = _replace(normalized, f"d{_TIE_BAR_BELOW}ʒ", f"d{_TIE_BAR}ʒ", "tie_bar_below_to_above", rule_ids)
    normalized = _sub(normalized, r"t\s*ʃ", f"t{_TIE_BAR}ʃ", "untied_tesh_to_tied", rule_ids)
    normalized = _sub(normalized, r"d\s*ʒ", f"d{_TIE_BAR}ʒ", "untied_dezh_to_tied", rule_ids)

    # English broad allophone/vocabulary normalization.
    normalized = _replace(normalized, "ɕ", "ʃ", "eng_alveolopalatal_fricative_to_esh", rule_ids)
    normalized = _replace(normalized, "ʂ", "ʃ", "eng_retroflex_fricative_to_esh", rule_ids)
    normalized = _replace(normalized, "ɫ", "l", "dark_l_to_l", rule_ids)
    normalized = _replace(normalized, "r", "ɹ", "r_to_english_approximant", rule_ids)

    for aspirated, base in (("pʰ", "p"), ("tʰ", "t"), ("kʰ", "k")):
        normalized = _replace(normalized, aspirated, base, "strip_stop_aspiration", rule_ids)

    normalized = _replace(normalized, "ː", "", "strip_length_mark", rule_ids)
    for stress_mark in _STRESS_MARKS:
        normalized = _replace(normalized, stress_mark, "", "strip_stress_mark", rule_ids)
    normalized = _replace(normalized, _DENTAL_DIACRITIC, "", "strip_dental_diacritic", rule_ids)
    normalized = _replace(normalized, _DOT_BELOW_DIACRITIC, "", "strip_dot_below_diacritic", rule_ids)
    normalized = _replace(normalized, _CANDRABINDU_DIACRITIC, "", "strip_candrabindu_diacritic", rule_ids)
    normalized = _replace(normalized, "˺", "", "strip_tone_letter", rule_ids)

    return IPANormalizationResult(
        raw=raw,
        normalized=normalized,
        profile=profile,
        changed=normalized != raw,
        rule_ids=tuple(rule_ids),
    )


def _utt_id(sample_key: str, item: MutableMapping[str, Any]) -> str:
    passthrough = item.get("passthrough")
    if isinstance(passthrough, dict):
        for key in ("utt_id", "key", "metadata_idx"):
            if key in passthrough:
                return str(passthrough[key])
    return str(sample_key)


def _report_row(
    utt_id: str,
    field: str,
    result: IPANormalizationResult,
) -> Dict[str, Any]:
    return {
        "utt_id": utt_id,
        "field": field,
        "raw": result.raw,
        "normalized": result.normalized,
        "profile": result.profile,
        "changed": result.changed,
        "rule_ids": ";".join(result.rule_ids),
    }


def _normalize_mapping_field(
    mapping: MutableMapping[str, Any],
    field: str,
    field_path: str,
    utt_id: str,
    report_rows: List[Dict[str, Any]],
    profile: str,
    *,
    force: bool,
    allow_single_ascii_phone: bool = False,
) -> None:
    if field not in mapping:
        return

    raw_value = mapping[field]
    if not isinstance(raw_value, str):
        return
    should_normalize = looks_ipa_like(raw_value)
    if allow_single_ascii_phone and raw_value.strip() in _ASCII_IPA_TOKENS:
        should_normalize = True
    if not force and not should_normalize:
        return

    result = normalize_ipa_text(raw_value, profile=profile)
    mapping.setdefault(f"{field}_raw", raw_value)
    mapping[field] = result.normalized
    report_rows.append(_report_row(utt_id, field_path, result))


def normalize_transcription_item(
    sample_key: str,
    item: Dict[str, Any],
    profile: str = DEFAULT_IPA_NORMALIZATION_PROFILE,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Normalize one distributed-inference item and return audit rows."""
    normalized_item = copy.deepcopy(item)
    report_rows: List[Dict[str, Any]] = []
    uid = _utt_id(sample_key, normalized_item)

    pred = normalized_item.get("pred")
    pred_entries: List[Tuple[str, MutableMapping[str, Any]]] = []
    if isinstance(pred, list):
        pred_entries = [
            (f"pred[{i}]", entry)
            for i, entry in enumerate(pred)
            if isinstance(entry, dict)
        ]
    elif isinstance(pred, dict):
        pred_entries = [("pred", pred)]

    for pred_path, entry in pred_entries:
        _normalize_mapping_field(
            entry,
            "processed_transcript",
            f"{pred_path}.processed_transcript",
            uid,
            report_rows,
            profile,
            force=True,
        )
        _normalize_mapping_field(
            entry,
            "predicted_transcript",
            f"{pred_path}.predicted_transcript",
            uid,
            report_rows,
            profile,
            force=False,
        )

    passthrough = normalized_item.get("passthrough")
    if isinstance(passthrough, dict):
        _normalize_mapping_field(
            passthrough,
            "target",
            "passthrough.target",
            uid,
            report_rows,
            profile,
            force=False,
            allow_single_ascii_phone=True,
        )
        _normalize_mapping_field(
            passthrough,
            "canonical_ipa",
            "passthrough.canonical_ipa",
            uid,
            report_rows,
            profile,
            force=True,
        )

    normalized_item["normalization_profile"] = profile
    return normalized_item, report_rows


def normalize_transcription_data(
    data: Dict[str, Any],
    profile: str = DEFAULT_IPA_NORMALIZATION_PROFILE,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Normalize a merged transcription JSON object."""
    normalized: Dict[str, Any] = {}
    report_rows: List[Dict[str, Any]] = []
    for sample_key, item in data.items():
        if not isinstance(item, dict):
            normalized[sample_key] = copy.deepcopy(item)
            continue
        normalized_item, item_rows = normalize_transcription_item(sample_key, item, profile=profile)
        normalized[sample_key] = normalized_item
        report_rows.extend(item_rows)
    return normalized, report_rows
