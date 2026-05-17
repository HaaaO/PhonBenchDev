"""Project IPA phone strings onto a CMU39-style IPA inventory."""

from __future__ import annotations

import string
from typing import Dict, List, Optional, Sequence, Tuple

from src.core.ipa_normalization import normalize_ipa_text

CMU39_CONSONANTS: Tuple[str, ...] = (
    "p",
    "b",
    "t",
    "d",
    "k",
    "ɡ",
    "t͡ʃ",
    "d͡ʒ",
    "f",
    "v",
    "θ",
    "ð",
    "s",
    "z",
    "ʃ",
    "ʒ",
    "h",
    "m",
    "n",
    "ŋ",
    "l",
    "ɹ",
    "w",
    "j",
)

CMU39_VOWELS: Tuple[str, ...] = (
    "i",
    "ɪ",
    "eɪ",
    "ɛ",
    "æ",
    "ɑ",
    "ɔ",
    "oʊ",
    "ʊ",
    "u",
    "ʌ",
    "ə˞",
    "aɪ",
    "aʊ",
    "ɔɪ",
)

CMU39_IPA: Tuple[str, ...] = CMU39_CONSONANTS + CMU39_VOWELS
CMU39_IPA_SET = set(CMU39_IPA)
CMU39_PROJECTION_PROFILE = "ga39_canonical_context_v1"

_FLAP_TOKENS = {"ɾ", "ɾ̃"}

_ALIASES: Dict[str, str] = {
    # CMU stressless vowel collapses.
    "ə": "ʌ",
    "ɨ": "ɪ",
    "ɚ": "ə˞",
    "ɝ": "ə˞",
    "ɜ˞": "ə˞",
    "e": "eɪ",
    "o": "oʊ",
    # Common reduced or allophonic consonant forms.
    "l̩": "l",
    "m̩": "m",
    "n̩": "n",
    "ŋ̩": "ŋ",
    "ʉ": "u",
    "tʃ": "t͡ʃ",
    "dʒ": "d͡ʒ",
    "t͜ʃ": "t͡ʃ",
    "d͜ʒ": "d͡ʒ",
}

_DIRECT_MAP: Dict[str, str] = {phone: phone for phone in CMU39_IPA}
_DIRECT_MAP.update(_ALIASES)

_SCAN_TOKENS = tuple(
    sorted(
        set(_DIRECT_MAP) | _FLAP_TOKENS,
        key=len,
        reverse=True,
    )
)


def _clean_for_scan(text: str) -> str:
    normalized = normalize_ipa_text(text).normalized
    return "".join(
        ch
        for ch in normalized
        if not ch.isspace() and ch not in string.punctuation
    )


def segment_ipa_for_cmu39(text: str) -> List[str]:
    """Greedily segment an IPA string using CMU39 phones and known aliases.

    The scanner intentionally keeps English diphthongs such as ``aɪ`` and
    ``oʊ`` atomic. Unknown symbols are skipped; projection will therefore never
    emit phones outside the CMU39 inventory.
    """
    compact = _clean_for_scan(text)
    out: List[str] = []
    idx = 0
    while idx < len(compact):
        matched = False
        for token in _SCAN_TOKENS:
            if compact.startswith(token, idx):
                out.append(token)
                idx += len(token)
                matched = True
                break
        if not matched:
            idx += 1
    return out


def _fallback_projection(token: str) -> Optional[str]:
    if token in _FLAP_TOKENS:
        return "t"
    return _DIRECT_MAP.get(token)


def _align_context(
    canonical: Sequence[str],
    sequence: Sequence[str],
    raw_sequence: Optional[Sequence[str]] = None,
) -> List[Optional[str]]:
    """Return the aligned canonical phone for each projected sequence phone."""
    if not sequence:
        return []
    if not canonical:
        return [None] * len(sequence)

    n_c = len(canonical)
    n_s = len(sequence)
    dp = [[0] * (n_s + 1) for _ in range(n_c + 1)]
    back: List[List[Optional[str]]] = [
        [None] * (n_s + 1) for _ in range(n_c + 1)
    ]
    for i in range(1, n_c + 1):
        dp[i][0] = i
        back[i][0] = "delete"
    for j in range(1, n_s + 1):
        dp[0][j] = j
        back[0][j] = "insert"

    for i in range(1, n_c + 1):
        for j in range(1, n_s + 1):
            raw = raw_sequence[j - 1] if raw_sequence is not None else sequence[j - 1]
            is_flap_match = raw in _FLAP_TOKENS and canonical[i - 1] in {"t", "d"}
            sub_cost = (
                0
                if canonical[i - 1] == sequence[j - 1] or is_flap_match
                else 1
            )
            diag_priority = 0 if sub_cost == 0 else 3
            candidates = (
                (dp[i - 1][j - 1] + sub_cost, diag_priority, "diag"),
                (dp[i - 1][j] + 1, 1, "delete"),
                (dp[i][j - 1] + 1, 2, "insert"),
            )
            cost, _, step = min(candidates, key=lambda item: (item[0], item[1]))
            dp[i][j] = cost
            back[i][j] = step

    contexts: List[Optional[str]] = [None] * n_s
    i, j = n_c, n_s
    while i > 0 or j > 0:
        step = back[i][j]
        if step == "diag":
            contexts[j - 1] = canonical[i - 1]
            i -= 1
            j -= 1
        elif step == "insert":
            contexts[j - 1] = None
            j -= 1
        elif step == "delete":
            i -= 1
        else:
            break
    return contexts


def _project_tokens(
    tokens: Sequence[str],
    canonical_context: Optional[Sequence[str]],
) -> List[str]:
    projected_items: List[Tuple[str, str]] = []
    for token in tokens:
        projected = _fallback_projection(token)
        if projected is not None:
            projected_items.append((token, projected))

    tentative = [projected for _, projected in projected_items]
    raw_items = [raw for raw, _ in projected_items]
    contexts = (
        _align_context(canonical_context, tentative, raw_items)
        if canonical_context is not None
        else [None] * len(tentative)
    )

    out: List[str] = []
    for (raw, projected), context in zip(projected_items, contexts):
        if raw in _FLAP_TOKENS:
            out.append(context if context in {"t", "d"} else "t")
        else:
            out.append(projected)
    return out


def project_ipa_to_cmu39(
    ipa_sequence: str,
    canonical_ipa: str,
    *,
    separator: str = " ",
) -> str:
    """Project one IPA sequence to CMU39 IPA using canonical context."""
    canonical_tokens = segment_ipa_for_cmu39(canonical_ipa)
    canonical_projected = _project_tokens(canonical_tokens, None)
    projected = _project_tokens(
        segment_ipa_for_cmu39(ipa_sequence),
        canonical_projected,
    )
    return separator.join(projected)


def project_ipa_triplet_to_cmu39(
    canonical_ipa: str,
    uttered_ipa: str,
    predicted_ipa: str,
    *,
    separator: str = " ",
) -> Tuple[str, str, str]:
    """Project canonical, uttered, and predicted IPA with one context."""
    canonical_tokens = segment_ipa_for_cmu39(canonical_ipa)
    canonical_projected = _project_tokens(canonical_tokens, None)
    uttered_projected = _project_tokens(
        segment_ipa_for_cmu39(uttered_ipa),
        canonical_projected,
    )
    predicted_projected = _project_tokens(
        segment_ipa_for_cmu39(predicted_ipa),
        canonical_projected,
    )
    return (
        separator.join(canonical_projected),
        separator.join(uttered_projected),
        separator.join(predicted_projected),
    )
