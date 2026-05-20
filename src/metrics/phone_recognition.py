"""Evaluate phone recognition output using panphon feature-based metrics.

Evaluation projects canonical, uttered, and predicted IPA onto a CMU39-style
IPA inventory before scoring. Every run must provide canonical IPA via
`--canonical_file` or `--canonical_field`.

Usage:
    python -m src.metrics.phone_recognition \
        --prediction_file exp/runs/inf_doreco_xlsr53/20251220_085642/transcription.withlang.json \
        --gt_field target \
        --key_field utt_id \
        --language_field lang_sym
    
    python -m src.metrics.phone_recognition --evaluation_name powsmctc \
        --prediction_file exp/runs/inf_doreco_powsm_ctc/8jobARR/transcription.json \
        --output_file exp/runs/inf_doreco_powsm_ctc/8jobARR/inventory_results.csv \
        --gt_field target \
        --key_field utt_id \
        --language_field lang_sym
    
    python -m src.metrics.phone_recognition --evaluation_name qweni \
        --prediction_file exp/runs/inf_doreco_qweni/1jobArr/transcription.json \
        --output_file exp/runs/inf_doreco_qweni/1jobArr/inventory_results.csv \
        --gt_field target \
        --key_field utt_id \
        --language_field lang_sym
    
    python -m src.metrics.phone_recognition --evaluation_name gemini \
        --prediction_file exp/runs/inf_tusom2021_gemini/20251224_142557/transcription.withlang.json \
        --output_file exp/runs/inf_tusom2021_gemini/20251224_142557/inventory_results.csv \
        --gt_field target \
        --key_field utt_id \
        --language_field lang_sym
    
    # PR results are in the output_file, inventory on terminal
    python -m src.metrics.phone_recognition --evaluation_name qweni \
        --prediction_file  exp/runs/inf_tusom2021_qweni/1jobArr/transcription.json \
        --output_file exp/runs/inf_tusom2021_qweni/1jobArr/inventory_results.csv \
        --gt_field target \
        --key_field utt_id
        
    python -m src.metrics.phone_recognition \
        --prediction_file exp/runs/inf_doreco_lv60/20251220_085643/transcription.withlang.json \
        --gt_field target \
        --key_field utt_id \
        --language_field lang_sym \
        --noisy_pr # for noisy phone recognition
"""

import argparse
import string
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from tqdm import tqdm
import unicodedata
from collections import Counter
from itertools import chain, combinations

import panphon
import panphon.distance
from kaldialign import align as _ka_align, edit_distance as _ka_edit_distance
from phone_inventory_metric import get_metrics as get_inventory_metrics
from phone_inventory_metric.common import setkeydict
from rich.console import Console
from rich.table import Table

from src.core.cmu39_projection import (
    CMU39_PROJECTION_PROFILE,
    project_ipa_triplet_to_cmu39,
)
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


MDD_BLANK = "-"
IPA_STRESS_MARKS = "ˈˌ"


def _levenshtein_align(
    seq_a: List[str], seq_b: List[str], blank: str = MDD_BLANK
) -> Tuple[List[str], List[str]]:
    """Levenshtein-align two phone sequences via kaldialign.

    Returns a pair of equal-length lists where indels are filled with `blank`.
    Edit distance is identical to a hand-rolled DP, but when multiple optimal
    alignments exist kaldialign's tiebreak may differ from a purely
    match-first/sub-next walk.
    """
    if not seq_a and not seq_b:
        return [], []
    pairs = _ka_align(seq_a, seq_b, blank)
    a_aligned = [p[0] for p in pairs]
    b_aligned = [p[1] for p in pairs]
    return a_aligned, b_aligned


def _align_to_prompted_slots(
    prompted: List[str], other: List[str], blank: str = MDD_BLANK
) -> Tuple[List[str], List[List[str]]]:
    """Align `other` to canonical `prompted` slots.

    Returns one symbol per prompted phone, plus insertion gaps before phone 0,
    between prompted phones, and after the final prompted phone. Anchoring both
    uttered and predicted sequences to these slots keeps unequal-length
    insertions/deletions comparable for MDD bucket counts.
    """
    if not prompted:
        return [], [list(other)]

    phone_slots = [blank] * len(prompted)
    gap_slots: List[List[str]] = [[] for _ in range(len(prompted) + 1)]
    p_aligned, o_aligned = _levenshtein_align(prompted, other, blank=blank)

    prompt_idx = 0
    for p_sym, o_sym in zip(p_aligned, o_aligned):
        if p_sym == blank:
            if o_sym != blank:
                gap_slots[prompt_idx].append(o_sym)
            continue

        phone_slots[prompt_idx] = o_sym
        prompt_idx += 1

    return phone_slots, gap_slots


def _mdd_correctness(prompted_sym: str, other_sym: str, blank: str) -> str:
    if prompted_sym == blank:
        return "C" if other_sym == blank else "E"
    if other_sym == blank:
        return "E"
    return "C" if prompted_sym == other_sym else "E"


def _update_mdd_counts(
    counts: Dict[str, int],
    prompted_sym: str,
    u_sym: str,
    h_sym: str,
    corr_U: List[str],
    corr_P: List[str],
    blank: str,
) -> None:
    cu = _mdd_correctness(prompted_sym, u_sym, blank=blank)
    cp = _mdd_correctness(prompted_sym, h_sym, blank=blank)
    corr_U.append(cu)
    corr_P.append(cp)

    if cu == "E" and cp == "E":
        if u_sym == h_sym:
            counts["CD"] += 1
        else:
            counts["DE"] += 1
    elif cu == "C" and cp == "C":
        counts["TA"] += 1
    elif cu == "C" and cp == "E":
        counts["FR"] += 1
    elif cu == "E" and cp == "C":
        counts["FA"] += 1


def _mdd_bucket(prompted_sym: str, u_sym: str, h_sym: str,
                blank: str) -> str:
    cu = _mdd_correctness(prompted_sym, u_sym, blank=blank)
    cp = _mdd_correctness(prompted_sym, h_sym, blank=blank)
    if cu == "E" and cp == "E":
        return "CD" if u_sym == h_sym else "DE"
    if cu == "C" and cp == "C":
        return "TA"
    if cu == "C" and cp == "E":
        return "FR"
    return "FA"


def _canonical_edit_cost(c_sym: str, other_sym: str,
                         blank: str = MDD_BLANK) -> int:
    if c_sym == blank and other_sym == blank:
        return 0
    if c_sym == blank or other_sym == blank:
        return 1
    return 0 if c_sym == other_sym else 1


def _align_prompted_to_uttered_predicted_rows(
    prompted: List[str],
    uttered_aligned: List[str],
    predicted_aligned: List[str],
    blank: str = MDD_BLANK,
) -> List[Tuple[str, str, str]]:
    """Align canonical phones to an existing uttered/predicted alignment.

    This implements the pairwise-plus-correction MDD protocol: U and H are
    aligned first, then C is placed on that shared row grid so the two
    correctness vectors are comparable. The child/canonical edit cost remains
    the primary objective, while later tie-breaks prevent exact C/H matches from
    being split into a predicted insertion plus a separate canonical deletion.
    """
    if len(uttered_aligned) != len(predicted_aligned):
        raise ValueError("uttered and predicted alignments must have equal length")

    n_p, n_rows = len(prompted), len(uttered_aligned)
    if n_p == 0 and n_rows == 0:
        return []

    # consume_prompted, consume_existing_uttered_predicted_row.
    transitions = [
        (True, True),
        (False, True),
        (True, False),
    ]

    Score = Tuple[int, int, int, int, int]
    State = Tuple[int, int]
    Row = Tuple[str, str, str]
    dp: Dict[State, Tuple[Score, Optional[State], Optional[Row]]] = {
        (0, 0): ((0, 0, 0, 0, 0), None, None)
    }

    for i in range(n_p + 1):
        for j in range(n_rows + 1):
            state = (i, j)
            if state not in dp:
                continue
            base_score = dp[state][0]
            for use_p, use_row in transitions:
                if use_p and i >= n_p:
                    continue
                if use_row and j >= n_rows:
                    continue

                next_state = (i + int(use_p), j + int(use_row))
                p_sym = prompted[i] if use_p else blank
                u_sym = uttered_aligned[j] if use_row else blank
                h_sym = predicted_aligned[j] if use_row else blank
                cu_cost = _canonical_edit_cost(p_sym, u_sym, blank=blank)
                ch_cost = _canonical_edit_cost(p_sym, h_sym, blank=blank)
                ch_exact_match = (
                    1 if p_sym == h_sym and p_sym != blank else 0)
                good_mdd = 1 if _mdd_bucket(
                    p_sym, u_sym, h_sym, blank=blank) in {"TA", "CD"} else 0
                row_count = 1
                candidate_score: Score = (
                    base_score[0] + cu_cost,
                    base_score[1] - ch_exact_match,
                    base_score[2] - good_mdd,
                    base_score[3] + ch_cost,
                    base_score[4] + row_count,
                )
                current = dp.get(next_state)
                if current is None or candidate_score < current[0]:
                    dp[next_state] = (
                        candidate_score, state, (p_sym, u_sym, h_sym))

    final_state = (n_p, n_rows)
    aligned_rows: List[Row] = []
    state: Optional[State] = final_state
    while state is not None and state != (0, 0):
        score, prev_state, row = dp[state]
        del score
        if row is not None:
            aligned_rows.append(row)
        state = prev_state
    aligned_rows.reverse()
    return aligned_rows


def _mdd_counts(
    prompted: List[str],
    uttered: List[str],
    predicted: List[str],
    blank: str = MDD_BLANK,
) -> Tuple[Dict[str, int], List[str], List[str]]:
    """Compute published pairwise-corrected MDD TR/TA/FR/FA plus CD/DE.

    Following Gelin et al. (SLaTE 2023), uttered and predicted phones are first
    Levenshtein-aligned. Prompted phones are then aligned to that shared U/H
    grid so prompted/uttered and prompted/predicted correctness vectors have
    comparable row coordinates. TR is split into CD/DE by checking whether the
    predicted error phone matches the uttered error phone.

    Returns (counts, corr_U, corr_P) where corr_U/corr_P are space-friendly
    "C"/"E" lists for per-utt error analysis.
    """
    counts = {"TR": 0, "TA": 0, "FR": 0, "FA": 0, "CD": 0, "DE": 0}
    if not prompted and not uttered and not predicted:
        return counts, [], []

    u_aligned, h_aligned = _levenshtein_align(uttered, predicted, blank=blank)
    aligned_rows = _align_prompted_to_uttered_predicted_rows(
        prompted, u_aligned, h_aligned, blank=blank)
    corr_U: List[str] = []
    corr_P: List[str] = []

    for p_sym, u_sym, h_sym in aligned_rows:
        _update_mdd_counts(
            counts, p_sym, u_sym, h_sym, corr_U, corr_P, blank=blank)

    counts["TR"] = counts["CD"] + counts["DE"]
    return counts, corr_U, corr_P


def _joint_mdd_counts(
    prompted: List[str],
    uttered: List[str],
    predicted: List[str],
    blank: str = MDD_BLANK,
) -> Tuple[Dict[str, int], List[str], List[str], List[str], List[str],
           List[str]]:
    """Jointly align canonical, uttered, and predicted phones for MDD.

    The child/canonical edit distance is the primary objective. The hypothesis
    only breaks ties between equally good child/canonical alignments, preferring
    alignments where uttered and predicted phones agree instead of shifting a
    predicted-only insertion onto the next canonical slot.
    """
    counts = {"TR": 0, "TA": 0, "FR": 0, "FA": 0, "CD": 0, "DE": 0}
    if not prompted and not uttered and not predicted:
        return counts, [], [], [], [], []

    n_p, n_u, n_h = len(prompted), len(uttered), len(predicted)
    # consume_prompted, consume_uttered, consume_predicted. The order is the
    # final deterministic tiebreak when all objective terms are equal.
    transitions = [
        (True, True, True),
        (False, True, True),
        (False, False, True),
        (True, False, False),
        (True, True, False),
        (True, False, True),
        (False, True, False),
    ]

    # Minimize child/canonical edit cost first. Among equally good child truth
    # alignments, prefer non-gap U/H agreement, preserve exact canonical/H
    # matches, prefer good MDD rows, then minimize H/canonical edit cost and
    # keep the alignment compact.
    Score = Tuple[int, int, int, int, int, int, int]
    State = Tuple[int, int, int]
    Row = Tuple[str, str, str]
    dp: Dict[State, Tuple[Score, Optional[State], Optional[Row]]] = {
        (0, 0, 0): ((0, 0, 0, 0, 0, 0, 0), None, None)
    }

    for i in range(n_p + 1):
        for j in range(n_u + 1):
            for k in range(n_h + 1):
                state = (i, j, k)
                if state not in dp:
                    continue
                base_score = dp[state][0]
                for use_p, use_u, use_h in transitions:
                    if use_p and i >= n_p:
                        continue
                    if use_u and j >= n_u:
                        continue
                    if use_h and k >= n_h:
                        continue

                    next_state = (
                        i + int(use_p),
                        j + int(use_u),
                        k + int(use_h),
                    )
                    p_sym = prompted[i] if use_p else blank
                    u_sym = uttered[j] if use_u else blank
                    h_sym = predicted[k] if use_h else blank
                    cu_cost = _canonical_edit_cost(p_sym, u_sym, blank=blank)
                    ch_cost = _canonical_edit_cost(p_sym, h_sym, blank=blank)
                    uh_agreement = (
                        1 if u_sym == h_sym and u_sym != blank else 0)
                    ch_exact_match = (
                        1 if p_sym == h_sym and p_sym != blank else 0)
                    good_mdd = 1 if _mdd_bucket(
                        p_sym, u_sym, h_sym, blank=blank) in {"TA", "CD"} else 0
                    row_count = 1
                    gap_row = 0 if use_p else 1
                    candidate_score: Score = (
                        base_score[0] + cu_cost,
                        base_score[1] - uh_agreement,
                        base_score[2] - ch_exact_match,
                        base_score[3] - good_mdd,
                        base_score[4] + ch_cost,
                        base_score[5] + row_count,
                        base_score[6] + gap_row,
                    )
                    current = dp.get(next_state)
                    if current is None or candidate_score < current[0]:
                        dp[next_state] = (
                            candidate_score, state, (p_sym, u_sym, h_sym))

    final_state = (n_p, n_u, n_h)
    aligned_rows: List[Row] = []
    state: Optional[State] = final_state
    while state is not None and state != (0, 0, 0):
        score, prev_state, row = dp[state]
        del score
        if row is not None:
            aligned_rows.append(row)
        state = prev_state
    aligned_rows.reverse()

    corr_U: List[str] = []
    corr_P: List[str] = []
    prompted_aligned: List[str] = []
    uttered_aligned: List[str] = []
    predicted_aligned: List[str] = []
    for p_sym, u_sym, h_sym in aligned_rows:
        prompted_aligned.append(p_sym)
        uttered_aligned.append(u_sym)
        predicted_aligned.append(h_sym)
        _update_mdd_counts(
            counts, p_sym, u_sym, h_sym, corr_U, corr_P, blank=blank)

    counts["TR"] = counts["CD"] + counts["DE"]
    return (counts, corr_U, corr_P, prompted_aligned, uttered_aligned,
            predicted_aligned)


@dataclass
class PhoneRecognitionSummary:
    """Aggregate metrics for a phone recognition experiment."""

    PFER: float
    FER: float
    FED: float
    PER: float
    SUB: float
    INS: float
    DEL: float
    N: int  # number of utterances
    phones: int  # total number of reference phones
    inventory: setkeydict[float]
    # MDD aggregates — populated only when canonical IPA is provided.
    has_mdd: bool = False
    has_joint_mdd: bool = False
    TR: int = 0
    TA: int = 0
    FR: int = 0
    FA: int = 0
    CD: int = 0
    DE: int = 0
    Detection_Accuracy: Optional[float] = None
    FRR: Optional[float] = None
    FAR: Optional[float] = None
    MDD_Precision: Optional[float] = None
    MDD_Recall: Optional[float] = None
    MDD_F1: Optional[float] = None
    Diagnostic_Accuracy: Optional[float] = None
    Diagnostic_Error_Rate: Optional[float] = None
    Overall_Diagnostic_Error_Rate: Optional[float] = None
    True_Diagnostic_Accuracy: Optional[float] = None
    Joint_TR: int = 0
    Joint_TA: int = 0
    Joint_FR: int = 0
    Joint_FA: int = 0
    Joint_CD: int = 0
    Joint_DE: int = 0
    Joint_Detection_Accuracy: Optional[float] = None
    Joint_FRR: Optional[float] = None
    Joint_FAR: Optional[float] = None
    Joint_MDD_Precision: Optional[float] = None
    Joint_MDD_Recall: Optional[float] = None
    Joint_MDD_F1: Optional[float] = None
    Joint_Diagnostic_Accuracy: Optional[float] = None
    Joint_Diagnostic_Error_Rate: Optional[float] = None
    Joint_Overall_Diagnostic_Error_Rate: Optional[float] = None
    Joint_True_Diagnostic_Accuracy: Optional[float] = None


class PhoneRecognitionEvaluator:
    """
    Evaluates CMU39-projected phone recognition output.

      * PER (phone error rate, %)
      * FER (feature error rate, %)
      * FED (total feature edit distance)
      * PFER (phone feature error rate averaged per utterance)
      * per-utterance metrics

    Assumes `test_data` is a dict:
        { utt_id: {"prediction": str, "transcription": str, "canonical": str}, ... }
    """

    def __init__(
        self,
        normalize_ipa: bool = True,
        compute_joint_mdd: bool = False,
    ):
        self.normalize_ipa = normalize_ipa
        self.compute_joint_mdd = compute_joint_mdd
        self.dst = panphon.distance.Distance()

    @staticmethod
    def clean_text(s: str) -> str:
        """Normalize IPA text: remove spaces/punct, NFC->NFD, fix 'g'→'ɡ',
        remap precomposed rhotic schwa ɚ/ɝ to decomposed ə˞/ɜ˞ so panphon
        recognizes them (panphon.ipa_segs('ɚ') returns []).
        """
        s = s.replace(" ", "").translate(str.maketrans("", "", string.punctuation))
        s = s.translate(str.maketrans("", "", IPA_STRESS_MARKS))
        s = unicodedata.normalize("NFD", s)
        s = s.replace("g", "ɡ")
        s = s.replace("ɚ", "ə˞").replace("ɝ", "ɜ˞")
        return s.strip()

    @staticmethod
    def clean_mdd_token(s: str) -> str:
        s = s.translate(str.maketrans("", "", string.punctuation))
        s = s.translate(str.maketrans("", "", IPA_STRESS_MARKS))
        s = unicodedata.normalize("NFD", s)
        s = s.replace("g", "ɡ")
        s = s.replace("ɚ", "ə˞").replace("ɝ", "ɜ˞")
        return s.strip()

    def _prepare(self, text: str) -> str:
        return self.clean_text(text) if self.normalize_ipa else text

    def _mdd_segments(self, text: str) -> List[str]:
        if " " in text.strip():
            if not self.normalize_ipa:
                return [token for token in text.split() if token]
            return [
                token
                for token in (self.clean_mdd_token(token) for token in text.split())
                if token
            ]
        return self.dst.fm.ipa_segs(self._prepare(text))

    def _compute_sid_metrics(self, hyp: List[str], ref: List[str]) -> Tuple[int, int, int]:
        """Substitution / insertion / deletion counts on phones via kaldialign.

        `ins` = phones in hyp not in ref; `del` = phones in ref not in hyp.
        """
        if not hyp and not ref:
            return 0, 0, 0
        res = _ka_edit_distance(ref, hyp)
        return res["sub"], res["ins"], res["del"]

    def _compute_utterance_metrics(
        self, hyp: str, ref: str
    ) -> Tuple[Dict[str, Union[int, float]]]:
        """
        Compute metrics for a single utterance.

        Returns:
            (metrics_dict, pfer, fed, per_errors, n_phones)
        """
        hyp_segs = self._mdd_segments(hyp)
        ref_segs = self._mdd_segments(ref)
        hyp_text = "".join(hyp_segs)
        ref_text = "".join(ref_segs)

        # Phone feature distances
        pfer = self.dst.hamming_feature_edit_distance(hyp_text, ref_text)
        fed = self.dst.feature_edit_distance(hyp_text, ref_text)

        # PER via min_edit_distance over projected CMU39 tokens.
        n_phones = len(ref_segs)

        per_errors = self.dst.min_edit_distance(
            lambda v: 1,  # deletion cost
            lambda v: 1,  # insertion cost
            lambda x, y: 0 if x == y else 1,  # substitution cost
            [[]],  # start
            hyp_segs,
            ref_segs,
        )
        sub_errors, ins_errors, del_errors = self._compute_sid_metrics(
            hyp_segs, ref_segs
        )

        metrics = {
            "pfer": float(pfer),
            "fed": float(fed),
            "per": float(per_errors / n_phones * 100) if n_phones > 0 else 0.0,
            "fer": float(fed / n_phones * 100) if n_phones > 0 else 0.0,
        }
        out = {
            "metrics": metrics,
            "pfer": pfer,
            "fed": fed,
            "per_errors": per_errors,
            "sub_errors": sub_errors,
            "ins_errors": ins_errors,
            "del_errors": del_errors,
            "n_phones": n_phones,
        }
        return out

    def _compute_mdd(
        self, prompted: str, uttered: str, predicted: str
    ) -> Dict[str, Any]:
        """Compute per-utt MDD counts. Returns dict with TR/TA/FR/FA plus the
        prompted/uttered/predicted segment strings and corr_U/corr_P vectors
        for downstream per-utt CSV writing.
        """
        p_segs = self._mdd_segments(prompted)
        u_segs = self._mdd_segments(uttered)
        h_segs = self._mdd_segments(predicted)
        counts, corr_U, corr_P = _mdd_counts(p_segs, u_segs, h_segs)
        out = {
            "TR": counts["TR"],
            "TA": counts["TA"],
            "FR": counts["FR"],
            "FA": counts["FA"],
            "CD": counts["CD"],
            "DE": counts["DE"],
            "prompted": " ".join(p_segs),
            "uttered": " ".join(u_segs),
            "predicted": " ".join(h_segs),
            "corr_U": " ".join(corr_U),
            "corr_P": " ".join(corr_P),
        }
        if self.compute_joint_mdd:
            (joint_counts, joint_corr_U, joint_corr_P, joint_prompted,
             joint_uttered, joint_predicted) = _joint_mdd_counts(
                 p_segs, u_segs, h_segs)
            out.update({
                "Joint_TR": joint_counts["TR"],
                "Joint_TA": joint_counts["TA"],
                "Joint_FR": joint_counts["FR"],
                "Joint_FA": joint_counts["FA"],
                "Joint_CD": joint_counts["CD"],
                "Joint_DE": joint_counts["DE"],
                "joint_prompted": " ".join(joint_prompted),
                "joint_uttered": " ".join(joint_uttered),
                "joint_predicted": " ".join(joint_predicted),
                "joint_corr_U": " ".join(joint_corr_U),
                "joint_corr_P": " ".join(joint_corr_P),
            })
        return out

    @classmethod
    def _get_phone_inventory_metrics(
        cls, test_data: dict[str, dict[str, Any]]
    ) -> setkeydict[float]:
        """
        Compute the phone inventory metrics on the dataset.

        The results are computed against a combination of different boolean options:
        - `featured`: if True, use a fuzzy notion of set membership based on
          phonetic feature similarity (provided by Panphon).
        - `exclusive`: if True, then phones in each inventory may match at most
          one other phone; if False, any phone matches its nearest neighbor in
          the other set (this only makes a difference for when `featured` is
          true.
        - `max`: if True, compute the optimal cutoff for the reference set in
          terms of F1-score (i.e., iteratively remove the least frequent phones
          from the predicted inventory).

        Returns
        -------
        setkeydict:
            A dict where the keys are tuples of strings that do not care about
            order.  Indexing with brackets works, but some methods (e.g.,
            `.get()`) may not work properly.


        """

        def get_inventory(key: str) -> list[str]:
            c = Counter()
            for _, sample in test_data.items():
                datum = sample.get(key, "")
                c.update(token for token in datum.split() if token)
            # This will return phones in order of descending frequency.  For
            # the reference set, this is not taken into account, but for the
            # predictions, it used to calculate an upper bound onf the
            # F1-score.
            return [x[0] for x in c.most_common()]

        pred_inventory = get_inventory("prediction")
        ref_inventory = get_inventory("transcription")
        return get_inventory_metrics(ref_inventory, pred_inventory, search_max=True)

    def evaluate(
        self, test_data: Dict[str, Dict[str, Any]]
    ) -> Tuple[PhoneRecognitionSummary, Dict[str, Dict[str, float]]]:
        """
        Evaluate a full dataset.

        Args:
            test_data: mapping from utt_id -> {"prediction": ..., "transcription": ..., "canonical": ...}

        Returns:
            summary: PhoneRecognitionSummary (aggregate metrics)
            instance_metrics: per-utterance metrics, same keys as original script:
                             {utt_id: {"pfer":..., "fed":..., "per":..., "fer":...}}
        """
        if not test_data:
            empty_summary = PhoneRecognitionSummary(
                PFER=0.0,
                FER=0.0,
                FED=0.0,
                PER=0.0,
                N=0,
                phones=0,
                SUB=0.0,
                INS=0.0,
                DEL=0.0,
                inventory=setkeydict([]),
            )
            return empty_summary, {}

        instance_metrics: Dict[str, Dict[str, Any]] = {}

        pfer_sum = 0.0
        fed_sum = 0.0
        per_err_sum = 0.0
        phones_sum = 0
        n_utts = 0
        sub_err_sum = 0
        ins_err_sum = 0
        del_err_sum = 0

        has_mdd = False
        tr_sum = ta_sum = fr_sum = fa_sum = 0
        cd_sum = de_sum = 0
        joint_tr_sum = joint_ta_sum = joint_fr_sum = joint_fa_sum = 0
        joint_cd_sum = joint_de_sum = 0
        projected_data: Dict[str, Dict[str, Any]] = {}

        for utt_id, sample in tqdm(
            test_data.items(), total=len(test_data), desc="Evaluating", leave=False
        ):
            hyp_raw = sample.get("prediction", "")
            ref_raw = sample.get("transcription", "")
            canonical_raw = sample.get("canonical")
            if canonical_raw is None or not str(canonical_raw).strip():
                raise ValueError(
                    "CMU39-projected evaluation requires canonical IPA for "
                    f"every utterance; missing canonical for {utt_id}."
                )

            canonical, ref, hyp = project_ipa_triplet_to_cmu39(
                str(canonical_raw),
                str(ref_raw),
                str(hyp_raw),
            )
            projected_data[utt_id] = {
                "prediction": hyp,
                "transcription": ref,
            }

            out = self._compute_utterance_metrics(hyp, ref)

            instance_metrics[utt_id] = dict(out["metrics"])
            pfer_sum += out["pfer"]
            fed_sum += out["fed"]
            per_err_sum += out["per_errors"]
            phones_sum += out["n_phones"]
            sub_err_sum += out["sub_errors"]
            ins_err_sum += out["ins_errors"]
            del_err_sum += out["del_errors"]
            n_utts += 1

            has_mdd = True
            mdd = self._compute_mdd(canonical, ref, hyp)
            tr_sum += mdd["TR"]
            ta_sum += mdd["TA"]
            fr_sum += mdd["FR"]
            fa_sum += mdd["FA"]
            cd_sum += mdd["CD"]
            de_sum += mdd["DE"]
            if self.compute_joint_mdd:
                joint_tr_sum += mdd["Joint_TR"]
                joint_ta_sum += mdd["Joint_TA"]
                joint_fr_sum += mdd["Joint_FR"]
                joint_fa_sum += mdd["Joint_FA"]
                joint_cd_sum += mdd["Joint_CD"]
                joint_de_sum += mdd["Joint_DE"]
            instance_metrics[utt_id]["mdd"] = mdd

        summary = PhoneRecognitionSummary(
            PFER=pfer_sum / n_utts if n_utts > 0 else 0.0,
            FER=(fed_sum / phones_sum * 100) if phones_sum > 0 else 0.0,
            FED=fed_sum,
            PER=(per_err_sum / phones_sum * 100) if phones_sum > 0 else 0.0,
            SUB=(sub_err_sum / phones_sum * 100) if phones_sum > 0 else 0.0,
            INS=(ins_err_sum / phones_sum * 100) if phones_sum > 0 else 0.0,
            DEL=(del_err_sum / phones_sum * 100) if phones_sum > 0 else 0.0,
            N=n_utts,
            phones=phones_sum,
            inventory=self._get_phone_inventory_metrics(projected_data),
        )

        if has_mdd:
            summary.has_mdd = True
            summary.TR = tr_sum
            summary.TA = ta_sum
            summary.FR = fr_sum
            summary.FA = fa_sum
            summary.CD = cd_sum
            summary.DE = de_sum
            total = tr_sum + ta_sum + fr_sum + fa_sum
            if total > 0:
                summary.Detection_Accuracy = (tr_sum + ta_sum) / total
                summary.Overall_Diagnostic_Error_Rate = (
                    fr_sum + fa_sum + de_sum
                ) / total
            if (fr_sum + ta_sum) > 0:
                summary.FRR = fr_sum / (fr_sum + ta_sum)
            if (fa_sum + tr_sum) > 0:
                summary.FAR = fa_sum / (fa_sum + tr_sum)
            if (tr_sum + fr_sum) > 0:
                summary.MDD_Precision = cd_sum / (tr_sum + fr_sum)
            if (tr_sum + fa_sum) > 0:
                summary.MDD_Recall = cd_sum / (tr_sum + fa_sum)
            if (
                summary.MDD_Precision is not None
                and summary.MDD_Recall is not None
                and (summary.MDD_Precision + summary.MDD_Recall) > 0
            ):
                summary.MDD_F1 = (
                    2
                    * summary.MDD_Precision
                    * summary.MDD_Recall
                    / (summary.MDD_Precision + summary.MDD_Recall)
                )
            if tr_sum > 0:
                summary.Diagnostic_Accuracy = cd_sum / tr_sum
                summary.Diagnostic_Error_Rate = de_sum / tr_sum
            if (tr_sum + fa_sum) > 0:
                summary.True_Diagnostic_Accuracy = cd_sum / (tr_sum + fa_sum)
            if self.compute_joint_mdd:
                summary.has_joint_mdd = True
                summary.Joint_TR = joint_tr_sum
                summary.Joint_TA = joint_ta_sum
                summary.Joint_FR = joint_fr_sum
                summary.Joint_FA = joint_fa_sum
                summary.Joint_CD = joint_cd_sum
                summary.Joint_DE = joint_de_sum
                joint_total = (
                    joint_tr_sum + joint_ta_sum + joint_fr_sum + joint_fa_sum)
                if joint_total > 0:
                    summary.Joint_Detection_Accuracy = (
                        joint_tr_sum + joint_ta_sum) / joint_total
                    summary.Joint_Overall_Diagnostic_Error_Rate = (
                        joint_fr_sum + joint_fa_sum + joint_de_sum
                    ) / joint_total
                if (joint_fr_sum + joint_ta_sum) > 0:
                    summary.Joint_FRR = (
                        joint_fr_sum / (joint_fr_sum + joint_ta_sum))
                if (joint_fa_sum + joint_tr_sum) > 0:
                    summary.Joint_FAR = (
                        joint_fa_sum / (joint_fa_sum + joint_tr_sum))
                if (joint_tr_sum + joint_fr_sum) > 0:
                    summary.Joint_MDD_Precision = (
                        joint_cd_sum / (joint_tr_sum + joint_fr_sum))
                if (joint_tr_sum + joint_fa_sum) > 0:
                    summary.Joint_MDD_Recall = (
                        joint_cd_sum / (joint_tr_sum + joint_fa_sum))
                if (
                    summary.Joint_MDD_Precision is not None
                    and summary.Joint_MDD_Recall is not None
                    and (summary.Joint_MDD_Precision
                         + summary.Joint_MDD_Recall) > 0
                ):
                    summary.Joint_MDD_F1 = (
                        2
                        * summary.Joint_MDD_Precision
                        * summary.Joint_MDD_Recall
                        / (summary.Joint_MDD_Precision
                           + summary.Joint_MDD_Recall)
                    )
                if joint_tr_sum > 0:
                    summary.Joint_Diagnostic_Accuracy = (
                        joint_cd_sum / joint_tr_sum)
                    summary.Joint_Diagnostic_Error_Rate = (
                        joint_de_sum / joint_tr_sum)
                if (joint_tr_sum + joint_fa_sum) > 0:
                    summary.Joint_True_Diagnostic_Accuracy = (
                        joint_cd_sum / (joint_tr_sum + joint_fa_sum))

        return summary, instance_metrics

    @staticmethod
    def pretty_print(summary: PhoneRecognitionSummary, **_kwargs: Any) -> None:
        """Rich summary table."""
        t = Table(title="Phone Recognition Results")
        t.add_column("Metric")
        t.add_column("Value", justify="right")
        t.add_row("Utterances (N)", f"{summary.N}")
        t.add_row("Total Phones", f"{summary.phones}")
        t.add_row("PFER (avg per utt)", f"{summary.PFER:.4f}")
        t.add_row("FER (%)", f"{summary.FER:.2f}")
        t.add_row("FED (total)", f"{summary.FED:.2f}")
        t.add_row("PER (%)", f"{summary.PER:.2f}")
        t.add_row("SUB (%)", f"{summary.SUB:.2f}")
        t.add_row("INS (%)", f"{summary.INS:.2f}")
        t.add_row("DEL (%)", f"{summary.DEL:.2f}")
        if summary.has_mdd:
            t.add_row("TR / TA / FR / FA", f"{summary.TR} / {summary.TA} / {summary.FR} / {summary.FA}")
            t.add_row("CD / DE", f"{summary.CD} / {summary.DE}")

            def _f(v):
                return f"{v:.4f}" if v is not None else "—"

            t.add_row("Detection_Accuracy", _f(summary.Detection_Accuracy))
            t.add_row("FRR", _f(summary.FRR))
            t.add_row("FAR", _f(summary.FAR))
            t.add_row("MDD_Precision", _f(summary.MDD_Precision))
            t.add_row("MDD_Recall", _f(summary.MDD_Recall))
            t.add_row("MDD_F1", _f(summary.MDD_F1))
            t.add_row("Diagnostic_Accuracy", _f(summary.Diagnostic_Accuracy))
            t.add_row("Diagnostic_Error_Rate", _f(summary.Diagnostic_Error_Rate))
            t.add_row("Overall_Diagnostic_Error_Rate", _f(summary.Overall_Diagnostic_Error_Rate))
            t.add_row("True_Diagnostic_Accuracy", _f(summary.True_Diagnostic_Accuracy))
            if summary.has_joint_mdd:
                t.add_row("Joint TR / TA / FR / FA", f"{summary.Joint_TR} / {summary.Joint_TA} / {summary.Joint_FR} / {summary.Joint_FA}")
                t.add_row("Joint CD / DE", f"{summary.Joint_CD} / {summary.Joint_DE}")
                t.add_row("Joint_Detection_Accuracy", _f(summary.Joint_Detection_Accuracy))
                t.add_row("Joint_FRR", _f(summary.Joint_FRR))
                t.add_row("Joint_FAR", _f(summary.Joint_FAR))
                t.add_row("Joint_MDD_Precision", _f(summary.Joint_MDD_Precision))
                t.add_row("Joint_MDD_Recall", _f(summary.Joint_MDD_Recall))
                t.add_row("Joint_MDD_F1", _f(summary.Joint_MDD_F1))
                t.add_row("Joint_Diagnostic_Accuracy", _f(summary.Joint_Diagnostic_Accuracy))
                t.add_row("Joint_Diagnostic_Error_Rate", _f(summary.Joint_Diagnostic_Error_Rate))
                t.add_row("Joint_Overall_Diagnostic_Error_Rate", _f(summary.Joint_Overall_Diagnostic_Error_Rate))
                t.add_row("Joint_True_Diagnostic_Accuracy", _f(summary.Joint_True_Diagnostic_Accuracy))
        Console().print(t)

        PhoneRecognitionEvaluator.pretty_print_inventory_metrics(summary.inventory)

    @staticmethod
    def pretty_print_inventory_metrics(inventory_metrics: setkeydict[float]) -> None:
        t = Table(title="Phone Inventory Metrics")
        t.add_column("Exclusive\nMatch", justify="center")
        t.add_column("Featured", justify="center")
        t.add_column("Upper\nBound", justify="center")
        t.add_column("F1", justify="center")
        t.add_column("Precision", justify="center")
        t.add_column("Recall", justify="center")

        # powerset
        base_key_elements = ["exclusive", "max", "featured"]
        base_keys = chain.from_iterable(
            combinations(base_key_elements, n)
            for n in range(len(base_key_elements) + 1)
        )

        for base_key in base_keys:
            if "featured" not in base_key and "exclusive" in base_key:
                continue
            f1 = inventory_metrics[base_key + ("f1_score",)]
            precision = inventory_metrics[base_key + ("precision",)]
            recall = inventory_metrics[base_key + ("recall",)]
            t.add_row(
                # If we are not using features, the matches are implicitly exclusive.
                "x" if ("exclusive" in base_key or "featured" not in base_key) else "",
                "x" if "featured" in base_key else "",
                "x" if "max" in base_key else "",
                f"{f1:.3f}",
                f"{precision:.3f}",
                f"{recall:.3f}",
            )

        Console().print(t)

    def write_to_csv(
        self,
        summary: PhoneRecognitionSummary,
        evalname: str,
        output_file: str,
        language: str,
    ) -> None:
        """Append summary metrics to a CSV file and a sibling readable .txt."""
        import csv
        import os

        base_headers = [
            "eval_name",
            "FER (%)",
            "PER (%)",
        ]
        mdd_headers = [
            "TR",
            "TA",
            "FR",
            "FA",
            "CD",
            "DE",
            "Detection_Accuracy",
            "FRR",
            "FAR",
            "MDD_Precision",
            "MDD_Recall",
            "MDD_F1",
            "Diagnostic_Accuracy",
            "Diagnostic_Error_Rate",
            "Overall_Diagnostic_Error_Rate",
            "True_Diagnostic_Accuracy",
        ]
        joint_headers = [
            "Joint_TR",
            "Joint_TA",
            "Joint_FR",
            "Joint_FA",
            "Joint_CD",
            "Joint_DE",
            "Joint_Detection_Accuracy",
            "Joint_FRR",
            "Joint_FAR",
            "Joint_MDD_Precision",
            "Joint_MDD_Recall",
            "Joint_MDD_F1",
            "Joint_Diagnostic_Accuracy",
            "Joint_Diagnostic_Error_Rate",
            "Joint_Overall_Diagnostic_Error_Rate",
            "Joint_True_Diagnostic_Accuracy",
        ]
        headers = [
            *base_headers,
            *mdd_headers,
            *(joint_headers if summary.has_joint_mdd else []),
            "N",
            "phones",
        ]

        def _fmt(v: Optional[float]) -> str:
            return f"{v:.4f}" if v is not None else ""

        if summary.has_mdd:
            mdd_cells = [
                summary.TR,
                summary.TA,
                summary.FR,
                summary.FA,
                summary.CD,
                summary.DE,
                _fmt(summary.Detection_Accuracy),
                _fmt(summary.FRR),
                _fmt(summary.FAR),
                _fmt(summary.MDD_Precision),
                _fmt(summary.MDD_Recall),
                _fmt(summary.MDD_F1),
                _fmt(summary.Diagnostic_Accuracy),
                _fmt(summary.Diagnostic_Error_Rate),
                _fmt(summary.Overall_Diagnostic_Error_Rate),
                _fmt(summary.True_Diagnostic_Accuracy),
            ]
            if summary.has_joint_mdd:
                mdd_cells.extend([
                    summary.Joint_TR,
                    summary.Joint_TA,
                    summary.Joint_FR,
                    summary.Joint_FA,
                    summary.Joint_CD,
                    summary.Joint_DE,
                    _fmt(summary.Joint_Detection_Accuracy),
                    _fmt(summary.Joint_FRR),
                    _fmt(summary.Joint_FAR),
                    _fmt(summary.Joint_MDD_Precision),
                    _fmt(summary.Joint_MDD_Recall),
                    _fmt(summary.Joint_MDD_F1),
                    _fmt(summary.Joint_Diagnostic_Accuracy),
                    _fmt(summary.Joint_Diagnostic_Error_Rate),
                    _fmt(summary.Joint_Overall_Diagnostic_Error_Rate),
                    _fmt(summary.Joint_True_Diagnostic_Accuracy),
                ])
        else:
            mdd_cells = [""] * (
                len(mdd_headers)
                + (len(joint_headers) if summary.has_joint_mdd else 0)
            )

        row = [
            evalname,
            f"{summary.FER:.2f}",
            f"{summary.PER:.2f}",
            *mdd_cells,
            summary.N,
            summary.phones,
        ]

        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        write_header = (
            not os.path.exists(output_file) or os.path.getsize(output_file) == 0
        )
        with open(output_file, mode="a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if write_header:
                writer.writerow(headers)
            writer.writerow(row)

        txt_path = os.path.splitext(output_file)[0] + ".txt"
        self._append_summary_txt(txt_path, headers, row)

    @staticmethod
    def _append_summary_txt(
        txt_path: str, headers: List[str], row: List[Any]
    ) -> None:
        """Append a human-readable block mirroring the CSV row to `txt_path`."""
        label_w = max(len(h) for h in headers)
        bar = "=" * (label_w + 16)
        lines = [bar, f"Evaluation: {row[0]}", "-" * len(bar)]
        for h, v in list(zip(headers, row))[1:]:
            lines.append(f"  {h:<{label_w}} : {v}")
        lines.append(bar)
        lines.append("")
        with open(txt_path, "a") as f:
            f.write("\n".join(lines) + "\n")


def _load_canonical(canonical_file: str) -> Dict[str, str]:
    """Parse Kaldi-style `text.canonical` (utt_id <space-separated-IPA>) into
    {utt_id: ipa_string}. Lines without IPA map to empty string.
    """
    out: Dict[str, str] = {}
    with open(canonical_file, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                out[parts[0]] = parts[1]
            else:
                out[parts[0]] = ""
    return out


def _prediction_value(item: Dict[str, Any], pred_field: str) -> str:
    pred = item.get("pred")
    if not isinstance(pred, list) or not pred or not isinstance(pred[0], dict):
        return ""
    value = pred[0].get(pred_field, "")
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _load_predictions(
    pred_file: str,
    language_field: str = None,
    canonical_data: Optional[Dict[str, str]] = None,
    canonical_field: Optional[str] = None,
) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Loads prediction file from JSON format.
    The returned structure is:
    {'language': { utt_id: {"prediction": str, "transcription": str, "canonical": str}, ... }}
    If language_field is None, 'language' is set to the string '"combined"'.

    `canonical_data` (utt_id -> ipa) takes precedence over `canonical_field`
    (a passthrough field name). The evaluator requires canonical IPA for CMU39
    projection, so CLI callers must provide one of those sources.
    """
    with open(pred_file, "r") as f:
        data = json.load(f)

    all_languages = set()
    if language_field is not None:
        assert (
            language_field in next(iter(data.values()))["passthrough"]
        ), f"Language field '{language_field}' not found in prediction file."
        all_languages = {item["passthrough"][language_field] for item in data.values()}
    else:
        all_languages = {"combined"}

    all_languages = sorted(all_languages)
    print(f"Found {len(all_languages)} languages: {all_languages}")
    has_canonical = canonical_data is not None or canonical_field is not None
    return_data = {}
    for lang in tqdm(all_languages, desc="Loading predictions"):
        D = {}
        for _, item in data.items():
            if item["passthrough"].get(language_field, "combined") != lang:
                continue
            utt_id = item["passthrough"][args.key_field]
            sample: Dict[str, str] = {
                "prediction": _prediction_value(item, args.pred_field),
                "transcription": (
                    item["passthrough"][args.gt_field]
                    if not args.noisy_pr
                    else "".join(
                        [
                            n
                            for n in item["passthrough"]["masked_phones"]
                            if n != "[NOISE]"
                        ]
                    )
                ),
            }
            if has_canonical:
                if canonical_data is not None:
                    sample["canonical"] = canonical_data.get(utt_id, "")
                else:
                    sample["canonical"] = item["passthrough"].get(
                        canonical_field, ""
                    )
            D[utt_id] = sample
        return_data[lang] = D
    return return_data


def write_mdd_per_utt_csv(
    output_path: str, rows: List[Dict[str, Any]]
) -> None:
    """Write per-utt MDD rows to a CSV. Columns: eval_name, language, utt_id,
    prompted, uttered, predicted, corr_U, corr_P, TR, TA, FR, FA.
    """
    import csv
    import os

    if not rows:
        return
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    headers = [
        "eval_name",
        "language",
        "utt_id",
        "projection_profile",
        "prompted",
        "uttered",
        "predicted",
        "corr_U",
        "corr_P",
        "TR",
        "TA",
        "FR",
        "FA",
        "CD",
        "DE",
    ]
    write_header = (
        not os.path.exists(output_path) or os.path.getsize(output_path) == 0
    )
    with open(output_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(headers)
        for r in rows:
            writer.writerow([r.get(h, "") for h in headers])


def write_mdd_joint_per_utt_csv(
    output_path: str, rows: List[Dict[str, Any]]
) -> None:
    """Write per-utt experimental joint MDD rows to a CSV."""
    import csv
    import os

    if not rows:
        return
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    headers = [
        "eval_name",
        "language",
        "utt_id",
        "projection_profile",
        "joint_prompted",
        "joint_uttered",
        "joint_predicted",
        "joint_corr_U",
        "joint_corr_P",
        "Joint_TR",
        "Joint_TA",
        "Joint_FR",
        "Joint_FA",
        "Joint_CD",
        "Joint_DE",
    ]
    write_header = (
        not os.path.exists(output_path) or os.path.getsize(output_path) == 0
    )
    with open(output_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(headers)
        for r in rows:
            writer.writerow([r.get(h, "") for h in headers])


def add_args(parser: argparse.ArgumentParser) -> None:
    """Add phone recognition evaluation arguments to an argparse parser."""
    parser.add_argument(
        "--prediction_file", required=True, help="Path to prediction JSON file"
    )
    parser.add_argument(
        "--gt_field",
        type=str,
        default="masked_phones",
        help="Field name for ground truth transcription in the prediction file",
    )
    parser.add_argument(
        "--pred_field",
        type=str,
        default="processed_transcript",
        help="Field name for predicted transcription in the prediction file",
    )
    parser.add_argument(
        "--key_field",
        type=str,
        default="utt_id",
        help="Field name for utterance ID in the prediction file",
    )
    parser.add_argument(
        "--noisy_pr",
        action="store_true",
        help="Whether to evaluate noisy phone recognition",
    )
    parser.add_argument(
        "--output_file", type=str, default=None, help="File to write results to"
    )
    parser.add_argument(
        "--language_field",
        type=str,
        default=None,
        help="If provided, language field must exist in the prediction file and "
        "will be used to produce per language metrics.",
    )
    parser.add_argument("--evaluation_name", type=str, help="name for the evaluation")
    parser.add_argument(
        "--canonical_file",
        type=str,
        default=None,
        help="Required unless --canonical_field is used. Path to a Kaldi-style "
        "text.canonical (utt_id <space-separated-IPA>) used for CMU39 projection.",
    )
    parser.add_argument(
        "--canonical_field",
        type=str,
        default=None,
        help="Read canonical IPA from passthrough[<field>] in the prediction "
        "file. Mutually exclusive with --canonical_file.",
    )
    parser.add_argument(
        "--mdd_per_utt_file",
        type=str,
        default=None,
        help="Output path for per-utt MDD CSV. Defaults to <output_file_dir>/mdd_per_utt.csv.",
    )
    parser.add_argument(
        "--mdd_joint_per_utt_file",
        type=str,
        default=None,
        help="Output path for experimental joint MDD CSV. Only used with --enable_joint_mdd. Defaults to <output_file_dir>/mdd_joint_per_utt.csv.",
    )
    parser.add_argument(
        "--enable_joint_mdd",
        action="store_true",
        help="Compute experimental 3-way joint MDD alignment metrics. Disabled by default because it is expensive on long utterances.",
    )


if __name__ == "__main__":
    import os

    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    if args.canonical_file and args.canonical_field:
        parser.error("--canonical_file and --canonical_field are mutually exclusive")
    if not args.canonical_file and not args.canonical_field:
        parser.error(
            "CMU39-projected evaluation requires --canonical_file or --canonical_field"
        )

    canonical_data: Optional[Dict[str, str]] = None
    if args.canonical_file:
        canonical_data = _load_canonical(args.canonical_file)
        print(f"Loaded {len(canonical_data)} canonical entries from {args.canonical_file}")

    loaded_predictions = _load_predictions(
        args.prediction_file,
        args.language_field,
        canonical_data=canonical_data,
        canonical_field=args.canonical_field,
    )
    print(
        f"Loaded predictions for {len(loaded_predictions)} languages containing {sum(len(v) for v in loaded_predictions.values())} utterances."
    )
    inventories = []
    langs_used = list(loaded_predictions.keys())
    mdd_per_utt_rows: List[Dict[str, Any]] = []
    mdd_joint_per_utt_rows: List[Dict[str, Any]] = []
    for lang, preds in tqdm(loaded_predictions.items(), desc="Evaluating languages"):
        evaluator = PhoneRecognitionEvaluator(
            normalize_ipa=True,
            compute_joint_mdd=args.enable_joint_mdd,
        )
        summary, instance_metrics = evaluator.evaluate(preds)
        inventories.append(summary.inventory)
        if args.output_file:
            assert args.evaluation_name is not None, "Please provide --evaluation_name"
            write_file = args.output_file
            evaluator.write_to_csv(summary, args.evaluation_name, write_file, lang)
            print(f"Appended results to {write_file}")
        if summary.has_mdd:
            for utt_id, m in instance_metrics.items():
                mdd = m.get("mdd")
                if mdd is None:
                    continue
                mdd_per_utt_rows.append(
                    {
                        "eval_name": args.evaluation_name,
                        "language": lang,
                        "utt_id": utt_id,
                        "projection_profile": CMU39_PROJECTION_PROFILE,
                        "prompted": mdd["prompted"],
                        "uttered": mdd["uttered"],
                        "predicted": mdd["predicted"],
                        "corr_U": mdd["corr_U"],
                        "corr_P": mdd["corr_P"],
                        "TR": mdd["TR"],
                        "TA": mdd["TA"],
                        "FR": mdd["FR"],
                        "FA": mdd["FA"],
                        "CD": mdd["CD"],
                        "DE": mdd["DE"],
                    }
                )
                if summary.has_joint_mdd:
                    mdd_joint_per_utt_rows.append(
                        {
                            "eval_name": args.evaluation_name,
                            "language": lang,
                            "utt_id": utt_id,
                            "projection_profile": CMU39_PROJECTION_PROFILE,
                            "joint_prompted": mdd["joint_prompted"],
                            "joint_uttered": mdd["joint_uttered"],
                            "joint_predicted": mdd["joint_predicted"],
                            "joint_corr_U": mdd["joint_corr_U"],
                            "joint_corr_P": mdd["joint_corr_P"],
                            "Joint_TR": mdd["Joint_TR"],
                            "Joint_TA": mdd["Joint_TA"],
                            "Joint_FR": mdd["Joint_FR"],
                            "Joint_FA": mdd["Joint_FA"],
                            "Joint_CD": mdd["Joint_CD"],
                            "Joint_DE": mdd["Joint_DE"],
                        }
                    )

    if mdd_per_utt_rows:
        mdd_path = args.mdd_per_utt_file
        if mdd_path is None and args.output_file:
            mdd_path = os.path.join(
                os.path.dirname(args.output_file) or ".", "mdd_per_utt.csv"
            )
        if mdd_path:
            write_mdd_per_utt_csv(mdd_path, mdd_per_utt_rows)
            print(f"Wrote {len(mdd_per_utt_rows)} per-utt MDD rows to {mdd_path}")
        else:
            print("Skipping per-utt MDD CSV (no --output_file or --mdd_per_utt_file)")

    if mdd_joint_per_utt_rows:
        joint_mdd_path = args.mdd_joint_per_utt_file
        if joint_mdd_path is None and args.output_file:
            joint_mdd_path = os.path.join(
                os.path.dirname(args.output_file) or ".",
                "mdd_joint_per_utt.csv",
            )
        if joint_mdd_path:
            write_mdd_joint_per_utt_csv(
                joint_mdd_path, mdd_joint_per_utt_rows)
            print(
                f"Wrote {len(mdd_joint_per_utt_rows)} per-utt joint MDD rows to {joint_mdd_path}"
            )
        else:
            print(
                "Skipping per-utt joint MDD CSV (no --output_file or --mdd_joint_per_utt_file)"
            )

    all_keys = set().union(*[inv.keys() for inv in inventories])
    macro_inv_dict = {}
    for k in all_keys:
        vals = [inv[k] for inv in inventories]
        macro_inv_dict[k] = sum(vals) / len(vals)
    macro_inventory = setkeydict(list(macro_inv_dict.items()))
    Console().print(
        f"\nMacro-averaged Phone Inventory Metrics over {len(langs_used)} languages:"
    )
    PhoneRecognitionEvaluator.pretty_print_inventory_metrics(macro_inventory)
