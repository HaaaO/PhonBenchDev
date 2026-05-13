"""Tests for PhoneRecognitionEvaluator."""

from types import SimpleNamespace

import pytest
from src.metrics import phone_recognition as phone_recognition_module
from src.metrics.phone_recognition import (
    PhoneRecognitionEvaluator,
    PhoneRecognitionSummary,
    _canonical_edit_cost,
    _joint_mdd_counts,
    _mdd_counts,
)


def test_phone_recognition_evaluator_init():
    """Test PhoneRecognitionEvaluator initialization."""
    evaluator = PhoneRecognitionEvaluator(normalize_ipa=True)
    assert evaluator.normalize_ipa is True
    assert evaluator.dst is not None
    
    evaluator_no_norm = PhoneRecognitionEvaluator(normalize_ipa=False)
    assert evaluator_no_norm.normalize_ipa is False


def test_clean_text():
    """Test clean_text static method."""
    # Test space removal
    assert PhoneRecognitionEvaluator.clean_text("a b c") == "abc"
    
    # Test punctuation removal
    assert PhoneRecognitionEvaluator.clean_text("a,b.c!") == "abc"
    
    # Test 'g' -> 'ɡ' replacement
    assert PhoneRecognitionEvaluator.clean_text("g") == "ɡ"

    # Test stress mark removal
    assert PhoneRecognitionEvaluator.clean_text("ˈt ˌg") == "tɡ"
    
    # Test normalization
    text = "ɑ"
    cleaned = PhoneRecognitionEvaluator.clean_text(text)
    assert isinstance(cleaned, str)


def test_evaluate_perfect_match():
    """Test evaluation with perfect predictions."""
    evaluator = PhoneRecognitionEvaluator(normalize_ipa=True)
    
    test_data = {
        "utt1": {"prediction": "ɑb", "transcription": "ɑb"},
        "utt2": {"prediction": "kæt", "transcription": "kæt"},
    }
    
    summary, instance_metrics = evaluator.evaluate(test_data)
    
    assert isinstance(summary, PhoneRecognitionSummary)
    assert summary.PER == pytest.approx(0.0, abs=0.1)  # Should be very close to 0
    assert summary.N == 2
    assert len(instance_metrics) == 2


def test_evaluate_with_errors():
    """Test evaluation with prediction errors."""
    evaluator = PhoneRecognitionEvaluator(normalize_ipa=True)
    
    test_data = {
        "utt1": {"prediction": "ɑb", "transcription": "ɑp"},  # substitution
        "utt2": {"prediction": "kæ", "transcription": "kæt"},  # deletion
    }
    
    summary, instance_metrics = evaluator.evaluate(test_data)
    
    assert isinstance(summary, PhoneRecognitionSummary)
    assert summary.PER > 0  # Should have errors
    assert summary.N == 2
    assert "utt1" in instance_metrics
    assert "utt2" in instance_metrics


def test_evaluate_empty_data():
    """Test evaluation with empty data."""
    evaluator = PhoneRecognitionEvaluator()
    
    test_data = {}
    summary, instance_metrics = evaluator.evaluate(test_data)
    
    assert summary.N == 0
    assert summary.phones == 0
    assert len(instance_metrics) == 0


def test_load_predictions_treats_malformed_or_error_preds_as_empty(tmp_path, monkeypatch):
    pred_file = tmp_path / "transcription.json"
    pred_file.write_text(
        """
        {
          "0": {
            "pred": [{"processed_transcript": "t"}],
            "passthrough": {"utt_id": "utt-0", "target": "t"}
          },
          "1": {
            "pred": [{"error": {"message": "boom"}}],
            "passthrough": {"utt_id": "utt-1", "target": "k"}
          },
          "2": {
            "pred": [],
            "passthrough": {"utt_id": "utt-2", "target": "s"}
          }
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(
        phone_recognition_module,
        "args",
        SimpleNamespace(
            key_field="utt_id",
            pred_field="processed_transcript",
            gt_field="target",
            noisy_pr=False,
        ),
        raising=False,
    )

    loaded = phone_recognition_module._load_predictions(str(pred_file))

    assert loaded["combined"]["utt-0"]["prediction"] == "t"
    assert loaded["combined"]["utt-1"]["prediction"] == ""
    assert loaded["combined"]["utt-2"]["prediction"] == ""
    assert len(loaded["combined"]) == 3


def test_evaluate_single_utterance():
    """Test evaluation with single utterance."""
    evaluator = PhoneRecognitionEvaluator()
    
    test_data = {
        "utt1": {"prediction": "ɑ", "transcription": "ɑ"},
    }
    
    summary, instance_metrics = evaluator.evaluate(test_data)
    
    assert summary.N == 1
    assert len(instance_metrics) == 1
    assert "utt1" in instance_metrics


def test_pretty_print(capsys):
    """Test pretty_print method."""
    evaluator = PhoneRecognitionEvaluator()
    
    test_data = {
        "utt1": {"prediction": "ɑb", "transcription": "ɑb"},
    }
    
    summary, _ = evaluator.evaluate(test_data)
    evaluator.pretty_print(summary, model_name="test_model", dataset_name="test_set")
    
    # Check that something was printed
    captured = capsys.readouterr()
    assert len(captured.out) > 0


def test_evaluate_with_mdd_cd_de():
    """Test MDD scoring with all five buckets (TA, FR, FA, CD, DE)."""
    evaluator = PhoneRecognitionEvaluator(normalize_ipa=True)

    test_data = {
        "ta": {"prediction": "ab", "transcription": "ab", "canonical": "ab"},  # 2 TA
        "fr": {"prediction": "ac", "transcription": "ab", "canonical": "ab"},  # 1 TA + 1 FR
        "fa": {"prediction": "ab", "transcription": "ac", "canonical": "ab"},  # 1 TA + 1 FA
        "cd": {"prediction": "ac", "transcription": "ac", "canonical": "ab"},  # 1 TA + 1 CD
        "de": {"prediction": "ad", "transcription": "ac", "canonical": "ab"},  # 1 TA + 1 DE
    }

    summary, _ = evaluator.evaluate(test_data)

    assert summary.has_mdd is True
    assert summary.TA == 6  # 2 from "ta" + 1 each from fr/fa/cd/de
    assert summary.FR == 1
    assert summary.FA == 1
    assert summary.CD == 1
    assert summary.DE == 1
    assert summary.TR == 2  # CD + DE
    assert summary.Diagnostic_Accuracy == pytest.approx(0.5)
    assert summary.Diagnostic_Error_Rate == pytest.approx(0.5)


@pytest.mark.parametrize(
    (
        "prompted",
        "uttered",
        "predicted",
        "expected_counts",
        "expected_corr_U",
        "expected_corr_P",
    ),
    [
        pytest.param(
            list("abcd"),
            list("acd"),
            list("axcdy"),
            {"TA": 3, "FR": 2, "FA": 0, "CD": 1, "DE": 0, "TR": 1},
            list("CCECCC"),
            list("CEECCE"),
            id="canonical-4-uttered-3-predicted-5",
        ),
        pytest.param(
            list("abcd"),
            list("abxde"),
            list("abd"),
            {"TA": 3, "FR": 0, "FA": 1, "CD": 0, "DE": 1, "TR": 1},
            list("CCECE"),
            list("CCECC"),
            id="canonical-4-uttered-5-predicted-3",
        ),
        pytest.param(
            list("ab"),
            list("axb"),
            list("axb"),
            {"TA": 2, "FR": 0, "FA": 0, "CD": 1, "DE": 0, "TR": 1},
            list("CEC"),
            list("CEC"),
            id="same-insertion-is-correct-diagnosis",
        ),
        pytest.param(
            list("ab"),
            list("axb"),
            list("ayb"),
            {"TA": 2, "FR": 0, "FA": 0, "CD": 0, "DE": 1, "TR": 1},
            list("CEC"),
            list("CEC"),
            id="different-insertion-is-diagnosis-error",
        ),
        pytest.param(
            list("abc"),
            list("axbc"),
            list("abyc"),
            {"TA": 3, "FR": 1, "FA": 1, "CD": 0, "DE": 0, "TR": 0},
            list("CECCC"),
            list("CCCEC"),
            id="insertions-in-different-canonical-gaps-do-not-match",
        ),
        pytest.param(
            list("abc"),
            list("ac"),
            list("ac"),
            {"TA": 2, "FR": 0, "FA": 0, "CD": 1, "DE": 0, "TR": 1},
            list("CEC"),
            list("CEC"),
            id="shared-deletion-is-correct-diagnosis",
        ),
        pytest.param(
            [],
            list("a"),
            list("ab"),
            {"TA": 0, "FR": 1, "FA": 0, "CD": 1, "DE": 0, "TR": 1},
            list("EC"),
            list("EE"),
            id="empty-canonical-aligns-insertion-gap",
        ),
    ],
)
def test_mdd_counts_uses_pairwise_corrected_alignment(
    prompted,
    uttered,
    predicted,
    expected_counts,
    expected_corr_U,
    expected_corr_P,
):
    """MDD counts should compare correctness vectors on a shared U/H grid."""
    counts, corr_U, corr_P = _mdd_counts(prompted, uttered, predicted)

    for key, expected in expected_counts.items():
        assert counts[key] == expected
    assert corr_U == expected_corr_U
    assert corr_P == expected_corr_P


def test_joint_mdd_alignment_handles_model_insertion_shift():
    """Joint MDD should not let a model-only insertion shift later U/H phones."""
    prompted = "p ɛ ŋ ɡ w ɪ n".split()
    uttered = "p ɪ m w ɪ n".split()
    predicted = "p ɹ ɪ m w ɪ n".split()

    strict_counts, _, _ = _mdd_counts(prompted, uttered, predicted)
    joint_counts, _, _, joint_p, joint_u, joint_h = _joint_mdd_counts(
        prompted, uttered, predicted
    )

    assert strict_counts["FR"] == 1
    assert strict_counts["FA"] == 0
    assert strict_counts["CD"] == 3
    assert strict_counts["DE"] == 0
    assert joint_counts["TA"] == 4
    assert joint_counts["FR"] == 1
    assert joint_counts["FA"] == 0
    assert joint_counts["CD"] == 3
    assert joint_counts["DE"] == 0
    assert joint_p == "p - ɛ ŋ ɡ w ɪ n".split()
    assert joint_u == "p - - ɪ m w ɪ n".split()
    assert joint_h == "p ɹ - ɪ m w ɪ n".split()


@pytest.mark.parametrize(
    ("prompted", "uttered", "predicted"),
    [
        pytest.param(list("abc"), list("abc"), list("abc"), id="exact"),
        pytest.param(list("abc"), list("axc"), list("axc"), id="shared-sub"),
        pytest.param(list("ab"), list("axb"), list("axb"), id="shared-ins"),
        pytest.param(list("abc"), list("ac"), list("ac"), id="shared-del"),
    ],
)
def test_joint_mdd_matches_strict_for_simple_cases(prompted, uttered, predicted):
    strict_counts, _, _ = _mdd_counts(prompted, uttered, predicted)
    joint_counts, _, _, _, _, _ = _joint_mdd_counts(prompted, uttered, predicted)

    assert joint_counts == strict_counts


def test_joint_mdd_keeps_child_canonical_edit_distance_optimal():
    prompted = "p ɛ ŋ ɡ w ɪ n".split()
    uttered = "p ɪ m w ɪ n".split()
    predicted = "p ɹ ɪ m w ɪ n".split()

    _, _, _, joint_p, joint_u, _ = _joint_mdd_counts(
        prompted, uttered, predicted
    )
    joint_cost = sum(
        _canonical_edit_cost(p_sym, u_sym)
        for p_sym, u_sym in zip(joint_p, joint_u)
    )
    dp = [[0] * (len(uttered) + 1) for _ in range(len(prompted) + 1)]
    for i in range(len(prompted) + 1):
        dp[i][0] = i
    for j in range(len(uttered) + 1):
        dp[0][j] = j
    for i, p_sym in enumerate(prompted, 1):
        for j, u_sym in enumerate(uttered, 1):
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + (0 if p_sym == u_sym else 1),
            )
    strict_cost = dp[-1][-1]

    assert joint_cost == strict_cost


def test_joint_mdd_preserves_exact_canonical_prediction_for_child_deletion():
    """A model that outputs the canonical phone missed the child deletion."""
    joint_counts, _, _, joint_p, joint_u, joint_h = _joint_mdd_counts(
        ["s"], [], ["s"]
    )

    assert joint_counts["FA"] == 1
    assert joint_counts["FR"] == 0
    assert joint_counts["CD"] == 0
    assert joint_counts["DE"] == 0
    assert joint_p == ["s"]
    assert joint_u == ["-"]
    assert joint_h == ["s"]


def test_joint_mdd_does_not_split_canonical_prediction_matches_to_create_cd():
    """Exact C/H matches should not become predicted insertions plus deletions."""
    prompted = "a b".split()
    uttered = ["a"]
    predicted = "a b".split()

    joint_counts, _, _, joint_p, joint_u, joint_h = _joint_mdd_counts(
        prompted, uttered, predicted
    )

    assert joint_counts["TA"] == 1
    assert joint_counts["FA"] == 1
    assert joint_counts["FR"] == 0
    assert joint_counts["CD"] == 0
    assert joint_counts["DE"] == 0
    assert joint_p == "a b".split()
    assert joint_u == "a -".split()
    assert joint_h == "a b".split()


def test_evaluate_mdd_summary_formulas_with_mixed_sequence_lengths():
    """Aggregate MDD metrics should use pairwise-corrected MDD counts."""
    evaluator = PhoneRecognitionEvaluator(normalize_ipa=True)

    test_data = {
        # TA=3, CD=1, FR=2
        "de_and_fr": {
            "canonical": "abcd",
            "transcription": "acd",
            "prediction": "axcdy",
        },
        # TA=2, CD=1
        "cd_insertion": {
            "canonical": "ab",
            "transcription": "axb",
            "prediction": "axb",
        },
        # TA=3, FA=1, FR=1
        "fa_and_fr": {
            "canonical": "abc",
            "transcription": "axbc",
            "prediction": "abyc",
        },
    }

    summary, instance_metrics = evaluator.evaluate(test_data)

    assert summary.has_mdd is True
    assert summary.has_joint_mdd is False
    assert summary.TA == 8
    assert summary.FR == 3
    assert summary.FA == 1
    assert summary.CD == 2
    assert summary.DE == 0
    assert summary.TR == 2
    assert summary.Detection_Accuracy == pytest.approx(10 / 14)
    assert summary.FRR == pytest.approx(3 / 11)
    assert summary.FAR == pytest.approx(1 / 3)
    assert summary.MDD_Precision == pytest.approx(2 / 5)
    assert summary.MDD_Recall == pytest.approx(2 / 3)
    assert summary.MDD_F1 == pytest.approx(0.5)
    assert summary.Diagnostic_Accuracy == pytest.approx(1.0)
    assert summary.Diagnostic_Error_Rate == pytest.approx(0.0)
    assert summary.True_Diagnostic_Accuracy == pytest.approx(2 / 3)
    assert instance_metrics["fa_and_fr"]["mdd"]["corr_U"] == "C E C C C"
    assert instance_metrics["fa_and_fr"]["mdd"]["corr_P"] == "C C C E C"
    assert "Joint_TR" not in instance_metrics["fa_and_fr"]["mdd"]


def test_evaluate_can_opt_into_joint_mdd_summary():
    evaluator = PhoneRecognitionEvaluator(
        normalize_ipa=True,
        compute_joint_mdd=True,
    )

    test_data = {
        "utt": {
            "canonical": "p ɛ ŋ ɡ w ɪ n",
            "transcription": "p ɪ m w ɪ n",
            "prediction": "p ɹ ɪ m w ɪ n",
        },
    }

    summary, instance_metrics = evaluator.evaluate(test_data)

    assert summary.has_joint_mdd is True
    assert summary.Joint_TA == 4
    assert summary.Joint_CD == 3
    assert "Joint_TR" in instance_metrics["utt"]["mdd"]


def test_mdd_alignment_preserves_space_separated_phone_tokens():
    evaluator = PhoneRecognitionEvaluator(normalize_ipa=True)

    mdd = evaluator._compute_mdd("t ɔɪ z", "t ɔɪ s", "")

    assert mdd["prompted"] == "t ɔɪ z"
    assert mdd["uttered"] == "t ɔɪ s"
    assert mdd["predicted"] == ""
    assert mdd["corr_U"] == "C C E"
    assert mdd["corr_P"] == "E E E"
    assert mdd["FR"] == 2
    assert mdd["DE"] == 1
    assert mdd["TR"] == 1


def test_mdd_tokenization_keeps_affricate_and_diphthong_tokens():
    evaluator = PhoneRecognitionEvaluator(normalize_ipa=True)

    assert evaluator._mdd_segments("t͡ʃ ɔɪ") == ["t͡ʃ", "ɔɪ"]


def test_evaluate_with_normalization():
    """Test evaluation with and without normalization."""
    test_data = {
        "utt1": {"prediction": "ɑ b", "transcription": "ɑb"},  # space difference
    }
    
    evaluator_norm = PhoneRecognitionEvaluator(normalize_ipa=True)
    summary_norm, _ = evaluator_norm.evaluate(test_data)
    
    evaluator_no_norm = PhoneRecognitionEvaluator(normalize_ipa=False)
    summary_no_norm, _ = evaluator_no_norm.evaluate(test_data)
    
    # Results may differ based on normalization
    assert isinstance(summary_norm, PhoneRecognitionSummary)
    assert isinstance(summary_no_norm, PhoneRecognitionSummary)
