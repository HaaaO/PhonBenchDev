"""Tests for PhoneRecognitionEvaluator."""

import pytest
from src.metrics.phone_recognition import (
    PhoneRecognitionEvaluator,
    PhoneRecognitionSummary,
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
            {"TA": 3, "FR": 1, "FA": 0, "CD": 0, "DE": 1, "TR": 1},
            list("CECCC"),
            list("CECCE"),
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
def test_mdd_counts_aligns_unequal_lengths_by_canonical_slots(
    prompted,
    uttered,
    predicted,
    expected_counts,
    expected_corr_U,
    expected_corr_P,
):
    """MDD counts should compare U/P errors in the same canonical phone/gap."""
    counts, corr_U, corr_P = _mdd_counts(prompted, uttered, predicted)

    for key, expected in expected_counts.items():
        assert counts[key] == expected
    assert corr_U == expected_corr_U
    assert corr_P == expected_corr_P


def test_evaluate_mdd_summary_formulas_with_mixed_sequence_lengths():
    """Aggregate MDD metrics should use canonical-slot counts across lengths."""
    evaluator = PhoneRecognitionEvaluator(normalize_ipa=True)

    test_data = {
        # TA=3, DE=1, FR=1
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
    assert summary.TA == 8
    assert summary.FR == 2
    assert summary.FA == 1
    assert summary.CD == 1
    assert summary.DE == 1
    assert summary.TR == 2
    assert summary.Detection_Accuracy == pytest.approx(10 / 13)
    assert summary.FRR == pytest.approx(2 / 10)
    assert summary.FAR == pytest.approx(1 / 3)
    assert summary.MDD_Precision == pytest.approx(2 / 4)
    assert summary.MDD_Recall == pytest.approx(2 / 3)
    assert summary.MDD_F1 == pytest.approx(4 / 7)
    assert summary.Diagnostic_Accuracy == pytest.approx(1 / 2)
    assert summary.Diagnostic_Error_Rate == pytest.approx(1 / 2)
    assert summary.True_Diagnostic_Accuracy == pytest.approx(1 / 3)
    assert instance_metrics["fa_and_fr"]["mdd"]["corr_U"] == "C E C C C"
    assert instance_metrics["fa_and_fr"]["mdd"]["corr_P"] == "C C C E C"


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
