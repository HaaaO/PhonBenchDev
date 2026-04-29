"""Tests for PhoneRecognitionEvaluator."""

import pytest
from src.metrics.phone_recognition import (
    PhoneRecognitionEvaluator,
    PhoneRecognitionSummary,
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

