"""Builders for the HuPER WavLM recognizer and edit-sequence Corrector.

Usage:
    from src.model.huper.builders import build_huper_recognizer_inference

    inference = build_huper_recognizer_inference(
        hf_repo="huper29/huper_recognizer",
        device="cuda",
    )
"""

from typing import Optional

from transformers import Wav2Vec2Processor, WavLMForCTC

from src.model.huper.huper_corrector_inference import HuperCorrectorInference
from src.model.huper.huper_inference import HuperRecognizerInference
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def build_huper_recognizer_processor(
    hf_repo: str = "huper29/huper_recognizer",
) -> Wav2Vec2Processor:
    """Build the HuPER recognizer's audio processor + phoneme tokenizer."""
    processor = Wav2Vec2Processor.from_pretrained(hf_repo)
    log.info(f"HuPER recognizer processor loaded from {hf_repo}")
    return processor


def build_huper_recognizer_model(
    hf_repo: str = "huper29/huper_recognizer",
) -> WavLMForCTC:
    """Build the HuPER WavLM recognizer model."""
    model = WavLMForCTC.from_pretrained(hf_repo)
    log.info(f"HuPER recognizer model loaded from {hf_repo}")
    return model


def build_huper_recognizer_inference(
    hf_repo: str = "huper29/huper_recognizer",
    device: str = "cpu",
) -> HuperRecognizerInference:
    """Build HuPER recognizer inference runner (compatible with distributed_inference)."""
    model = build_huper_recognizer_model(hf_repo=hf_repo)
    processor = build_huper_recognizer_processor(hf_repo=hf_repo)
    inference = HuperRecognizerInference(model=model, processor=processor, device=device)
    log.info(f"HuPER recognizer inference runner ready on {device}")
    return inference


def build_huper_corrector_inference(
    canonical_file: str,
    repo_id: str = "huper29/huper_corrector",
    checkpoint_path: Optional[str] = None,
    vocab_path: Optional[str] = None,
    audio_model_name: str = "facebook/hubert-large-ls960-ft",
    device: str = "cpu",
) -> HuperCorrectorInference:
    """Build HuPER Corrector inference runner.

    Args:
        canonical_file: Path to a Kaldi-style ``text.canonical`` file (IPA),
            looked up by ``utt_id`` per call.
        repo_id: HF Hub repo for the Corrector checkpoint + ``edit_seq_speech``
            package. Defaults to ``huper29/huper_corrector``.
        checkpoint_path / vocab_path: Optional overrides for files inside the
            snapshot directory.
        audio_model_name: HuBERT tokenizer used by the Corrector.
        device: Target device (injected by ``distributed_inference``).
    """
    inference = HuperCorrectorInference(
        canonical_file=canonical_file,
        repo_id=repo_id,
        checkpoint_path=checkpoint_path,
        vocab_path=vocab_path,
        audio_model_name=audio_model_name,
        device=device,
    )
    log.info(f"HuPER Corrector inference runner ready on {device}")
    return inference
