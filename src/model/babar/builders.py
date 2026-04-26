"""Builders for BabAR model in PhonBenchDev.

Usage:
    from src.model.babar.builders import build_babar_model

    model = build_babar_model(
        ckpt_path="exp/download/babar/best.ckpt",
        vocab_path="exp/download/babar/vocab-phoneme-tinyvox.json",
    )
"""

from transformers import Wav2Vec2PhonemeCTCTokenizer

from src.model.babar.babar_inference import BabarInference
from src.model.babar.babar_model import BabarModel
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def build_babar_tokenizer(vocab_path: str) -> Wav2Vec2PhonemeCTCTokenizer:
    """Build BabAR's phoneme tokenizer.

    All special tokens map to "<blank>" in BabAR's training config, so they
    resolve to the same id; this matches ``BaseModule.__init__``.
    """
    tokenizer = Wav2Vec2PhonemeCTCTokenizer(
        vocab_file=vocab_path,
        eos_token="<blank>",
        bos_token="<blank>",
        unk_token="<blank>",
        pad_token="<blank>",
        word_delimiter_token="<blank>",
        do_phonemize=False,
    )
    log.info(f"BabAR tokenizer loaded from {vocab_path}")
    return tokenizer


def build_babar_model(ckpt_path: str, vocab_path: str) -> BabarModel:
    """Build BabAR model from a Lightning checkpoint + phoneme vocab.

    Args:
        ckpt_path: Path to BabAR's best.ckpt.
        vocab_path: Path to vocab-phoneme-tinyvox.json.

    Returns:
        BabarModel instance.
    """
    model = BabarModel(ckpt_path=ckpt_path, vocab_path=vocab_path)
    log.info("BabAR model built")
    log.info(f"Model vocab size: {model.vocab_size}")
    return model


def build_babar_inference(
    ckpt_path: str,
    vocab_path: str,
    device: str = "cpu",
) -> BabarInference:
    """Build BabAR inference runner (compatible with distributed_inference).

    Args:
        ckpt_path: Path to BabAR's best.ckpt.
        vocab_path: Path to vocab-phoneme-tinyvox.json.
        device: Target device string (injected by distributed_inference).

    Returns:
        BabarInference callable.
    """
    model = build_babar_model(ckpt_path=ckpt_path, vocab_path=vocab_path)
    inference = BabarInference(model=model, device=device)
    log.info(f"BabAR inference runner ready on {device}")
    return inference
