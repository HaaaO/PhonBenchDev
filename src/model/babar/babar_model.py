"""BabAR model wrapper: BabyHuBERT encoder + MLP phoneme head (TinyVox-trained).

Adapts BabAR's PyTorch Lightning ``BaseModule`` to PhonBenchDev's shared model
interface (encode / ctc_logits / encoder_output_size / points_by_frames /
get_blank_id / sampling_rate).

BabAR's source tree is put on ``sys.path`` (override with env ``BABAR_SRC``)
rather than installed, because its pyproject pins ``python>=3.13`` and pulls
in pyannote/segma/torchcodec that are not needed for inference.

Usage:
    python -m src.model.babar.babar_model
"""

import os
import sys
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

_BABAR_SRC = os.environ.get("BABAR_SRC", "/n/iqss_sponsored/Lab/zshi/BabAR/src")
if _BABAR_SRC not in sys.path:
    sys.path.insert(0, _BABAR_SRC)

from babar.models.BaseModule import BaseModule  # noqa: E402

from src.utils import RankedLogger  # noqa: E402

log = RankedLogger(__name__, rank_zero_only=True)


def preprocess_inputs_babar(
    processor,
    speech: Union[List[torch.Tensor], torch.Tensor],
    speech_lengths: Union[List[int], torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Prepare a batch of variable-length waveforms for BabAR's processor.

    BabAR's feature extractor is built with ``return_attention_mask=False``
    (its training pipeline uses fixed 20s context windows). We explicitly
    request the mask here so variable-length batching works.
    """
    if isinstance(speech, torch.Tensor):
        speech = speech if speech.ndim == 1 else list(speech)
    batch = [
        x.detach().cpu().float().numpy().squeeze()[: int(xl)]
        for x, xl in zip(speech, speech_lengths)
    ]
    inputs = processor(
        batch,
        sampling_rate=processor.feature_extractor.sampling_rate,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
    )
    return {
        "input_values": inputs.input_values.to(device),
        "attention_mask": inputs.attention_mask.to(device),
    }


class BabarModel(nn.Module):
    """Wrap BabAR's BaseModule to expose PhonBenchDev's model interface."""

    def __init__(self, ckpt_path: str, vocab_path: str):
        super().__init__()
        self.babar = BaseModule.load_from_checkpoint(
            ckpt_path,
            vocab_phoneme_path=vocab_path,
            weights_only=False,
        )
        self.babar.eval()

        encoder_cfg = self.babar.model.encoder.config
        self.encoder_dim = encoder_cfg.hidden_size
        self.vocab_size = self.babar.phonemes_tokenizer.vocab_size
        self.sampling_rate = self.babar.processor.feature_extractor.sampling_rate
        self.blank_id = self.babar.phoneme_blank_id
        self._model_stride = int(np.prod(encoder_cfg.conv_stride))

        log.info(f"BabAR model loaded from {ckpt_path}")
        log.info(
            f"vocab_size={self.vocab_size} blank_id={self.blank_id} "
            f"encoder_dim={self.encoder_dim} stride={self._model_stride}"
        )

    @torch.no_grad()
    def points_by_frames(self) -> int:
        return self._model_stride

    def encoder_output_size(self) -> int:
        return self.encoder_dim

    def get_blank_id(self) -> int:
        return self.blank_id

    def _extract_feats(self, speech, speech_lengths) -> Dict[str, torch.Tensor]:
        return preprocess_inputs_babar(
            self.babar.processor,
            speech,
            speech_lengths,
            device=next(self.parameters()).device,
        )

    def _encode_raw(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoder = self.babar.model.encoder
        out = encoder(
            inputs["input_values"],
            attention_mask=inputs["attention_mask"],
        )
        feat_lens = encoder._get_feat_extract_output_lengths(
            inputs["attention_mask"].sum(-1)
        )
        return out.last_hidden_state, feat_lens

    def encode(
        self, speech, speech_lengths
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + encoder → (B, T, D), (B,)."""
        inputs = self._extract_feats(speech, speech_lengths)
        return self._encode_raw(inputs)

    def ctc_logits(
        self, speech, speech_lengths
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encoder + phoneme MLP head → (B, T, V), (B,)."""
        inputs = self._extract_feats(speech, speech_lengths)
        hidden, lens = self._encode_raw(inputs)
        logits = self.babar.model.phoneme_head(hidden)
        return logits, lens


if __name__ == "__main__":
    ckpt = os.environ.get(
        "BABAR_CKPT",
        "/n/iqss_sponsored/Lab/zshi/PhonBenchDev/exp/download/babar/best.ckpt",
    )
    vocab = os.environ.get(
        "BABAR_VOCAB",
        "/n/iqss_sponsored/Lab/zshi/PhonBenchDev/exp/download/babar/vocab-phoneme-tinyvox.json",
    )

    model = BabarModel(ckpt_path=ckpt, vocab_path=vocab)
    dummy_speech = [torch.randn(16000), torch.randn(8000)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    encoder_out, encoder_out_lens = model.encode(dummy_speech, [16000, 8000])
    print(f"Encoder output shape: {encoder_out.shape}")
    print(f"Encoder output lengths: {encoder_out_lens}")

    logits, logit_lens = model.ctc_logits(dummy_speech, [16000, 8000])
    print(f"CTC logits shape: {logits.shape}")
    print(f"CTC logit lengths: {logit_lens}")
    preds = logits.argmax(dim=-1)
    decoded = model.babar.phonemes_tokenizer.batch_decode(preds)
    print(f"Decoded (on random noise, expected garbage/blanks): {decoded}")
