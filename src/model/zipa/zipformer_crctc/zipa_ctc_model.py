"""A wrapper over icefall ZIPA-CRCTC implementation"

Usage:
    python -m src.model.zipa.zipformer_crctc.zipa_ctc_model
"""

import torch
import torch.nn as nn
from typing import Any, Tuple
import numpy as np
from lhotse.features.kaldi.extractors import Fbank
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class ZipaCtcModel(nn.Module):

    def __init__(self, encoder, blank_id, sampling_rate=16000):
        super().__init__()
        self.encoder = encoder
        self.fbank = Fbank()
        self.blank_id = blank_id
        self.sampling_rate = sampling_rate

    @torch.no_grad()
    def points_by_frames(self) -> int:
        return 320

    def forward(self, inputs) -> Any:
        pass
        # # TODO(shikhar): check with PR
        # # only called in phone recognition recipe when finetuning
        # simple_loss, pruned_loss, ctc_loss, attention_decoder_loss, cr_loss = (
        #     self.encoder(**inputs)
        # )
        # return {
        #     "simple_loss": simple_loss,
        #     "pruned_loss": pruned_loss,
        #     "ctc_loss": ctc_loss,
        #     "attention_decoder_loss": attention_decoder_loss,
        #     "cr_loss": cr_loss,
        # }

    def _extract_feats(self, speech, speech_lengths):
        features = self.fbank.extract_batch(
            speech, lengths=speech_lengths, sampling_rate=self.sampling_rate
        )  # ragged
        if isinstance(features, np.ndarray) and features.ndim == 2:
            # bs=1
            features = [features]
        feature_lens = torch.tensor([len(feature) for feature in features])
        features = [torch.tensor(f, dtype=torch.float32) for f in features]
        features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
        features = features.to(speech.device)
        feature_lens = feature_lens.to(speech.device)
        return features, feature_lens

    def encode(self, speech, speech_lengths) -> Tuple[torch.Tensor, torch.Tensor]:
        feat, featlens = self._extract_feats(speech, speech_lengths)
        encoder_out, encoder_out_lens = self.encoder.forward_encoder(feat, featlens)
        return encoder_out, encoder_out_lens

    def ctc_logits(self, speech, speech_lengths) -> Tuple[torch.Tensor, torch.Tensor]:
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        logits = self.encoder.ctc_output(encoder_out)  # (N, T, C)
        return logits, encoder_out_lens

    def encoder_output_size(self) -> int:
        return self.encoder.encoder_dim

    def get_blank_id(self) -> int:
        return self.blank_id


if __name__ == "__main__":
    from src.model.zipa.zipformer_crctc.builders import build_zipactc_model

    model = build_zipactc_model(
        work_dir="/work/nvme/bbjs/sbharadwaj/powsm/PRiSM/exp/zipactc_cache",
        hf_repo="anyspeech/zipa-large-crctc-500k",
        # hf_repo="anyspeech/zipa-large-crctc-ns-800k",
    )
    print("==" * 30)
    for sec in range(2, 3):
        print(f"Input seconds: {sec}")
        input_size = 16000 * sec
        dummy_input = torch.randn(1, input_size)
        dummy_input_lengths = torch.full((1,), input_size, dtype=torch.long)
        print(
            "SHAPE", dummy_input_lengths.shape, dummy_input.shape, dummy_input_lengths
        )
        logits, logit_lengths = model.ctc_logits(dummy_input, dummy_input_lengths)
        print(logit_lengths)
        print(logits.shape)
        print(model.encoder_output_size(), "encoder output size")
        print("RATIO", input_size / logit_lengths[0].item())
        print("---" * 20)
