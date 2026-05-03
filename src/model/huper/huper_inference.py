"""HuPER recognizer inference runner for distributed transcription.

Mirrors ``src/model/babar/babar_inference.py``: wraps a HuggingFace
``WavLMForCTC`` + ``Wav2Vec2Processor`` pair into an object whose ``__call__``
takes a raw speech tensor and returns a list of one dict with
``processed_transcript`` / ``predicted_transcript``.

The recognizer's native vocabulary is uppercase ARPAbet (see
``HuPER/wavlm_ft/vocab_phoneme.json``); outputs are mapped to IPA via
``src.core.ipa_utils.ARPABET_TO_IPA`` to stay consistent with the rest of
PhonBench.
"""

from typing import Any, Dict, List

import torch

from src.core.ipa_utils import ARPABET_TO_IPA, IPATokenizer

SPECIAL_TOKENS = {"<PAD>", "<UNK>", "<BOS>", "<EOS>", "|"}


class HuperRecognizerInference:
    """Single-utterance phoneme inference for HuPER's WavLM recognizer."""

    def __init__(self, model, processor, device: str = "cpu"):
        self.device = device
        self.model = model.to(device)
        self.model.eval()
        self.processor = processor
        self.blank_id = processor.tokenizer.pad_token_id
        self.id2label = getattr(model.config, "id2label", {}) or {}
        self.unk_token = IPATokenizer().unk_token

    def ctc_collapse(self, predicted_ids) -> List[int]:
        collapsed: List[int] = []
        prev = None
        for idx in predicted_ids:
            idx = int(idx)
            if idx == self.blank_id:
                prev = idx
                continue
            if idx == prev:
                continue
            collapsed.append(idx)
            prev = idx
        return collapsed

    def _id_to_arpabet(self, token_id: int) -> str:
        if token_id in self.id2label:
            return self.id2label[token_id]
        return self.processor.tokenizer.convert_ids_to_tokens(token_id)

    def _arpabet_to_ipa(self, arpabet: str) -> str:
        return ARPABET_TO_IPA.get(arpabet.lower(), self.unk_token)

    @torch.no_grad()
    def __call__(self, speech: torch.Tensor, *args, **kwargs) -> List[Dict[str, Any]]:
        """Greedy CTC transcription of a single waveform.

        Args:
            speech: (T,) raw waveform at 16 kHz.

        Returns:
            [{"processed_transcript": str, "predicted_transcript": str}]
        """
        waveform = speech.detach().to("cpu").float().squeeze().numpy()
        inputs = self.processor(
            waveform, sampling_rate=16000, return_tensors="pt"
        )
        input_values = inputs["input_values"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
            logits = self.model(
                input_values, attention_mask=attention_mask
            ).logits
        else:
            logits = self.model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        collapsed = self.ctc_collapse(predicted_ids[0].cpu().numpy())

        arpabet_tokens = [self._id_to_arpabet(i) for i in collapsed]
        arpabet_tokens = [t for t in arpabet_tokens if t not in SPECIAL_TOKENS]
        ipa_tokens = [self._arpabet_to_ipa(t) for t in arpabet_tokens]
        transcription = " ".join(ipa_tokens)
        processed = " ".join(t for t in ipa_tokens if t != self.unk_token)
        return [
            {
                "processed_transcript": processed,
                "predicted_transcript": transcription,
            }
        ]
