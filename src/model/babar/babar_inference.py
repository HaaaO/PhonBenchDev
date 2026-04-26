"""BabAR inference runner for distributed transcription.

Mirrors ``src/model/wav2vec2phoneme/wav2vec2phoneme_inference.py``: wraps a
``BabarModel`` into an object whose ``__call__`` takes a raw speech tensor and
returns a list of one dict with ``processed_transcript`` / ``predicted_transcript``.
"""

from typing import Any, Dict, List

import torch

from src.model.babar.babar_model import BabarModel


class BabarInference:
    """Single-utterance phoneme inference for BabAR via greedy CTC decoding."""

    def __init__(self, model: BabarModel, device: str = "cpu"):
        self.device = device
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = model.babar.phonemes_tokenizer
        self.blank_id = model.get_blank_id()
        self.unk_symbol = self.tokenizer.unk_token
        self.pad_symbol = self.tokenizer.pad_token

    def batchify(self, speech: torch.Tensor):
        speech = speech.unsqueeze(0).to(self.device)
        speech_length = torch.tensor(
            [speech.size(1)], dtype=torch.long, device=self.device
        )
        return speech, speech_length

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

    @torch.no_grad()
    def __call__(self, speech: torch.Tensor, *args, **kwargs) -> List[Dict[str, Any]]:
        """Greedy CTC transcription of a single waveform.

        Args:
            speech: (T,) raw waveform at 16 kHz.

        Returns:
            [{"processed_transcript": str, "predicted_transcript": str}]
        """
        speech_b, speech_length = self.batchify(speech)
        logits, _ = self.model.ctc_logits(speech_b, speech_length)
        predicted_ids = torch.argmax(logits, dim=-1)
        collapsed = self.ctc_collapse(predicted_ids[0].cpu().numpy())
        tokens = self.tokenizer.convert_ids_to_tokens(collapsed)
        # BabAR emits space-separated IPA, matching PhonBench target format.
        transcription = " ".join(tokens)
        processed = transcription.replace(self.unk_symbol, "").replace(
            self.pad_symbol, ""
        )
        processed = " ".join(processed.split())
        return [
            {
                "processed_transcript": processed,
                "predicted_transcript": transcription,
            }
        ]
