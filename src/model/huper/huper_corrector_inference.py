"""HuPER Corrector inference runner for distributed transcription.

The Corrector takes (audio, canonical phoneme sequence) and predicts the
realized phoneme sequence via per-position edit operations. Canonical
phonemes are loaded from a Kaldi-style ``text.canonical`` file (IPA,
space-tokenized or run-on) and looked up by ``utt_id`` at inference time.

Pipeline per utterance:
    text.canonical[utt_id] (IPA)
        -> longest-prefix segmentation against IPA_TO_ARPABET keys
        -> ARPAbet token list
        -> PhonemeCorrectionInference.predict(wav_path, " ".join(arpabet))
        -> ARPAbet output
        -> ARPABET_TO_IPA -> IPA tokens
"""

from __future__ import annotations

import os
import sys
import threading
from typing import Any, Dict, List, Optional

import torch

from src.core.ipa_utils import ARPABET_TO_IPA, IPA_TO_ARPABET, IPATokenizer
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

_CORRECTOR_IMPORT_LOCK = threading.Lock()


def _load_canonical(canonical_file: str) -> Dict[str, str]:
    """Parse Kaldi-style text.canonical into {utt_id: ipa_string}."""
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


def _import_phoneme_correction_inference(repo_dir: str):
    """Import HuPER's PhonemeCorrectionInference from a snapshot directory.

    The HF repo bundles the package as ``edit_seq_speech``; we add the
    snapshot dir to sys.path under a lock so concurrent workers do not
    corrupt the import state.
    """
    with _CORRECTOR_IMPORT_LOCK:
        if repo_dir not in sys.path:
            sys.path.append(repo_dir)
        from edit_seq_speech.inference import PhonemeCorrectionInference  # noqa: WPS433
    return PhonemeCorrectionInference


def _ipa_to_arpabet_tokens(ipa_string: str, mapping: Dict[str, str]) -> List[str]:
    """Greedy longest-prefix match against the IPA->ARPAbet table.

    Uses our own segmenter (not panphon) because panphon's ipa_segs splits
    diphthongs like ``aɪ`` into ``['a', 'ɪ']``, which loses the ARPAbet
    diphthong codes the Corrector was trained on (AY/AW/EY/OW/OY).
    """
    keys = sorted(mapping.keys(), key=len, reverse=True)
    s = "".join(ipa_string.split())
    out: List[str] = []
    i = 0
    while i < len(s):
        matched = False
        for k in keys:
            if s.startswith(k, i):
                out.append(mapping[k])
                i += len(k)
                matched = True
                break
        if not matched:
            i += 1
    return out


class HuperCorrectorInference:
    """Single-utterance phoneme inference for HuPER's Corrector model."""

    def __init__(
        self,
        canonical_file: str,
        repo_id: str = "huper29/huper_corrector",
        checkpoint_path: Optional[str] = None,
        vocab_path: Optional[str] = None,
        audio_model_name: str = "facebook/hubert-large-ls960-ft",
        device: str = "cpu",
    ):
        from huggingface_hub import snapshot_download

        repo_dir = snapshot_download(repo_id)
        ckpt = checkpoint_path or os.path.join(repo_dir, "model.safetensors")
        vocab = vocab_path or os.path.join(
            repo_dir, "edit_seq_speech/config/vocab.json"
        )
        PhonemeCorrectionInference = _import_phoneme_correction_inference(repo_dir)

        torch_device = torch.device(device) if isinstance(device, str) else device
        self.device = device
        self.infer = PhonemeCorrectionInference(
            checkpoint_path=ckpt,
            vocab_path=vocab,
            audio_model_name=audio_model_name,
            device=torch_device,
        )

        self.canonical = _load_canonical(canonical_file)
        log.info(
            f"HuPER Corrector loaded canonical for {len(self.canonical)} utts "
            f"from {canonical_file}"
        )
        self.unk_token = IPATokenizer().unk_token

    def _arpabet_seq_for(self, utt_id: str) -> List[str]:
        ipa_string = self.canonical.get(utt_id, "")
        if not ipa_string:
            return []
        return _ipa_to_arpabet_tokens(ipa_string, IPA_TO_ARPABET)

    def __call__(self, *args, **kwargs) -> List[Dict[str, Any]]:
        """Run the Corrector on one utterance.

        Reads ``utt_id`` and ``wavpath`` from kwargs (PhonBench's runner
        spreads the dataset item as kwargs). Returns the standard
        ``[{"processed_transcript", "predicted_transcript"}]`` list.
        """
        utt_id = kwargs.get("utt_id")
        wavpath = kwargs.get("wavpath")
        if not utt_id or not wavpath:
            raise ValueError(
                "HuperCorrectorInference requires utt_id and wavpath in dataset items"
            )

        arpabet = self._arpabet_seq_for(utt_id)
        if not arpabet:
            log.warning(f"No canonical for utt_id={utt_id}; emitting empty prediction")
            return [{"processed_transcript": "", "predicted_transcript": ""}]

        text = " ".join(arpabet)
        final_phns, _log = self.infer.predict(wavpath, text)

        ipa_tokens = [
            ARPABET_TO_IPA.get(p.lower(), self.unk_token) for p in final_phns
        ]
        transcription = " ".join(ipa_tokens)
        processed = " ".join(t for t in ipa_tokens if t != self.unk_token)
        return [
            {
                "processed_transcript": processed,
                "predicted_transcript": transcription,
            }
        ]
