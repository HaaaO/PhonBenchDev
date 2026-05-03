from src.model.huper import builders as huper_builders
from src.model.huper.huper_corrector_inference import HuperCorrectorInference
from src.model.huper.huper_inference import HuperRecognizerInference


class _DummyConfig:
    id2label = {0: "<PAD>", 5: "AA"}


class _DummyTokenizer:
    pad_token_id = 0

    @staticmethod
    def convert_ids_to_tokens(token_id):
        return "AA"


class _DummyProcessor:
    def __init__(self, hf_repo):
        self.hf_repo = hf_repo
        self.tokenizer = _DummyTokenizer()

    @classmethod
    def from_pretrained(cls, hf_repo):
        return cls(hf_repo)


class _DummyModel:
    def __init__(self, hf_repo):
        self.hf_repo = hf_repo
        self.config = _DummyConfig()

    @classmethod
    def from_pretrained(cls, hf_repo):
        return cls(hf_repo)

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.is_eval = True
        return self


def test_build_processor_uses_default_repo(monkeypatch):
    monkeypatch.setattr(huper_builders, "Wav2Vec2Processor", _DummyProcessor)

    processor = huper_builders.build_huper_recognizer_processor()

    assert isinstance(processor, _DummyProcessor)
    assert processor.hf_repo == "huper29/huper_recognizer"


def test_build_processor_passes_custom_repo(monkeypatch):
    monkeypatch.setattr(huper_builders, "Wav2Vec2Processor", _DummyProcessor)

    processor = huper_builders.build_huper_recognizer_processor(
        hf_repo="custom/processor"
    )

    assert processor.hf_repo == "custom/processor"


def test_build_model_uses_default_repo(monkeypatch):
    monkeypatch.setattr(huper_builders, "WavLMForCTC", _DummyModel)

    model = huper_builders.build_huper_recognizer_model()

    assert isinstance(model, _DummyModel)
    assert model.hf_repo == "huper29/huper_recognizer"


def test_build_inference_wires_components(monkeypatch):
    monkeypatch.setattr(huper_builders, "WavLMForCTC", _DummyModel)
    monkeypatch.setattr(huper_builders, "Wav2Vec2Processor", _DummyProcessor)

    inference = huper_builders.build_huper_recognizer_inference(
        hf_repo="custom/recognizer", device="cpu"
    )

    assert isinstance(inference, HuperRecognizerInference)
    assert inference.device == "cpu"
    assert inference.model.hf_repo == "custom/recognizer"
    assert inference.processor.hf_repo == "custom/recognizer"
    assert inference.blank_id == 0
    assert inference.model.is_eval is True


def test_build_corrector_inference_passes_args(monkeypatch, tmp_path):
    canonical_file = tmp_path / "text.canonical"
    canonical_file.write_text("utt1 b i t͡ʃ\nutt2 \n")

    captured = {}

    class _StubCorrectorInference:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.canonical = {"utt1": "b i t͡ʃ", "utt2": ""}
            self.device = kwargs.get("device")

    monkeypatch.setattr(
        huper_builders, "HuperCorrectorInference", _StubCorrectorInference
    )

    out = huper_builders.build_huper_corrector_inference(
        canonical_file=str(canonical_file),
        repo_id="custom/corrector",
        checkpoint_path="/tmp/ckpt.safetensors",
        vocab_path="/tmp/vocab.json",
        device="cpu",
    )

    assert isinstance(out, _StubCorrectorInference)
    assert captured["canonical_file"] == str(canonical_file)
    assert captured["repo_id"] == "custom/corrector"
    assert captured["checkpoint_path"] == "/tmp/ckpt.safetensors"
    assert captured["vocab_path"] == "/tmp/vocab.json"
    assert captured["device"] == "cpu"


def test_corrector_ipa_to_arpabet_handles_diphthongs():
    from src.model.huper.huper_corrector_inference import _ipa_to_arpabet_tokens
    from src.core.ipa_utils import IPA_TO_ARPABET

    # Continuous IPA: tokens should preserve diphthongs (AY, OW) and rhotic ER.
    arpa = _ipa_to_arpabet_tokens("aɪtoʊɝ", IPA_TO_ARPABET)
    assert arpa == ["AY", "T", "OW", "ER"]


def test_corrector_load_canonical_parses_kaldi_text(tmp_path):
    from src.model.huper.huper_corrector_inference import _load_canonical

    f = tmp_path / "text.canonical"
    f.write_text("u1 a b c\nu2\n\nu3 x y\n")
    out = _load_canonical(str(f))
    assert out == {"u1": "a b c", "u2": "", "u3": "x y"}


def test_corrector_call_skips_unknown_utt(monkeypatch, tmp_path):
    """If utt_id has no canonical, return empty transcripts."""
    canonical_file = tmp_path / "text.canonical"
    canonical_file.write_text("utt1 b i t͡ʃ\n")

    inference = HuperCorrectorInference.__new__(HuperCorrectorInference)
    inference.canonical = {"utt1": "b i t͡ʃ"}
    inference.unk_token = "<unk>"
    inference.infer = None  # should not be called

    out = inference(speech=None, utt_id="missing", wavpath="/tmp/x.wav")
    assert out == [{"processed_transcript": "", "predicted_transcript": ""}]
