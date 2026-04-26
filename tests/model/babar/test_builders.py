from src.model.babar import builders as babar_builders


def test_build_tokenizer_passes_vocab_path(monkeypatch):
    created = {}

    class DummyTokenizer:
        def __init__(self, **kwargs):
            created.update(kwargs)

    monkeypatch.setattr(
        babar_builders, "Wav2Vec2PhonemeCTCTokenizer", DummyTokenizer
    )

    vocab_path = "/tmp/vocab-phoneme-tinyvox.json"
    tokenizer = babar_builders.build_babar_tokenizer(vocab_path=vocab_path)

    assert isinstance(tokenizer, DummyTokenizer)
    assert created["vocab_file"] == vocab_path
    # All special tokens map to <blank> in BabAR's config
    for key in (
        "eos_token",
        "bos_token",
        "unk_token",
        "pad_token",
        "word_delimiter_token",
    ):
        assert created[key] == "<blank>"
    assert created["do_phonemize"] is False


def test_build_model_passes_paths(monkeypatch):
    created = {}

    class DummyModel:
        def __init__(self, ckpt_path, vocab_path):
            created["ckpt_path"] = ckpt_path
            created["vocab_path"] = vocab_path
            self.vocab_size = 58

    monkeypatch.setattr(babar_builders, "BabarModel", DummyModel)

    ckpt_path = "/tmp/best.ckpt"
    vocab_path = "/tmp/vocab-phoneme-tinyvox.json"
    model = babar_builders.build_babar_model(
        ckpt_path=ckpt_path, vocab_path=vocab_path
    )

    assert isinstance(model, DummyModel)
    assert created["ckpt_path"] == ckpt_path
    assert created["vocab_path"] == vocab_path
    assert model.vocab_size == 58
