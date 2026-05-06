import wave

import pytest

from src.data.kaldi_dataset import build_kaldi_datamodule


def _write_wav(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 160)


def _write_kaldi_dataset(
    tmp_path,
    include_canonical=True,
    include_word_canonical=True,
):
    data_dir = tmp_path / "data"
    ds_dir = data_dir / "sample"
    ds_dir.mkdir(parents=True)
    _write_wav(ds_dir / "utt1.wav")

    (ds_dir / "wav.scp").write_text("utt1 sample/utt1.wav\n", encoding="utf-8")
    (ds_dir / "text.good").write_text("utt1 h aw s\n", encoding="utf-8")
    (ds_dir / "text.lang").write_text("utt1 <eng><test>\n", encoding="utf-8")
    if include_canonical:
        (ds_dir / "text.canonical").write_text("utt1 h aʊ s\n", encoding="utf-8")
    if include_word_canonical:
        (ds_dir / "text_word.canonical").write_text(
            "utt1 house\n", encoding="utf-8"
        )

    config_path = tmp_path / "dataset_index.yaml"
    config_path.write_text(
        "\n".join(
            [
                "datasets:",
                "  sample:",
                '    wav_scp: "sample/wav.scp"',
                '    text_phoneme: "sample/text.good"',
                '    text_canonical: "sample/text.canonical"',
                '    text_word_canonical: "sample/text_word.canonical"',
                '    language: "sample/text.lang"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    return data_dir, config_path


def test_build_kaldi_datamodule_loads_canonical_ipa_from_index(tmp_path):
    data_dir, config_path = _write_kaldi_dataset(tmp_path)

    datamodule = build_kaldi_datamodule(
        "sample",
        data_dir=data_dir,
        dataset_config_path=config_path,
        batch_size=1,
        num_workers=0,
        require_canonical=True,
    )
    datamodule.setup(stage="predict")

    sample = datamodule.predict_dataloader().dataset[0]
    assert sample["canonical_ipa"] == "h aʊ s"

    batch = next(iter(datamodule.predict_dataloader()))
    assert batch["canonical_ipa"] == ["h aʊ s"]


def test_build_kaldi_datamodule_loads_word_canonical_from_index(tmp_path):
    data_dir, config_path = _write_kaldi_dataset(tmp_path)

    datamodule = build_kaldi_datamodule(
        "sample",
        data_dir=data_dir,
        dataset_config_path=config_path,
        batch_size=1,
        num_workers=0,
        require_word_canonical=True,
    )
    datamodule.setup(stage="predict")

    sample = datamodule.predict_dataloader().dataset[0]
    assert sample["canonical_words"] == "house"

    batch = next(iter(datamodule.predict_dataloader()))
    assert batch["canonical_words"] == ["house"]


def test_build_kaldi_datamodule_requires_canonical_file_when_enabled(tmp_path):
    data_dir, config_path = _write_kaldi_dataset(tmp_path, include_canonical=False)

    datamodule = build_kaldi_datamodule(
        "sample",
        data_dir=data_dir,
        dataset_config_path=config_path,
        batch_size=1,
        num_workers=0,
        require_canonical=True,
    )

    with pytest.raises(FileNotFoundError, match="canonical IPA file not found"):
        datamodule.setup(stage="predict")


def test_build_kaldi_datamodule_requires_word_canonical_file_when_enabled(tmp_path):
    data_dir, config_path = _write_kaldi_dataset(
        tmp_path,
        include_word_canonical=False,
    )

    datamodule = build_kaldi_datamodule(
        "sample",
        data_dir=data_dir,
        dataset_config_path=config_path,
        batch_size=1,
        num_workers=0,
        require_word_canonical=True,
    )

    with pytest.raises(
        FileNotFoundError,
        match="word-level canonical text file not found",
    ):
        datamodule.setup(stage="predict")
