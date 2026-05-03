from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _load_prompt_config(name: str) -> dict:
    path = ROOT / "configs" / "prompt" / f"{name}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_canonical_prompt_only_adds_canonical_ipa_block():
    plain = _load_prompt_config("transcribe_ipa")
    canonical = _load_prompt_config("transcribe_ipa_canonical")

    plain_prompt = plain["prompt_config"]
    canonical_prompt = canonical["prompt_config"]

    canonical_block = (
        "\n\n### Canonical Context\n"
        "The intended canonical IPA pronunciation is:\n"
        "{canonical_ipa}\n\n"
    )
    canonical_without_block = canonical_prompt["system_prompt"].replace(
        canonical_block,
        "\n\n",
    )

    assert canonical_without_block == plain_prompt["system_prompt"]
    assert canonical_prompt["user_prompt"] == plain_prompt["user_prompt"]
    assert canonical["client_config"] == plain["client_config"]
    assert canonical["output_key"] == plain["output_key"]
    assert canonical["clean_response"] == plain["clean_response"]
