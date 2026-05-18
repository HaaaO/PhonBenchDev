import pytest

from src.core.cmu39_projection import (
    CMU39_IPA_SET,
    project_ipa_to_cmu39,
    project_ipa_triplet_to_cmu39,
    segment_ipa_for_cmu39,
)
from src.core.cmu39_model_vocab import (
    DROPPED_MODEL_VOCAB_TOKENS,
    MODEL_VOCAB_TOKEN_SET,
    MODEL_VOCAB_TO_CMU39,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("o ʊ", ["oʊ"]),
        ("oʊ", ["oʊ"]),
        ("e ɪ", ["eɪ"]),
        ("eɪ", ["eɪ"]),
        ("a ɪ", ["aɪ"]),
        ("aɪ", ["aɪ"]),
        ("a ʊ", ["aʊ"]),
        ("aʊ", ["aʊ"]),
        ("ɔ ɪ", ["ɔɪ"]),
        ("ɔɪ", ["ɔɪ"]),
    ],
)
def test_segmenter_keeps_cmu_diphthongs_atomic(raw, expected):
    assert segment_ipa_for_cmu39(raw) == expected


def test_projection_collapses_common_allophones_and_vowel_variants():
    raw = "t ʃ dʒ pʰ tʰ kʰ r ɫ ɨ ə ɝ ɚ ɜ˞ o e a aː ɑː ʉ"
    canonical = "t͡ʃ d͡ʒ p t k ɹ l ɪ ʌ ə˞ ə˞ ə˞ oʊ eɪ ɑ ɑ ɑ u"

    projected = project_ipa_to_cmu39(raw, canonical)

    assert projected == (
        "t͡ʃ d͡ʒ p t k ɹ l ɪ ʌ ə˞ ə˞ ə˞ oʊ eɪ ɑ ɑ ɑ u"
    )


def test_projection_maps_standalone_open_a_to_aa():
    raw = "dejkad͡ʑefuʃibaɹfwaʃ"

    projected = project_ipa_to_cmu39(raw, "")

    assert projected == "d eɪ j k ɑ d͡ʒ eɪ f u ʃ i b ɑ ɹ f w ɑ ʃ"


def test_projection_maps_flaps_with_canonical_context():
    canonical, uttered, predicted = project_ipa_triplet_to_cmu39(
        "t d k",
        "ɾ ɾ ɾ",
        "ɾ ɾ",
    )

    assert canonical == "t d k"
    assert uttered == "t d t"
    assert predicted == "t d"


def test_projection_drops_unmapped_symbols_and_only_emits_cmu39():
    projected = project_ipa_to_cmu39("p [PAD] # ʰ aɪ", "p aɪ")
    tokens = projected.split()

    assert projected == "p aɪ"
    assert all(token in CMU39_IPA_SET for token in tokens)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ɐ", "ʌ"),
        ("ɻ", "ɹ"),
        ("ʐ", "ʒ"),
        ("ʋ", "v"),
        ("ɲ", "n"),
        ("ʈ", "t"),
        ("ɒ", "ɑ"),
        ("ø", "ʊ"),
        ("œ", "ɛ"),
        ("ɜ", "ʌ"),
        ("x", "h"),
        ("t͡s", "t s"),
        ("d͡ʑ", "d͡ʒ"),
    ],
)
def test_projection_maps_model_vocab_non_cmu39_symbols(raw, expected):
    assert project_ipa_to_cmu39(raw, raw) == expected


def test_model_vocab_tokens_are_mapped_or_intentionally_dropped():
    for token in MODEL_VOCAB_TOKEN_SET:
        projected = tuple(project_ipa_to_cmu39(token, token).split())
        if token in DROPPED_MODEL_VOCAB_TOKENS:
            assert projected == ()
            assert token not in MODEL_VOCAB_TO_CMU39
            continue

        assert projected
        assert projected == MODEL_VOCAB_TO_CMU39[token]
        assert all(phone in CMU39_IPA_SET for phone in projected)
