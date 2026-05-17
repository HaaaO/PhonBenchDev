import pytest

from src.core.cmu39_projection import (
    CMU39_IPA_SET,
    project_ipa_to_cmu39,
    project_ipa_triplet_to_cmu39,
    segment_ipa_for_cmu39,
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
    raw = "t ʃ dʒ pʰ tʰ kʰ r ɫ ɨ ə ɝ ɚ ɜ˞ o e ʉ"
    canonical = "t͡ʃ d͡ʒ p t k ɹ l ɪ ʌ ə˞ ə˞ ə˞ oʊ eɪ u"

    projected = project_ipa_to_cmu39(raw, canonical)

    assert projected == (
        "t͡ʃ d͡ʒ p t k ɹ l ɪ ʌ ə˞ ə˞ ə˞ oʊ eɪ u"
    )


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
    projected = project_ipa_to_cmu39("p β q ʔ aɪ", "p aɪ")
    tokens = projected.split()

    assert projected == "p aɪ"
    assert all(token in CMU39_IPA_SET for token in tokens)
