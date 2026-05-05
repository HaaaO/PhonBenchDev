from src.core.ipa_normalization import (
    DEFAULT_IPA_NORMALIZATION_PROFILE,
    looks_ipa_like,
    normalize_ipa_text,
    normalize_transcription_item,
)


def test_normalize_ipa_text_strict_rules():
    raw = "ˈt ʃ ˌd ʒ pʰ tʰ kʰ iː uː oː ɑː r ɫ t̪ s̪ l̪ g ɚ ɝ"

    result = normalize_ipa_text(raw)

    assert result.normalized == "t͡ʃ d͡ʒ p t k i u o ɑ ɹ l t s l ɡ ə˞ ɜ˞"
    assert result.changed is True
    assert set(result.rule_ids) >= {
        "untied_tesh_to_tied",
        "untied_dezh_to_tied",
        "strip_stop_aspiration",
        "strip_length_mark",
        "r_to_english_approximant",
        "dark_l_to_l",
        "strip_dental_diacritic",
        "latin_g_to_ipa_g",
        "rhotic_schwa_decompose",
        "strip_stress_mark",
    }


def test_normalize_ipa_text_english_broad_rules():
    raw = "t ɕʰ ʈ ʂʰ ɕ ʂ d ʑ ɖ ʐ"

    result = normalize_ipa_text(raw)

    assert result.normalized == "t͡ʃ t͡ʃ ʃ ʃ d͡ʒ d͡ʒ"
    assert set(result.rule_ids) >= {
        "eng_t_alveolopalatal_to_tesh",
        "eng_retroflex_to_tesh",
        "eng_alveolopalatal_fricative_to_esh",
        "eng_retroflex_fricative_to_esh",
        "eng_d_alveolopalatal_to_dezh",
        "eng_retroflex_to_dezh",
    }


def test_normalize_ipa_text_preserves_core_english_contrasts():
    raw = "i ɪ u ʊ ɔ ɑ f θ d ð p t k"

    result = normalize_ipa_text(raw)

    assert result.normalized == raw
    assert result.changed is False
    assert result.rule_ids == ()


def test_looks_ipa_like_is_conservative_for_free_form_text():
    assert looks_ipa_like("h aʊ s") is True
    assert looks_ipa_like("p t k") is True
    assert looks_ipa_like("The response is teacher.") is False
    assert looks_ipa_like("raw IPA: t ʃ") is False
    assert looks_ipa_like('{"lat": 42.0, "lon": -71.0}') is False


def test_normalize_transcription_item_preserves_raw_fields_and_reports_rows():
    item = {
        "pred": [
            {
                "processed_transcript": "t ʃ pʰ r g",
                "predicted_transcript": "t ʃ pʰ r g",
            }
        ],
        "passthrough": {
            "utt_id": "utt-1",
            "target": "t ʃ pʰ r g",
            "canonical_ipa": "d ʒ kʰ ɫ",
        },
    }

    normalized, rows = normalize_transcription_item("0", item)

    assert normalized["normalization_profile"] == DEFAULT_IPA_NORMALIZATION_PROFILE
    pred = normalized["pred"][0]
    assert pred["processed_transcript"] == "t͡ʃ p ɹ ɡ"
    assert pred["processed_transcript_raw"] == "t ʃ pʰ r g"
    assert pred["predicted_transcript"] == "t͡ʃ p ɹ ɡ"
    assert pred["predicted_transcript_raw"] == "t ʃ pʰ r g"
    assert normalized["passthrough"]["target"] == "t͡ʃ p ɹ ɡ"
    assert normalized["passthrough"]["target_raw"] == "t ʃ pʰ r g"
    assert normalized["passthrough"]["canonical_ipa"] == "d͡ʒ k l"
    assert normalized["passthrough"]["canonical_ipa_raw"] == "d ʒ kʰ ɫ"
    assert {row["field"] for row in rows} == {
        "pred[0].processed_transcript",
        "pred[0].predicted_transcript",
        "passthrough.target",
        "passthrough.canonical_ipa",
    }
    assert all(row["utt_id"] == "utt-1" for row in rows)


def test_normalize_transcription_item_normalizes_single_ascii_reference_phone():
    item = {
        "pred": [{"processed_transcript": "r"}],
        "passthrough": {"utt_id": "utt-r", "target": "r"},
    }

    normalized, rows = normalize_transcription_item("0", item)

    assert normalized["pred"][0]["processed_transcript"] == "ɹ"
    assert normalized["passthrough"]["target"] == "ɹ"
    assert normalized["passthrough"]["target_raw"] == "r"
    assert {row["field"] for row in rows} == {
        "pred[0].processed_transcript",
        "passthrough.target",
    }
