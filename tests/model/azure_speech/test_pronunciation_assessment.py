import base64
import json

from src.model.azure_speech.pronunciation_assessment import (
    AzurePronunciationAssessmentInference,
)


def _azure_json(*phones):
    phonemes = []
    for expected, spoken in phones:
        phonemes.append(
            {
                "Phoneme": expected,
                "PronunciationAssessment": {
                    "AccuracyScore": 90.0,
                    "NBestPhonemes": [
                        {"Phoneme": spoken, "Score": 100.0},
                        {"Phoneme": expected, "Score": 50.0},
                    ],
                },
            }
        )
    return json.dumps(
        {
            "NBest": [
                {
                    "Words": [
                        {
                            "Word": "hello",
                            "Phonemes": phonemes,
                        }
                    ]
                }
            ]
        },
        ensure_ascii=False,
    )


class FakeProperties:
    def __init__(self, result_json):
        self.result_json = result_json

    def get(self, property_id):
        assert property_id == "json"
        return self.result_json


class FakeSpeechResult:
    def __init__(self, result_json):
        self.properties = FakeProperties(result_json)


class FakeSpeechSdk:
    result_json = _azure_json(("h", "h"))
    configs = []
    recognizers = []

    class PropertyId:
        SpeechServiceResponse_JsonResult = "json"

    class PronunciationAssessmentGradingSystem:
        HundredMark = "HundredMark"

    class PronunciationAssessmentGranularity:
        Phoneme = "Phoneme"

    class SpeechConfig:
        def __init__(self, subscription, region):
            self.subscription = subscription
            self.region = region

    class audio:
        class AudioConfig:
            def __init__(self, filename):
                self.filename = filename

    class PronunciationAssessmentConfig:
        def __init__(
            self,
            reference_text,
            grading_system,
            granularity,
            enable_miscue,
        ):
            self.reference_text = reference_text
            self.grading_system = grading_system
            self.granularity = granularity
            self.enable_miscue = enable_miscue
            self.phoneme_alphabet = None
            self.nbest_phoneme_count = None
            self.prosody_enabled = False
            FakeSpeechSdk.configs.append(self)

        def enable_prosody_assessment(self):
            self.prosody_enabled = True

        def apply_to(self, recognizer):
            recognizer.pronunciation_config = self

    class SpeechRecognizer:
        def __init__(self, speech_config, language, audio_config):
            self.speech_config = speech_config
            self.language = language
            self.audio_config = audio_config
            self.pronunciation_config = None
            FakeSpeechSdk.recognizers.append(self)

        def recognize_once(self):
            return FakeSpeechResult(FakeSpeechSdk.result_json)


def _reset_fake_sdk():
    FakeSpeechSdk.result_json = _azure_json(("h", "h"))
    FakeSpeechSdk.configs = []
    FakeSpeechSdk.recognizers = []


def _inference(tmp_path, **overrides):
    kwargs = {
        "speech_key": "key",
        "region": "region",
        "backend": "sdk",
        "speechsdk_module": FakeSpeechSdk,
        "cache_path": tmp_path / "cache.jsonl",
        "error_log_path": tmp_path / "errors.jsonl",
        "retry_config": {"max_retries": 1},
    }
    kwargs.update(overrides)
    return AzurePronunciationAssessmentInference(**kwargs)


def test_azure_inference_uses_canonical_words_and_outputs_spoken_ipa(tmp_path):
    _reset_fake_sdk()
    FakeSpeechSdk.result_json = _azure_json(
        ("h", "h"),
        ("ɛ", "ə"),
        ("l", "l"),
        ("oʊ", "oʊ"),
    )
    inf = _inference(tmp_path)

    pred = inf(wavpath="audio.wav", canonical_words="hello", utt_id="utt1")

    assert pred[0]["processed_transcript"] == "h ə l oʊ"
    assert pred[0]["predicted_transcript"] == "h ə l oʊ"
    assert json.loads(pred[0]["raw_model_response"])["NBest"]

    config = FakeSpeechSdk.configs[0]
    assert config.reference_text == "hello"
    assert config.grading_system == "HundredMark"
    assert config.granularity == "Phoneme"
    assert config.enable_miscue is True
    assert config.phoneme_alphabet == "IPA"
    assert config.nbest_phoneme_count == 5

    recognizer = FakeSpeechSdk.recognizers[0]
    assert recognizer.language == "en-US"
    assert recognizer.audio_config.filename == "audio.wav"


def test_azure_inference_returns_error_when_canonical_words_missing(tmp_path):
    _reset_fake_sdk()
    inf = _inference(tmp_path)

    pred = inf(wavpath="audio.wav", utt_id="utt1")

    assert pred[0]["processed_transcript"] == ""
    assert pred[0]["error"]["type"] == "ValueError"
    assert "canonical_words" in pred[0]["error"]["message"]
    assert "use_reference_text=True" in pred[0]["error"]["message"]
    assert FakeSpeechSdk.recognizers == []


def test_azure_inference_can_run_without_reference_text(tmp_path):
    _reset_fake_sdk()
    FakeSpeechSdk.result_json = _azure_json(("h", "h"), ("ɛ", "ə"))
    inf = _inference(tmp_path, use_reference_text=False)

    pred = inf(wavpath="audio.wav", utt_id="utt1")

    assert pred[0]["processed_transcript"] == "h ə"
    assert FakeSpeechSdk.configs[0].reference_text == ""
    assert len(FakeSpeechSdk.recognizers) == 1


def test_azure_inference_returns_error_when_audio_path_missing(tmp_path):
    _reset_fake_sdk()
    inf = _inference(tmp_path)

    pred = inf(canonical_words="hello", utt_id="utt1")

    assert pred[0]["processed_transcript"] == ""
    assert pred[0]["error"]["type"] == "ValueError"
    assert "audio_path or wavpath" in pred[0]["error"]["message"]
    assert FakeSpeechSdk.recognizers == []


def test_azure_inference_uses_cache_on_resume(tmp_path):
    _reset_fake_sdk()
    FakeSpeechSdk.result_json = _azure_json(("h", "h"))
    inf = _inference(tmp_path)

    first = inf(wavpath="audio-a.wav", canonical_words="hello", utt_id="utt1")
    second = inf(wavpath="audio-b.wav", canonical_words="hello", utt_id="utt1")

    assert second == first
    assert len(FakeSpeechSdk.recognizers) == 1
    record = json.loads((tmp_path / "cache.jsonl").read_text(encoding="utf-8"))
    assert record["key"] == "utt1"


def test_azure_inference_falls_back_to_expected_phoneme_without_nbest(tmp_path):
    _reset_fake_sdk()
    FakeSpeechSdk.result_json = json.dumps(
        {
            "NBest": [
                {
                    "Words": [
                        {
                            "Word": "cat",
                            "Phonemes": [
                                {
                                    "Phoneme": "k",
                                    "PronunciationAssessment": {
                                        "AccuracyScore": 100.0,
                                    },
                                }
                            ],
                        }
                    ]
                }
            ]
        },
        ensure_ascii=False,
    )
    inf = _inference(tmp_path)

    pred = inf(wavpath="audio.wav", canonical_words="cat", utt_id="utt1")

    assert pred[0]["processed_transcript"] == "k"


def test_rest_pronunciation_header_omits_reference_text_when_unscripted(tmp_path):
    inf = AzurePronunciationAssessmentInference(
        speech_key="key",
        region="region",
        backend="rest",
        use_reference_text=False,
        cache_path=tmp_path / "cache.jsonl",
        retry_config={"max_retries": 1},
    )

    header = inf._pronunciation_assessment_header(None)
    payload = json.loads(base64.b64decode(header).decode("utf-8"))

    assert "ReferenceText" not in payload
    assert payload["phonemeAlphabet"] == "IPA"
    assert payload["nBestPhonemeCount"] == 5


def test_rest_pronunciation_header_keeps_reference_text_when_scripted(tmp_path):
    inf = AzurePronunciationAssessmentInference(
        speech_key="key",
        region="region",
        backend="rest",
        use_reference_text=True,
        cache_path=tmp_path / "cache.jsonl",
        retry_config={"max_retries": 1},
    )

    header = inf._pronunciation_assessment_header("hello")
    payload = json.loads(base64.b64decode(header).decode("utf-8"))

    assert payload["ReferenceText"] == "hello"
