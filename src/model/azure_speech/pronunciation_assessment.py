"""Azure Pronunciation Assessment inference wrapper.

The wrapper adapts Azure Speech's pronunciation assessment to the PhonBench
distributed inference contract. It can run either scripted assessment with
word-level canonical text as Azure's reference text, or unscripted assessment
without a reference text. It then converts Azure's IPA N-best phoneme output
into the standard PhonBench ``processed_transcript`` field.
"""

from __future__ import annotations

import base64
from io import BytesIO
import json
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from pathlib import Path
from typing import Any, Optional


class AzurePronunciationAssessmentInference:
    """Single-utterance Azure Pronunciation Assessment runner."""

    def __init__(
        self,
        speech_key: Optional[str] = None,
        region: Optional[str] = None,
        backend: str = "rest",
        endpoint: Optional[str] = None,
        timeout: float = 60.0,
        language: str = "en-US",
        grading_system: str = "HundredMark",
        granularity: str = "Phoneme",
        dimension: str = "Comprehensive",
        phoneme_alphabet: str = "IPA",
        nbest_phoneme_count: int = 5,
        enable_miscue: bool = True,
        enable_prosody: bool = False,
        use_reference_text: bool = True,
        reference_text_field: str = "canonical_words",
        cache_key_field: str = "utt_id",
        cache_path: Optional[str | Path] = None,
        error_log_path: Optional[str | Path] = None,
        resume: bool = True,
        retry_config: Optional[dict[str, Any]] = None,
        device: Optional[str] = None,
        speechsdk_module: Optional[Any] = None,
    ) -> None:
        del device  # API-based model; kept for distributed_inference injection.

        key = speech_key or os.getenv("AZURE_SPEECH_KEY")
        service_region = region or os.getenv("AZURE_SPEECH_REGION")
        if not key or not service_region:
            raise ValueError(
                "Azure Speech credentials are not configured. Provide speech_key "
                "and region, or set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION."
            )

        self.speech_key = key
        self.region = service_region
        self.backend = backend
        self.endpoint = endpoint or (
            f"https://{service_region}.stt.speech.microsoft.com/"
            "speech/recognition/conversation/cognitiveservices/v1"
        )
        self.timeout = float(timeout)
        self.speechsdk = None
        if self.backend == "sdk":
            self.speechsdk = speechsdk_module or self._import_speechsdk()
            self.speech_config = self.speechsdk.SpeechConfig(
                subscription=key,
                region=service_region,
            )
        elif speechsdk_module is not None:
            self.speechsdk = speechsdk_module
        elif self.backend != "rest":
            raise ValueError(f"Unsupported Azure pronunciation backend: {backend}")
        self.language = language
        self.grading_system = grading_system
        self.granularity = granularity
        self.dimension = dimension
        self.phoneme_alphabet = phoneme_alphabet
        self.nbest_phoneme_count = int(nbest_phoneme_count)
        self.enable_miscue = enable_miscue
        self.enable_prosody = enable_prosody
        self.use_reference_text = use_reference_text
        self.reference_text_field = reference_text_field
        self.cache_key_field = cache_key_field
        self.cache_path = Path(cache_path) if cache_path else None
        self.error_log_path = Path(error_log_path) if error_log_path else None
        self.resume = resume
        default_retry = {"max_retries": 3, "initial_delay": 1.0, "backoff_factor": 2.0}
        self.retry_config = {**default_retry, **(retry_config or {})}
        self._cache: dict[str, Any] = {}
        if self.cache_path and self.resume:
            self._load_cache()

    @staticmethod
    def _import_speechsdk() -> Any:
        try:
            import azure.cognitiveservices.speech as speechsdk  # noqa: WPS433
        except ImportError as e:
            raise ImportError(
                "Azure Speech SDK is required for Azure pronunciation assessment. "
                "Install azure-cognitiveservices-speech."
            ) from e
        return speechsdk

    def _load_cache(self) -> None:
        assert self.cache_path is not None
        if not self.cache_path.exists():
            return
        try:
            with self.cache_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    key = rec.get("key")
                    pred = rec.get("pred")
                    if key is not None and pred is not None:
                        self._cache[str(key)] = pred
        except Exception:
            return

    @staticmethod
    def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, default=str)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            try:
                f.flush()
                os.fsync(f.fileno())
            except OSError:
                pass

    @staticmethod
    def _is_error_pred(pred: Any) -> bool:
        if not isinstance(pred, list) or not pred:
            return True
        first = pred[0]
        return isinstance(first, dict) and "error" in first

    def _cache_key(self, kwargs: dict[str, Any]) -> str:
        if self.cache_key_field in kwargs:
            return str(kwargs[self.cache_key_field])
        return str(
            kwargs.get("utt_id")
            or kwargs.get("key")
            or kwargs.get("audio_path")
            or kwargs.get("wavpath")
            or kwargs.get("metadata_idx")
        )

    def _enum_value(self, enum_name: str, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        if self.speechsdk is None:
            return value
        enum_container = getattr(self.speechsdk, enum_name, None)
        if enum_container is None:
            return value
        return getattr(enum_container, value, value)

    def _make_pronunciation_config(self, reference_text: Optional[str]) -> Any:
        config = self.speechsdk.PronunciationAssessmentConfig(
            reference_text=reference_text or "",
            grading_system=self._enum_value(
                "PronunciationAssessmentGradingSystem", self.grading_system
            ),
            granularity=self._enum_value(
                "PronunciationAssessmentGranularity", self.granularity
            ),
            enable_miscue=self.enable_miscue,
        )
        config.phoneme_alphabet = self.phoneme_alphabet
        config.nbest_phoneme_count = self.nbest_phoneme_count
        if self.enable_prosody:
            config.enable_prosody_assessment()
        return config

    def _response_json_from_result(self, result: Any) -> str:
        property_id = self.speechsdk.PropertyId.SpeechServiceResponse_JsonResult
        properties = result.properties
        if hasattr(properties, "get"):
            raw = properties.get(property_id)
        elif hasattr(properties, "get_property"):
            raw = properties.get_property(property_id)
        else:
            raise ValueError("Azure result properties do not expose JSON lookup.")
        if not raw:
            raise ValueError("Azure pronunciation assessment returned no JSON result.")
        return str(raw)

    def _recognize_once(self, wavpath: str | Path, reference_text: Optional[str]) -> str:
        audio_config = self.speechsdk.audio.AudioConfig(filename=str(wavpath))
        recognizer = self.speechsdk.SpeechRecognizer(
            speech_config=self.speech_config,
            language=self.language,
            audio_config=audio_config,
        )
        pronunciation_config = self._make_pronunciation_config(reference_text)
        pronunciation_config.apply_to(recognizer)
        result = recognizer.recognize_once()
        return self._response_json_from_result(result)

    def _pronunciation_assessment_header(self, reference_text: Optional[str]) -> str:
        params = {
            "GradingSystem": self.grading_system,
            "Granularity": self.granularity,
            "Dimension": self.dimension,
            "EnableMiscue": "True" if self.enable_miscue else "False",
            "phonemeAlphabet": self.phoneme_alphabet,
            "nBestPhonemeCount": self.nbest_phoneme_count,
        }
        if reference_text:
            params["ReferenceText"] = reference_text
        if self.enable_prosody:
            params["EnableProsodyAssessment"] = "True"
        payload = json.dumps(params, ensure_ascii=False, separators=(",", ":"))
        return base64.b64encode(payload.encode("utf-8")).decode("ascii")

    @staticmethod
    def _is_wav_pcm_16k_mono(path: str | Path) -> bool:
        try:
            with wave.open(str(path), "rb") as wav:
                return (
                    wav.getnchannels() == 1
                    and wav.getsampwidth() == 2
                    and wav.getframerate() == 16000
                    and wav.getcomptype() == "NONE"
                )
        except (wave.Error, OSError):
            return False

    @staticmethod
    def _to_rest_wav_bytes(path: str | Path) -> bytes:
        if AzurePronunciationAssessmentInference._is_wav_pcm_16k_mono(path):
            return Path(path).read_bytes()

        import torch
        import torchaudio

        waveform, sr = torchaudio.load(str(path))
        if waveform.ndim == 2 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.squeeze(0)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        waveform = torch.clamp(waveform.to(torch.float32), -1.0, 1.0)
        pcm16 = (waveform.numpy() * 32767.0).astype("<i2")

        buffer = BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(pcm16.tobytes())
        return buffer.getvalue()

    def _recognize_once_rest(
        self,
        wavpath: str | Path,
        reference_text: Optional[str],
    ) -> str:
        query = urllib.parse.urlencode({"language": self.language, "format": "detailed"})
        url = f"{self.endpoint}?{query}"
        headers = {
            "Ocp-Apim-Subscription-Key": self.speech_key,
            "Content-Type": "audio/wav; codecs=audio/pcm; samplerate=16000",
            "Accept": "application/json",
            "Pronunciation-Assessment": self._pronunciation_assessment_header(
                reference_text
            ),
        }
        request = urllib.request.Request(
            url,
            data=self._to_rest_wav_bytes(wavpath),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Azure Speech REST request failed with HTTP {e.code}: {body}"
            ) from e

    def _recognize_with_retries(
        self,
        wavpath: str | Path,
        reference_text: Optional[str],
    ) -> str:
        delay = float(self.retry_config["initial_delay"])
        max_retries = int(self.retry_config["max_retries"])
        backoff = float(self.retry_config["backoff_factor"])
        last_error: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                if self.backend == "sdk":
                    return self._recognize_once(wavpath, reference_text)
                return self._recognize_once_rest(wavpath, reference_text)
            except Exception as e:
                last_error = e
                if attempt >= max_retries - 1:
                    raise
                jitter = random.uniform(0.0, min(0.25, delay * 0.1))
                time.sleep(delay + jitter)
                delay *= backoff

        raise RuntimeError(f"Failed after {max_retries} attempts") from last_error

    @staticmethod
    def _phoneme_prediction(payload: dict[str, Any]) -> str:
        nbest = payload.get("NBest") or []
        if not nbest:
            return ""
        words = nbest[0].get("Words") or []
        phones: list[str] = []
        for word in words:
            for phoneme in word.get("Phonemes") or []:
                assessment = phoneme.get("PronunciationAssessment") or {}
                nbest_phonemes = assessment.get("NBestPhonemes") or []
                if nbest_phonemes:
                    phone = nbest_phonemes[0].get("Phoneme")
                else:
                    phone = phoneme.get("Phoneme")
                if phone:
                    phones.append(str(phone))
        return " ".join(phones)

    def _reference_text(self, kwargs: dict[str, Any]) -> Optional[str]:
        if not self.use_reference_text:
            return None
        if not self.reference_text_field:
            raise ValueError(
                "Azure pronunciation assessment requires reference_text_field "
                "when use_reference_text=True."
            )
        reference_text = str(kwargs.get(self.reference_text_field) or "").strip()
        if not reference_text:
            raise ValueError(
                f"Azure pronunciation assessment requires "
                f"{self.reference_text_field} in dataset items when "
                "use_reference_text=True."
            )
        return reference_text

    def __call__(self, **kwargs: Any) -> Any:
        cache_key = self._cache_key(kwargs)
        if self.cache_path and self.resume and cache_key in self._cache:
            return self._cache[cache_key]

        try:
            reference_text = self._reference_text(kwargs)
            wavpath = kwargs.get("audio_path") or kwargs.get("wavpath")
            if not wavpath:
                raise ValueError(
                    "Azure pronunciation assessment requires audio_path or wavpath."
                )

            raw_model_response = self._recognize_with_retries(wavpath, reference_text)
            payload = json.loads(raw_model_response)
            transcript = self._phoneme_prediction(payload)
        except Exception as e:
            err = {
                "type": type(e).__name__,
                "message": str(e),
            }
            pred = [
                {
                    "processed_transcript": "",
                    "predicted_transcript": "",
                    "raw_model_response": "",
                    "error": err,
                }
            ]
            if self.error_log_path:
                self._append_jsonl(
                    self.error_log_path,
                    {
                        "key": cache_key,
                        "audio_path": str(
                            kwargs.get("audio_path") or kwargs.get("wavpath") or ""
                        ),
                        "error": err,
                    },
                )
            return pred

        pred = [
            {
                "processed_transcript": transcript,
                "predicted_transcript": transcript,
                "raw_model_response": raw_model_response,
            }
        ]
        if self.cache_path and cache_key and not self._is_error_pred(pred):
            self._cache[cache_key] = pred
            self._append_jsonl(self.cache_path, {"key": cache_key, "pred": pred})
        return pred
