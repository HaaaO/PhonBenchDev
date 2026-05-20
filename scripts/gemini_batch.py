#!/usr/bin/env python3
"""Gemini Batch API helpers for PhonBenchDev.

Subcommands:
- ``submit`` builds and uploads a Gemini Batch JSONL job from a Hydra inference
  config.
- ``status`` checks or waits on an existing Gemini batch job.
- ``collect`` downloads/converts results, merges to ``transcription.json``, and
  runs the standard phone-recognition/MDD evaluation.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import mimetypes
import json
import os
import shutil
import string
import subprocess
import sys
import tempfile
import unicodedata
import uuid
from pathlib import Path
from typing import Any, Iterable, Optional

from omegaconf import DictConfig, ListConfig, OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_ENV_FILE = REPO_ROOT / ".env"
TERMINAL_BATCH_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}
DEFAULT_PASSTHROUGH_KEYS = (
    "target",
    "split",
    "utt_id",
    "metadata_idx",
    "lang_sym",
    "canonical_ipa",
)
DEFAULT_SUBMIT_OVERRIDES = (
    "experiment=inference/transcribe_gemini31pro_batch",
    "data=powsmeval",
    "data.dataset_name=authentic_kids_kaldi",
    "data.data_dir=/n/iqss_sponsored/Lab/zshi/prism-evalsets",
    "data.portable_wavscp=True",
)
DEFAULT_SUBMIT_TASK_PREFIX = "inf_authentic_kids_kaldi_gemini31pro_batch"
SUBMIT_PRESETS = {
    "authentic_kids_kaldi": {
        "task_prefix": DEFAULT_SUBMIT_TASK_PREFIX,
        "evaluation_name": "gemini31pro_batch",
        "canonical_file": "/n/iqss_sponsored/Lab/zshi/prism-evalsets/authentic_kids_kaldi/text.canonical",
        "overrides": DEFAULT_SUBMIT_OVERRIDES,
    },
    "cmu_kids_final_kaldi": {
        "task_prefix": "inf_cmu_kids_final_kaldi_gemini31pro_batch",
        "evaluation_name": "gemini31pro_batch",
        "canonical_file": "/n/iqss_sponsored/Lab/zshi/prism-evalsets/cmu_kids_final_kaldi/text.canonical",
        "overrides": (
            "experiment=inference/transcribe_gemini31pro_batch",
            "data=powsmeval",
            "data.dataset_name=cmu_kids_final_kaldi",
            "data.data_dir=/n/iqss_sponsored/Lab/zshi/prism-evalsets/cmu_kids_final_kaldi",
            "data.portable_wavscp=True",
        ),
    },
    "test_l2arctic_perceived": {
        "task_prefix": "inf_test_l2arctic_perceived_gemini31pro_batch",
        "evaluation_name": "gemini31pro_batch",
        "canonical_file": "/n/netscratch/iqss_sponsored/Lab/zshi/prism-evalsets/test_l2arctic_perceived/text.canonical",
        "overrides": (
            "experiment=inference/transcribe_gemini31pro_batch",
            "data=powsmeval",
            "data.dataset_name=test_l2arctic_perceived",
            "data.data_dir=/n/netscratch/iqss_sponsored/Lab/zshi/prism-evalsets",
            "data.portable_wavscp=True",
        ),
    },
    "l2_arctic": {
        "task_prefix": "inf_test_l2arctic_perceived_gemini31pro_batch",
        "evaluation_name": "gemini31pro_batch",
        "canonical_file": "/n/netscratch/iqss_sponsored/Lab/zshi/prism-evalsets/test_l2arctic_perceived/text.canonical",
        "overrides": (
            "experiment=inference/transcribe_gemini31pro_batch",
            "data=powsmeval",
            "data.dataset_name=test_l2arctic_perceived",
            "data.data_dir=/n/netscratch/iqss_sponsored/Lab/zshi/prism-evalsets",
            "data.portable_wavscp=True",
        ),
    },
    "synthetic_word_kaldi_5_19": {
        "task_prefix": "inf_synthetic_word_kaldi_5_19_gemini31pro_batch",
        "evaluation_name": "gemini31pro_batch",
        "canonical_file": "/n/iqss_sponsored/Lab/zshi/prism-evalsets/synthetic_word_kaldi_5_19/text.canonical",
        "overrides": (
            "experiment=inference/transcribe_gemini31pro_batch",
            "data=powsmeval",
            "data.dataset_name=synthetic_word_kaldi_5_19",
            "data.data_dir=/n/iqss_sponsored/Lab/zshi/prism-evalsets/synthetic_word_kaldi_5_19",
            "data.portable_wavscp=True",
        ),
    },
}
DEFAULT_SUBMIT_PRESET = "authentic_kids_kaldi"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on {path}:{line_no}: {e}") from e
            if not isinstance(row, dict):
                raise ValueError(f"Expected object on {path}:{line_no}, got {type(row)}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=False)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return to_jsonable(value.model_dump(by_alias=True, exclude_none=True))
        except TypeError:
            return to_jsonable(value.model_dump())
    if hasattr(value, "__dict__"):
        return to_jsonable(
            {k: v for k, v in vars(value).items() if not k.startswith("_")}
        )
    return str(value)


def cfg_to_container(value: Any, *, resolve: bool = False) -> Any:
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=resolve)
    return value


def get_batch_state(batch_job: Any) -> str:
    state = getattr(batch_job, "state", None)
    if hasattr(state, "name"):
        return str(state.name)
    if state is None and isinstance(batch_job, dict):
        state = batch_job.get("state") or batch_job.get("metadata", {}).get("state")
    return str(state or "")


def get_batch_dest_file_name(batch_job: Any) -> Optional[str]:
    dest = getattr(batch_job, "dest", None)
    if dest is not None:
        file_name = getattr(dest, "file_name", None) or getattr(dest, "fileName", None)
        if file_name:
            return str(file_name)
    if isinstance(batch_job, dict):
        dest_dict = batch_job.get("dest") or batch_job.get("response") or {}
        if isinstance(dest_dict, dict):
            file_name = (
                dest_dict.get("file_name")
                or dest_dict.get("fileName")
                or dest_dict.get("responsesFile")
            )
            if file_name:
                return str(file_name)
    return None


def make_client(api_key: Optional[str] = None) -> Any:
    try:
        from google import genai
    except ImportError as e:
        raise ImportError(
            "google-genai is required for Gemini batch submit/status/collect."
        ) from e

    load_default_env()
    resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
    return genai.Client(api_key=resolved_key) if resolved_key else genai.Client()


def clean_response(text: str) -> str:
    text = "".join(str(text).split())
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = unicodedata.normalize("NFD", text)
    text = text.replace("g", "ɡ")
    return text.strip()


def parse_json_response(response: str, key: str) -> str:
    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        return response
    if isinstance(parsed, dict) and key in parsed:
        return str(parsed[key])
    return response


def manifest_key(row: dict[str, Any]) -> str:
    for field in ("key", "batch_key", "utt_id", "request_key"):
        value = row.get(field)
        if value is not None:
            return str(value)
    raise ValueError(f"Manifest row is missing key/batch_key/utt_id: {row}")


def manifest_index(row: dict[str, Any], fallback: int) -> str:
    for field in ("idx", "index", "dataset_index", "metadata_idx"):
        value = row.get(field)
        if value is not None:
            return str(value)
    passthrough = row.get("passthrough")
    if isinstance(passthrough, dict) and passthrough.get("metadata_idx") is not None:
        return str(passthrough["metadata_idx"])
    return str(fallback)


def manifest_passthrough(row: dict[str, Any]) -> dict[str, Any]:
    passthrough = row.get("passthrough")
    if isinstance(passthrough, dict):
        return dict(passthrough)
    return {
        key: row[key]
        for key in DEFAULT_PASSTHROUGH_KEYS
        if key in row
    }


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    manifest: dict[str, dict[str, Any]] = {}
    for fallback, row in enumerate(read_jsonl(path)):
        key = manifest_key(row)
        manifest[key] = {
            "index": manifest_index(row, fallback),
            "passthrough": manifest_passthrough(row),
            "audio_path": row.get("audio_path") or row.get("wavpath"),
            "raw": row,
        }
    return manifest


def result_key(row: dict[str, Any]) -> Optional[str]:
    for field in ("key", "batch_key", "request_key"):
        value = row.get(field)
        if value is not None:
            return str(value)
    metadata = row.get("metadata")
    if isinstance(metadata, dict) and metadata.get("key") is not None:
        return str(metadata["key"])
    return None


def response_text(response: Any) -> str:
    if not isinstance(response, dict):
        return ""
    for field in ("text", "output_text"):
        value = response.get(field)
        if value is not None:
            return str(value).strip()
    candidates = response.get("candidates") or []
    parts: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content")
        if not isinstance(content, dict):
            continue
        for part in content.get("parts") or []:
            if isinstance(part, dict) and part.get("text") is not None:
                parts.append(str(part["text"]))
    return "".join(parts).strip()


def response_payload(row: dict[str, Any]) -> Any:
    if "response" in row:
        return row["response"]
    result = row.get("result")
    if isinstance(result, dict):
        return result.get("response")
    return None


def row_error(row: dict[str, Any]) -> Any:
    if row.get("error"):
        return row["error"]
    result = row.get("result")
    if isinstance(result, dict) and result.get("error"):
        return result["error"]
    return None


def empty_error_prediction(error: dict[str, Any], raw: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "processed_transcript": "",
            "predicted_transcript": "",
            "raw_model_response": json.dumps(raw, ensure_ascii=False, default=str),
            "error": error,
        }
    ]


def prediction_from_result(
    row: dict[str, Any],
    *,
    output_key: Optional[str],
    clean: bool,
) -> list[dict[str, Any]]:
    error_value = row_error(row)
    if error_value:
        error = error_value if isinstance(error_value, dict) else {"message": str(error_value)}
        return empty_error_prediction(error, row)

    text = response_text(response_payload(row))
    if not text:
        return empty_error_prediction(
            {"type": "EmptyGeminiBatchResponse", "message": "No text found in batch response"},
            row,
        )

    raw_transcript = parse_json_response(text, output_key) if output_key else text
    processed = clean_response(raw_transcript) if clean else raw_transcript
    return [
        {
            "processed_transcript": processed,
            "predicted_transcript": raw_transcript,
            "raw_model_response": text,
        }
    ]


def error_log_key(key: str, item: dict[str, Any]) -> str:
    passthrough = item.get("passthrough")
    if isinstance(passthrough, dict):
        for field in ("utt_id", "key", "metadata_idx"):
            value = passthrough.get(field)
            if value is not None:
                return str(value)
    return key


def compose_hydra_config(overrides: list[str]) -> DictConfig:
    os.environ.setdefault("PROJECT_ROOT", str(REPO_ROOT))
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(REPO_ROOT / "configs"),
        version_base="1.3",
    ):
        return compose(config_name="main", overrides=overrides)


def override_sets_key(overrides: list[str], key: str) -> bool:
    prefixes = (f"{key}=", f"+{key}=", f"++{key}=", f"~{key}")
    return any(override.startswith(prefixes) for override in overrides)


def submit_overrides(
    user_overrides: list[str],
    *,
    preset_name: str = DEFAULT_SUBMIT_PRESET,
) -> list[str]:
    if preset_name not in SUBMIT_PRESETS:
        valid = ", ".join(sorted(SUBMIT_PRESETS))
        raise ValueError(f"Unknown submit preset {preset_name!r}; valid presets: {valid}")

    preset = SUBMIT_PRESETS[preset_name]
    overrides: list[str] = []
    for default_override in preset["overrides"]:
        key = default_override.split("=", 1)[0]
        if not override_sets_key(user_overrides, key):
            overrides.append(default_override)

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not override_sets_key(user_overrides, "task_name"):
        overrides.append(f"task_name={preset['task_prefix']}_{run_tag}")
    if not override_sets_key(user_overrides, "run_folder"):
        overrides.append(f"run_folder='{run_tag}'")

    overrides.extend(user_overrides)
    return overrides


def run_dir_batch_config(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "batch_config.json"
    if not config_path.exists():
        return {}
    data = read_json(config_path)
    return data if isinstance(data, dict) else {}


def preset_metadata_from_run_dir(run_dir: Path) -> dict[str, Any]:
    preset_name = run_dir_batch_config(run_dir).get("preset")
    if not isinstance(preset_name, str):
        return {}
    preset = SUBMIT_PRESETS.get(preset_name, {})
    return preset if isinstance(preset, dict) else {}


def collect_default_evaluation_name(run_dir: Path) -> str:
    preset_eval = preset_metadata_from_run_dir(run_dir).get("evaluation_name")
    if isinstance(preset_eval, str) and preset_eval:
        return preset_eval
    return run_dir.parent.name


def collect_default_canonical_file(run_dir: Path) -> Optional[Path]:
    preset_canonical = preset_metadata_from_run_dir(run_dir).get("canonical_file")
    if not isinstance(preset_canonical, str) or not preset_canonical:
        return None
    path = Path(preset_canonical).expanduser()
    return path.resolve() if path.exists() else None


def node_to_plain_dict(node: Any) -> dict[str, Any]:
    if node is None:
        return {}
    value = cfg_to_container(node, resolve=False)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"Expected config mapping, got {type(value)}")
    return dict(value)


def resolve_config_values(value: Any, cfg: DictConfig) -> Any:
    if isinstance(value, dict):
        return {k: resolve_config_values(v, cfg) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_config_values(v, cfg) for v in value]
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        inner = value[2:-1]
        if inner.startswith("oc.env:"):
            return value
        selected = OmegaConf.select(cfg, inner, default=None)
        return selected if selected is not None else value
    return value


def resolve_env_api_key(value: Any) -> Optional[str]:
    load_default_env()
    if value is None:
        return os.environ.get("GEMINI_API_KEY")
    value_str = str(value)
    if value_str.startswith("${oc.env:") and value_str.endswith("}"):
        inner = value_str[len("${oc.env:"):-1]
        env_name = inner.split(",", 1)[0].strip()
        return os.environ.get(env_name)
    return value_str


def load_default_env() -> None:
    """Load repo-local .env without overriding already-exported variables."""
    if not DEFAULT_ENV_FILE.exists():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(DEFAULT_ENV_FILE, override=False)


def cfg_value(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, (DictConfig, ListConfig)):
        return cfg.get(key, default)
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def render_prompt(template: str, values: dict[str, Any]) -> str:
    if not template:
        return template
    rendered_values = {key: "" if value is None else value for key, value in values.items()}
    try:
        return template.format_map(rendered_values)
    except KeyError as e:
        missing_key = e.args[0]
        raise ValueError(f"Prompt template requires missing field: {missing_key}") from e


def default_run_folder(cfg: DictConfig) -> str:
    value = str(cfg.get("run_folder", "") or "")
    if "${now:" in value or not value:
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    return value


def default_run_dir(cfg: DictConfig) -> Path:
    task_name = str(cfg.get("task_name", "gemini_batch"))
    exp_dir = REPO_ROOT / "exp"
    paths_cfg = cfg.get("paths")
    if paths_cfg is not None:
        root_dir = str(paths_cfg.get("root_dir", str(REPO_ROOT)))
        if root_dir.startswith("${oc.env:"):
            root_dir = os.environ.get("PROJECT_ROOT", str(REPO_ROOT))
        exp_dir_value = str(paths_cfg.get("exp_dir", ""))
        if exp_dir_value and "${" not in exp_dir_value:
            exp_dir = Path(exp_dir_value)
        else:
            exp_dir = Path(root_dir) / "exp"
    return exp_dir / "runs" / task_name / default_run_folder(cfg)


def guess_mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(path))
    if mime_type:
        return mime_type
    if path.suffix.lower() in {".wav", ".wave"}:
        return "audio/wav"
    if path.suffix.lower() == ".flac":
        return "audio/flac"
    if path.suffix.lower() == ".mp3":
        return "audio/mpeg"
    return "application/octet-stream"


def upload_file(
    client: Any,
    path: Path,
    *,
    mime_type: Optional[str] = None,
    display_name: Optional[str] = None,
    anonymize: bool = True,
) -> Any:
    from google.genai import types

    upload_path = path
    temp_file: Optional[Path] = None
    if anonymize:
        suffix = path.suffix or ".bin"
        temp_file = Path(tempfile.gettempdir()) / f"{uuid.uuid4().hex}{suffix}"
        shutil.copy2(path, temp_file)
        upload_path = temp_file
    try:
        return client.files.upload(
            file=str(upload_path),
            config=types.UploadFileConfig(
                mimeType=mime_type or guess_mime_type(path),
                displayName=display_name,
            ),
        )
    finally:
        if temp_file and temp_file.exists():
            temp_file.unlink()


def uploaded_file_record(
    uploaded: Any,
    *,
    original_path: Optional[Path],
    role: str,
    key: Optional[str] = None,
) -> dict[str, Any]:
    return {
        "role": role,
        "key": key,
        "original_path": str(original_path) if original_path else None,
        "name": getattr(uploaded, "name", None),
        "uri": getattr(uploaded, "uri", None),
        "mime_type": getattr(uploaded, "mime_type", None) or getattr(uploaded, "mimeType", None),
        "raw": to_jsonable(uploaded),
    }


def generation_config_from_client_config(client_config: dict[str, Any]) -> dict[str, Any]:
    generation_config: dict[str, Any] = {
        "temperature": client_config.get("temperature", 1.0),
        "topP": client_config.get("top_p", client_config.get("topP", 0.95)),
        "seed": client_config.get("seed", 0),
        "candidateCount": 1,
        "responseModalities": ["TEXT"],
    }
    if client_config.get("thinking_budget") is not None:
        generation_config["thinkingConfig"] = {
            "thinkingBudget": client_config.get("thinking_budget")
        }
    if client_config.get("max_output_tokens") is not None:
        generation_config["maxOutputTokens"] = client_config["max_output_tokens"]
    response_schema = client_config.get("response_schema")
    if response_schema:
        generation_config["responseMimeType"] = "application/json"
        generation_config["responseSchema"] = response_schema
    return {key: value for key, value in generation_config.items() if value is not None}


def build_generate_content_request(
    *,
    user_prompt: str,
    system_prompt: str,
    file_uri: str,
    mime_type: str,
    generation_config: dict[str, Any],
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "fileData": {
                            "fileUri": file_uri,
                            "mimeType": mime_type,
                        }
                    },
                    {"text": user_prompt},
                ],
            }
        ],
        "generationConfig": generation_config,
    }
    if system_prompt:
        request["systemInstruction"] = {
            "parts": [{"text": system_prompt}],
        }
    return request


def sample_batch_key(sample: dict[str, Any], idx: int) -> str:
    for field in ("utt_id", "key", "audio_path", "wavpath"):
        value = sample.get(field)
        if value is not None:
            return str(value)
    return str(idx)


def sample_wav_path(sample: dict[str, Any]) -> Path:
    value = sample.get("wavpath") or sample.get("audio_path")
    if not value:
        raise ValueError(f"Dataset sample is missing wavpath/audio_path: {sample.keys()}")
    value_str = str(value)
    if ":" in value_str and not Path(value_str).exists():
        raise ValueError(
            "Gemini Batch submit cannot upload Kaldi ark slices directly: "
            f"{value_str}"
        )
    return Path(value_str)


def build_batch_artifacts(
    *,
    cfg: DictConfig,
    run_dir: Path,
    client: Optional[Any],
    dry_run: bool,
    anonymize_uploads: bool,
) -> tuple[Path, Path, Path, list[dict[str, Any]]]:
    from src.core.distributed_inference import get_dataset_from_cfg

    runner_cfg = cfg.inference.inference_runner
    prompt_config = runner_cfg.get("prompt_config", {})
    client_config = resolve_config_values(
        node_to_plain_dict(runner_cfg.get("client_config", {})),
        cfg,
    )
    generation_config = generation_config_from_client_config(client_config)
    passthrough_keys = list(cfg.inference.get("passthrough_keys", DEFAULT_PASSTHROUGH_KEYS))
    limit_samples = cfg.inference.get("limit_samples")

    print("Loading dataset for Gemini batch submit...", flush=True)
    dataset = get_dataset_from_cfg(cfg.data)
    total = len(dataset)
    if limit_samples is not None and int(limit_samples) > 0:
        total = min(total, int(limit_samples))
    print(f"Preparing {total} Gemini batch requests.", flush=True)

    requests_path = run_dir / "batch_requests.jsonl"
    manifest_path = run_dir / "batch_manifest.jsonl"
    uploaded_path = run_dir / "uploaded_files.jsonl"
    request_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    uploaded_rows: list[dict[str, Any]] = []

    for idx in range(total):
        sample = dataset[idx]
        key = sample_batch_key(sample, idx)
        wav_path = sample_wav_path(sample)
        print(f"[{idx + 1}/{total}] Preparing {key}: {wav_path}", flush=True)
        prompt_values = {**sample, "wavpath": str(wav_path)}
        user_prompt = render_prompt(str(prompt_config.get("user_prompt", "")), prompt_values)
        system_prompt = render_prompt(str(prompt_config.get("system_prompt", "")), prompt_values)

        if dry_run:
            uploaded_record = {
                "role": "audio",
                "key": key,
                "original_path": str(wav_path),
                "name": f"dry-run/{key}",
                "uri": f"dry-run://{wav_path}",
                "mime_type": guess_mime_type(wav_path),
                "raw": {},
            }
        else:
            if client is None:
                raise ValueError("A Gemini client is required when dry_run=False")
            print(f"[{idx + 1}/{total}] Uploading audio for {key}", flush=True)
            uploaded = upload_file(
                client,
                wav_path,
                display_name=f"audio-{idx}",
                anonymize=anonymize_uploads,
            )
            uploaded_record = uploaded_file_record(
                uploaded,
                original_path=wav_path,
                role="audio",
                key=key,
            )
            print(f"[{idx + 1}/{total}] Uploaded audio for {key}", flush=True)
        uploaded_rows.append(uploaded_record)

        request_rows.append(
            {
                "key": key,
                "request": build_generate_content_request(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    file_uri=str(uploaded_record["uri"]),
                    mime_type=str(uploaded_record["mime_type"]),
                    generation_config=generation_config,
                ),
            }
        )
        manifest_rows.append(
            {
                "key": key,
                "idx": idx,
                "utt_id": sample.get("utt_id") or sample.get("key") or key,
                "audio_path": str(wav_path),
                "uploaded_file": uploaded_record,
                "prompt": {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                },
                "passthrough": {
                    k: to_jsonable(sample[k])
                    for k in passthrough_keys
                    if k in sample
                },
            }
        )

    write_jsonl(requests_path, request_rows)
    write_jsonl(manifest_path, manifest_rows)
    write_jsonl(uploaded_path, uploaded_rows)
    return requests_path, manifest_path, uploaded_path, uploaded_rows


def upload_batch_requests_file(client: Any, requests_path: Path, task_name: str) -> Any:
    from google.genai import types

    return client.files.upload(
        file=str(requests_path),
        config=types.UploadFileConfig(
            mimeType="jsonl",
            displayName=f"{task_name}-batch-requests",
        ),
    )


def convert_results_to_transcription_shard(
    *,
    run_dir: Path,
    manifest_path: Path,
    results_path: Path,
    output_key: Optional[str],
    clean: bool,
) -> Path:
    for stale_path in run_dir.glob("transcription.*.jsonl"):
        if ".error" not in stale_path.name and ".errors" not in stale_path.name:
            stale_path.unlink()

    manifest = load_manifest(manifest_path)
    results = read_jsonl(results_path)
    seen: set[str] = set()
    records: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []

    for row in results:
        key = result_key(row)
        if key is None:
            raise ValueError(f"Batch result row is missing key: {row}")
        if key not in manifest:
            raise ValueError(f"Batch result key {key!r} not found in {manifest_path}")
        seen.add(key)
        item = manifest[key]
        pred = prediction_from_result(row, output_key=output_key, clean=clean)
        records.append(
            {
                item["index"]: {
                    "pred": pred,
                    "passthrough": item["passthrough"],
                }
            }
        )
        if pred and isinstance(pred[0], dict) and pred[0].get("error"):
            error_rows.append(
                {
                    "key": error_log_key(key, item),
                    "audio_path": item.get("audio_path") or "",
                    "error": pred[0]["error"],
                }
            )

    missing = sorted(set(manifest) - seen)
    for key in missing:
        item = manifest[key]
        error = {
            "type": "MissingGeminiBatchResult",
            "message": f"No batch result row found for key {key}",
        }
        pred = empty_error_prediction(error, {"key": key})
        records.append(
            {
                item["index"]: {
                    "pred": pred,
                    "passthrough": item["passthrough"],
                }
            }
        )
        error_rows.append(
            {
                "key": error_log_key(key, item),
                "audio_path": item.get("audio_path") or "",
                "error": error,
            }
        )

    shard_path = run_dir / "transcription.0.jsonl"
    error_path = run_dir / "transcription.errors.jsonl"
    write_jsonl(shard_path, records)
    if error_rows:
        write_jsonl(error_path, error_rows)
    elif error_path.exists():
        error_path.unlink()
    print(f"Wrote {len(records)} records to {shard_path}", flush=True)
    if error_rows:
        print(
            f"Wrote {len(error_rows)} error rows to {error_path}",
            flush=True,
        )
    return shard_path


def get_batch_job_name(job: dict[str, Any]) -> Optional[str]:
    for field in ("name", "job_name", "batch_job_name"):
        value = job.get(field)
        if value:
            return str(value)
    batch_job = job.get("batch_job")
    if isinstance(batch_job, dict):
        return get_batch_job_name(batch_job)
    return None


def download_results_if_needed(run_dir: Path, results_path: Path) -> Path:
    if results_path.exists():
        return results_path

    job_path = run_dir / "batch_job.json"
    if not job_path.exists():
        raise FileNotFoundError(
            f"No {results_path} and no {job_path}; pass --results-file or collect after saving batch_results.jsonl"
        )

    job_name = get_batch_job_name(read_json(job_path))
    if not job_name:
        raise ValueError(f"Could not find job name in {job_path}")

    client = make_client()
    batch_job = client.batches.get(name=job_name)
    state = get_batch_state(batch_job)
    if not state.endswith("JOB_STATE_SUCCEEDED"):
        raise RuntimeError(f"Gemini batch job {job_name} is not succeeded; current state={state}")
    file_name = get_batch_dest_file_name(batch_job)
    if not file_name:
        raise RuntimeError(f"Gemini batch job {job_name} has no result file")
    content = client.files.download(file=file_name)
    results_path.write_bytes(content)
    print(f"Downloaded Gemini batch results to {results_path}", flush=True)
    return results_path


def infer_canonical_file(manifest: dict[str, dict[str, Any]]) -> Optional[Path]:
    for item in manifest.values():
        audio_path = item.get("audio_path")
        if not audio_path:
            continue
        path = Path(str(audio_path))
        if path.parent.name in {"audio", "wavs", "wav"}:
            canonical = path.parent.parent / "text.canonical"
            if canonical.exists():
                return canonical
    return None


def manifest_has_canonical_field(manifest: dict[str, dict[str, Any]]) -> bool:
    return any(
        isinstance(item.get("passthrough"), dict)
        and item["passthrough"].get("canonical_ipa") is not None
        for item in manifest.values()
    )


def run_checked(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def merge_transcription_outputs(run_dir: Path) -> Path:
    """Merge only transcription shards, not Gemini batch artifact JSONL files."""
    from scripts.jsonl2json import merge_jsonl_dir

    shard_paths = sorted(
        path
        for path in run_dir.glob("transcription.*.jsonl")
        if ".error" not in path.name and ".errors" not in path.name
    )
    if not shard_paths:
        raise FileNotFoundError(f"No transcription.*.jsonl shards found in {run_dir}")

    with tempfile.TemporaryDirectory(
        prefix=".gemini_collect_merge_",
        dir=str(run_dir),
    ) as tmp_name:
        tmp_dir = Path(tmp_name)
        for path in shard_paths:
            shutil.copy2(path, tmp_dir / path.name)
        for path in sorted(run_dir.glob("transcription*.error*.jsonl")):
            shutil.copy2(path, tmp_dir / path.name)

        outputs = merge_jsonl_dir(tmp_dir)
        for path in outputs.values():
            shutil.copy2(path, run_dir / path.name)

    pred_path = run_dir / "transcription.json"
    print(f"Merged transcription shards into {pred_path}", flush=True)
    return pred_path


def run_evaluation(
    *,
    run_dir: Path,
    manifest_path: Path,
    evaluation_name: str,
    canonical_file: Optional[Path],
    canonical_field: Optional[str],
    gt_field: str,
    pred_field: str,
    key_field: str,
    language_field: Optional[str],
) -> None:
    result_txt = run_dir / "inventory_results.txt"
    if result_txt.exists() and result_txt.stat().st_size > 0:
        print(f"Skipping evaluation; already found {result_txt}", flush=True)
        return

    pred_path = merge_transcription_outputs(run_dir)
    if not pred_path.exists() or pred_path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing/empty merged prediction file: {pred_path}")

    manifest = load_manifest(manifest_path)
    canonical_args: list[str] = []
    if canonical_file:
        canonical_args = ["--canonical_file", str(canonical_file)]
    else:
        inferred = infer_canonical_file(manifest)
        if inferred:
            canonical_args = ["--canonical_file", str(inferred)]
        elif canonical_field:
            canonical_args = ["--canonical_field", canonical_field]
        elif manifest_has_canonical_field(manifest):
            canonical_args = ["--canonical_field", "canonical_ipa"]
        else:
            raise ValueError(
                "CMU39-projected evaluation requires a canonical IPA file or "
                "canonical_ipa passthrough field."
            )

    command = [
        sys.executable,
        "-m",
        "src.metrics.phone_recognition",
        "--evaluation_name",
        evaluation_name,
        "--prediction_file",
        str(pred_path),
        "--output_file",
        str(run_dir / "inventory_results.csv"),
        "--gt_field",
        gt_field,
        "--pred_field",
        pred_field,
        "--key_field",
        key_field,
    ]
    if language_field:
        command.extend(["--language_field", language_field])
    command.extend(canonical_args)
    run_checked(command)
    print(f"Evaluation complete: {run_dir / 'inventory_results.csv'}", flush=True)


def submit(args: argparse.Namespace) -> None:
    overrides = submit_overrides(
        list(args.overrides or []),
        preset_name=args.preset,
    )
    cfg = compose_hydra_config(overrides)
    runner_cfg = cfg.inference.inference_runner
    client_config = resolve_config_values(
        node_to_plain_dict(runner_cfg.get("client_config", {})),
        cfg,
    )
    model_name = args.model or str(client_config.get("model_name", "gemini-3.1-pro-preview"))
    task_name = str(cfg.get("task_name", "gemini_batch"))
    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else default_run_dir(cfg)

    if (run_dir / "batch_job.json").exists() and not args.force:
        raise FileExistsError(
            f"{run_dir / 'batch_job.json'} already exists. "
            "Use --force only if you intentionally want to create a new batch job."
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        run_dir / "batch_config.json",
        {
            "created_at": datetime.now().isoformat(),
            "dry_run": args.dry_run,
            "model": model_name,
            "preset": args.preset,
            "task_name": task_name,
            "overrides": overrides,
            "config": to_jsonable(cfg),
        },
    )

    client = None if args.dry_run else make_client(resolve_env_api_key(client_config.get("api_key")))
    requests_path, manifest_path, uploaded_path, _ = build_batch_artifacts(
        cfg=cfg,
        run_dir=run_dir,
        client=client,
        dry_run=args.dry_run,
        anonymize_uploads=not args.no_anonymize_uploads,
    )
    manifest_count = len(read_jsonl(manifest_path))
    print(f"Wrote {manifest_count} batch requests to {requests_path}", flush=True)
    print(f"Wrote manifest to {manifest_path}", flush=True)
    print(f"Wrote uploaded file records to {uploaded_path}", flush=True)

    if args.dry_run:
        print(f"Dry run complete. No files uploaded and no batch job submitted. Run dir: {run_dir}", flush=True)
        return

    assert client is not None
    from google.genai import types

    uploaded_requests = upload_batch_requests_file(client, requests_path, task_name)
    request_upload_record = uploaded_file_record(
        uploaded_requests,
        original_path=requests_path,
        role="batch_requests",
    )
    with uploaded_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(request_upload_record, ensure_ascii=False, default=str) + "\n")

    batch_job = client.batches.create(
        model=model_name,
        src=uploaded_requests.name,
        config=types.CreateBatchJobConfig(displayName=args.display_name or task_name),
    )
    batch_job_data = {
        "created_at": datetime.now().isoformat(),
        "name": getattr(batch_job, "name", None),
        "model": model_name,
        "display_name": args.display_name or task_name,
        "run_dir": str(run_dir),
        "request_file": request_upload_record,
        "raw": to_jsonable(batch_job),
    }
    write_json(run_dir / "batch_job.json", batch_job_data)
    write_json(run_dir / "batch_status.json", to_jsonable(batch_job))
    print(f"Submitted Gemini batch job: {batch_job_data['name']}", flush=True)
    print(f"Run dir: {run_dir}", flush=True)


def status(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else None
    job_name = args.job_name
    if not job_name:
        if run_dir is None:
            raise ValueError("Provide --run-dir or --job-name")
        job_path = run_dir / "batch_job.json"
        if not job_path.exists():
            raise FileNotFoundError(f"Missing batch job file: {job_path}")
        job_name = get_batch_job_name(read_json(job_path))
    if not job_name:
        raise ValueError("Could not determine Gemini batch job name")

    client = make_client()
    while True:
        batch_job = client.batches.get(name=job_name)
        state = get_batch_state(batch_job)
        status_data = to_jsonable(batch_job)
        if run_dir is not None:
            write_json(run_dir / "batch_status.json", status_data)

        print(f"Job: {job_name}", flush=True)
        print(f"State: {state}", flush=True)
        dest_file = get_batch_dest_file_name(batch_job)
        if dest_file:
            print(f"Result file: {dest_file}", flush=True)
        error = getattr(batch_job, "error", None)
        if error:
            print(f"Error: {to_jsonable(error)}", flush=True)

        if not args.wait or state in TERMINAL_BATCH_STATES:
            break
        import time

        time.sleep(args.poll_interval)

    if args.download_results:
        if run_dir is None:
            raise ValueError("--download-results requires --run-dir")
        if get_batch_state(batch_job) != "JOB_STATE_SUCCEEDED":
            raise RuntimeError(f"Cannot download results until job succeeds; current state={state}")
        dest_file = get_batch_dest_file_name(batch_job)
        if not dest_file:
            raise RuntimeError(f"Gemini batch job {job_name} has no result file")
        content = client.files.download(file=dest_file)
        results_path = run_dir / "batch_results.jsonl"
        results_path.write_bytes(content)
        print(f"Downloaded Gemini batch results to {results_path}", flush=True)


def collect(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir).expanduser().resolve()
    manifest_path = Path(args.manifest_file).expanduser().resolve() if args.manifest_file else run_dir / "batch_manifest.jsonl"
    results_path = Path(args.results_file).expanduser().resolve() if args.results_file else run_dir / "batch_results.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing batch manifest: {manifest_path}")
    results_path = download_results_if_needed(run_dir, results_path)

    convert_results_to_transcription_shard(
        run_dir=run_dir,
        manifest_path=manifest_path,
        results_path=results_path,
        output_key=args.output_key,
        clean=not args.no_clean_response,
    )

    if not args.no_evaluate:
        evaluation_name = args.evaluation_name or collect_default_evaluation_name(run_dir)
        canonical_file = (
            Path(args.canonical_file).expanduser().resolve()
            if args.canonical_file
            else collect_default_canonical_file(run_dir)
        )
        run_evaluation(
            run_dir=run_dir,
            manifest_path=manifest_path,
            evaluation_name=evaluation_name,
            canonical_file=canonical_file,
            canonical_field=args.canonical_field,
            gt_field=args.gt_field,
            pred_field=args.pred_field,
            key_field=args.key_field,
            language_field=args.language_field,
        )


def add_collect_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", required=True, help="Hydra run dir containing batch artifacts")
    parser.add_argument("--manifest-file", help="Defaults to <run-dir>/batch_manifest.jsonl")
    parser.add_argument("--results-file", help="Defaults to <run-dir>/batch_results.jsonl; downloads if absent")
    parser.add_argument("--output-key", default="transcription", help="JSON field to extract from model text")
    parser.add_argument("--no-clean-response", action="store_true", help="Do not apply Gemini IPA cleaning")
    parser.add_argument("--no-evaluate", action="store_true", help="Only collect/convert; skip merge and scoring")
    parser.add_argument("--evaluation-name", help="Defaults to parent task directory name")
    parser.add_argument("--canonical-file", help="Overrides automatic text.canonical inference")
    parser.add_argument("--canonical-field", help="Use passthrough field for canonical IPA if no canonical file")
    parser.add_argument("--gt-field", default="target")
    parser.add_argument("--pred-field", default="processed_transcript")
    parser.add_argument("--key-field", default="utt_id")
    parser.add_argument("--language-field", default="lang_sym")


def add_submit_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--preset",
        choices=sorted(SUBMIT_PRESETS),
        default=DEFAULT_SUBMIT_PRESET,
        help=(
            "Default submit preset. Available presets set dataset name, "
            "dataset root, task prefix, and timestamped run folder. Explicit "
            "Hydra overrides still win."
        ),
    )
    parser.add_argument(
        "--run-dir",
        help="Output run dir. Defaults to exp/runs/<task_name>/<run_folder> from the composed Hydra config.",
    )
    parser.add_argument(
        "--model",
        help="Override Gemini model name. Defaults to inference.inference_runner.client_config.model_name.",
    )
    parser.add_argument("--display-name", help="Gemini batch display name")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write batch artifacts without uploading files or creating a Gemini batch job.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow submit to create a new job even if <run-dir>/batch_job.json already exists.",
    )
    parser.add_argument(
        "--no-anonymize-uploads",
        action="store_true",
        help="Upload original audio filenames instead of temporary anonymized copies.",
    )
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help=(
            "Optional Hydra overrides. Defaults come from --preset; the default "
            "preset is authentic_kids_kaldi."
        ),
    )


def add_status_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", help="Run dir containing batch_job.json")
    parser.add_argument("--job-name", help="Explicit Gemini batch job name, e.g. batches/123")
    parser.add_argument("--wait", action="store_true", help="Poll until terminal state")
    parser.add_argument("--poll-interval", type=float, default=30.0)
    parser.add_argument(
        "--download-results",
        action="store_true",
        help="If the job succeeded, download results to <run-dir>/batch_results.jsonl.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit_parser = subparsers.add_parser(
        "submit",
        help="Build Gemini batch artifacts, upload them, and submit a batch job",
    )
    add_submit_args(submit_parser)
    submit_parser.set_defaults(func=submit)

    status_parser = subparsers.add_parser(
        "status",
        help="Check or wait for a Gemini batch job",
    )
    add_status_args(status_parser)
    status_parser.set_defaults(func=status)

    collect_parser = subparsers.add_parser(
        "collect",
        help="Download/convert Gemini batch results and run PhonBench evaluation",
    )
    add_collect_args(collect_parser)
    collect_parser.set_defaults(func=collect)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
