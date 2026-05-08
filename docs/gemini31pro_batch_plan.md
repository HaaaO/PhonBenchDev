# Gemini 3.1 Pro Preview Batch Inference Plan

## Summary

Add `gemini-3.1-pro-preview` to PhonBenchDev as a batch-only inference backend.
This avoids the low normal API request limit during full evaluations while keeping
the downstream PhonBench scoring pipeline unchanged.

The Gemini Batch API is asynchronous. The machine only needs to stay live until
the batch job is submitted and a job name is saved. After that, the job runs on
Google's side and can be checked or collected later from the saved run directory.

## Goals

- Run full PhonBenchDev IPA transcription evaluations with
  `gemini-3.1-pro-preview`.
- Use the same prompts, cleaning, passthrough metadata, and MDD scoring as the
  existing Gemini synchronous runner.
- Produce the same `transcription.*.jsonl` output shape expected by the current
  merge and evaluation scripts.
- Keep batch latency separate from accuracy metrics because batch inference is
  not an interactive serving mode.

## Implementation Plan

Current implementation status:

- `scripts/gemini_batch.py submit` is implemented.
- `scripts/gemini_batch.py status` is implemented.
- `scripts/gemini_batch.py collect` is implemented and runs evaluation by
  default.
- `configs/experiment/inference/transcribe_gemini31pro_batch.yaml` provides
  the default `gemini-3.1-pro-preview` batch config.

### 1. Add a batch workflow beside the existing Gemini runner

Do not force batch jobs through `src/core/distributed_inference.py`, because that
loop calls one inference object per utterance. Batch inference needs one global
submit step followed by a later collection step.

Add a script such as:

```bash
scripts/gemini_batch.py
```

with three subcommands:

```bash
python scripts/gemini_batch.py submit ...
python scripts/gemini_batch.py status --run-dir <run_dir>
python scripts/gemini_batch.py collect --run-dir <run_dir>
```

### 2. Add a Gemini 3.1 Pro batch config

Add a config such as:

```bash
configs/experiment/inference/transcribe_gemini31pro_batch.yaml
```

Use these defaults:

- `model_name: gemini-3.1-pro-preview`
- `api_key: ${oc.env:GEMINI_API_KEY}`
- same prompt config as the current Gemini IPA transcription run
- same `passthrough_keys`
- same `clean_response` behavior
- same `output_key` behavior if structured JSON output is used

### 3. Submit step

The submit step should:

1. Instantiate the configured dataset.
2. Apply `inference.limit_samples` if provided.
3. Render the same user/system prompt for each utterance.
4. Upload each audio file or otherwise create a Gemini file reference.
5. Write a batch JSONL request file.
6. Upload the JSONL file with the Gemini File API.
7. Create a Gemini Batch API job.
8. Save the returned batch job name and all local metadata needed to collect
   results later.

Each JSONL request should use a stable key, preferably `utt_id`. If `utt_id` is
missing, use the dataset index as a fallback. The key must be saved in the local
manifest so returned results can be mapped back to the original dataset row.

Expected run artifacts:

```bash
batch_requests.jsonl
batch_manifest.jsonl
uploaded_files.jsonl
batch_job.json
```

The manifest should include at least:

- batch key
- dataset index
- `utt_id`
- audio path
- rendered prompt metadata needed for debugging
- passthrough fields used by the evaluator

### 4. Status step

The status step should:

1. Read `batch_job.json`.
2. Call `client.batches.get(name=<job_name>)`.
3. Print the job state and any available error details.
4. Exit without blocking by default.

Optionally support a `--wait` flag later, but polling should not be required.

Terminal states to handle:

- `JOB_STATE_SUCCEEDED`
- `JOB_STATE_FAILED`
- `JOB_STATE_CANCELLED`
- `JOB_STATE_EXPIRED`

### 5. Collect step

The collect step should:

1. Read `batch_job.json` and `batch_manifest.jsonl`.
2. Check that the job state is `JOB_STATE_SUCCEEDED`.
3. Download the batch result file.
4. Save it as:

```bash
batch_results.jsonl
```

5. Parse each result by batch key.
6. Convert successful responses into the existing PhonBench prediction shape:

```json
{
  "processed_transcript": "...",
  "predicted_transcript": "...",
  "raw_model_response": "..."
}
```

7. Convert failed per-request responses into empty predictions with an `error`
   object.
8. Write PhonBench-compatible output:

```bash
transcription.0.jsonl
```

Each line should match the current distributed inference format:

```json
{
  "123": {
    "pred": [
      {
        "processed_transcript": "...",
        "predicted_transcript": "...",
        "raw_model_response": "..."
      }
    ],
    "passthrough": {
      "target": "...",
      "split": "...",
      "utt_id": "...",
      "metadata_idx": 123,
      "lang_sym": "...",
      "canonical_ipa": "..."
    }
  }
}
```

After collection, `collect` should automatically run the existing merge and
scoring pipeline:

```bash
python scripts/jsonl2json.py --dirname <run_dir>
python -m src.metrics.phone_recognition ...
```

In implementation, the merge step must only merge `transcription.*.jsonl`
prediction shards. The run directory also contains `batch_manifest.jsonl` and
`batch_results.jsonl`, so blindly merging every `*jsonl` file in the directory
would corrupt `transcription.json`.

Expected evaluation artifacts after `collect`:

```bash
transcription_raw.json
transcription_normalized.json
transcription.json
normalization_report.csv
inventory_results.csv
inventory_results.txt
mdd_per_utt.csv
mdd_joint_per_utt.csv
```

## Runtime And Error Checking

The batch job is asynchronous. For the 148 short `authentic_kids_kaldi` audio
files, a reasonable expectation is minutes to tens of minutes, but the exact
runtime depends on Google-side queueing, model availability, quota, and file
processing. The compute node only needs to stay alive until `submit` finishes
and writes `batch_job.json`; after that, the job continues on Google's side.

Check progress without blocking:

```bash
python scripts/gemini_batch.py status --run-dir <run_dir>
```

Or poll until a terminal state:

```bash
python scripts/gemini_batch.py status \
  --run-dir <run_dir> \
  --wait \
  --poll-interval 60
```

Terminal states:

```text
JOB_STATE_SUCCEEDED
JOB_STATE_FAILED
JOB_STATE_CANCELLED
JOB_STATE_EXPIRED
```

You cannot know that there were no request-level failures until results are
collected and parsed. `collect` refuses to continue if the batch job has not
succeeded. If the batch succeeded but individual requests failed, were missing,
or returned empty text, the collector writes empty predictions with error
objects and records them in:

```bash
<run_dir>/transcription.errors.jsonl
```

After collection, check for per-item errors:

```bash
ls -lh <run_dir>
test ! -f <run_dir>/transcription.errors.jsonl || wc -l <run_dir>/transcription.errors.jsonl
test ! -f <run_dir>/transcription.errors.jsonl || sed -n '1,5p' <run_dir>/transcription.errors.jsonl
```

No `transcription.errors.jsonl` means the collector did not record per-item
failures. Also confirm evaluation artifacts exist:

```bash
ls \
  <run_dir>/inventory_results.txt \
  <run_dir>/mdd_per_utt.csv \
  <run_dir>/mdd_joint_per_utt.csv
```

## Fairness Notes

- Use exactly the same dataset split as other model runs.
- Use the same IPA prompt and response cleaning.
- Use the same canonical/uttered/predicted evaluation code.
- Report strict and joint 3-way DP metrics exactly as for other models.
- Do not compare batch wall-clock latency directly against realtime or sync
  model latency.
- Track failed or empty Gemini responses explicitly so they are auditable rather
  than silently dropped.

## Cleanup Policy

Do not delete uploaded Gemini files immediately after submission. The batch job
may still need those file references while it runs.

Add cleanup only as an explicit step or flag after successful collection, for
example:

```bash
python scripts/gemini_batch.py cleanup --run-dir <run_dir>
```

The cleanup step can delete uploaded audio files and uploaded request JSONL files
recorded in `uploaded_files.jsonl`.

## Test Plan

### Unit tests

- Build requests from fake dataset rows.
- Verify prompt rendering uses the same fields as the synchronous Gemini runner.
- Verify stable key mapping with `utt_id` and dataset-index fallback.
- Parse fake successful batch results.
- Parse fake failed per-request batch results.
- Convert parsed results into valid `transcription.0.jsonl` lines.
- Refuse to resubmit if `batch_job.json` already exists unless `--force` is
  passed.

### Live smoke test

Run a tiny batch with 2-5 audio files before a full evaluation.

Check:

- job submission succeeds
- status can be queried after reconnecting
- results can be downloaded
- IPA text is extracted correctly
- existing `jsonl2json.py` and MDD scoring scripts accept the collected output

## Example Run Flow

Submit:

```bash
python scripts/gemini_batch.py submit
```

By default, `submit` uses `experiment=inference/transcribe_gemini31pro_batch`,
`data=powsmeval`, `data.dataset_name=authentic_kids_kaldi`,
`data.data_dir=/n/iqss_sponsored/Lab/zshi/prism-evalsets`, and
`data.portable_wavscp=True`. It also creates a timestamped task name of the
form `inf_authentic_kids_kaldi_gemini31pro_batch_<YYYYMMDD_HHMMSS>`. Hydra
overrides can still be appended for smoke tests or alternate datasets. The
script loads `/n/iqss_sponsored/Lab/zshi/PhonBenchDev/.env` by default for
`GEMINI_API_KEY`, without overriding a key that is already exported in the
shell. Example:

```bash
python scripts/gemini_batch.py submit --dry-run inference.limit_samples=2
```

Check later:

```bash
python scripts/gemini_batch.py status --run-dir <run_dir>
```

Collect after success:

```bash
python scripts/gemini_batch.py collect --run-dir <run_dir>
```

This writes the collected PhonBench shard and runs evaluation by default. Use
`--no-evaluate` only when you want to inspect or repair the collected
`transcription.0.jsonl` before scoring.

## References

- Gemini 3.1 Pro Preview model card:
  https://ai.google.dev/gemini-api/docs/models/gemini-3.1-pro-preview
- Gemini Batch API:
  https://ai.google.dev/gemini-api/docs/batch-api
