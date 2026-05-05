# IPA Normalization Plan for English Phonetic Error Detection

This document describes the proposed IPA normalization policy for PhonBench
evaluation. The current assumption is that all evaluation datasets are English.
Under that assumption, the default normalization profile should be
`ipa_eng_broad_v1`.

The goal is not to hide true phonetic errors. The goal is to avoid counting
model- or tokenizer-specific IPA spelling conventions as recognition errors.
For phonetic error detection, the evaluation should focus on whether a model
captures the child's realized pronunciation relative to the expected canonical
pronunciation, not whether it chooses one equivalent IPA notation over another.

## Output Files

The merge step from per-worker `transcription.*.jsonl` files should produce both
raw and normalized outputs:

- `transcription_raw.json`: exact model outputs, unchanged.
- `transcription_normalized.json`: outputs after IPA normalization.
- `transcription.json`: compatibility alias for `transcription_normalized.json`.
- `normalization_report.csv`: audit file recording every changed field.

`transcription_raw.json` remains the source of truth for auditing model behavior.
`transcription_normalized.json` should be the default input for evaluation
metrics.

In the normalized JSON, raw values should be preserved alongside normalized
values. For example:

- `pred[0].processed_transcript_raw`
- `pred[0].predicted_transcript_raw`
- `passthrough.target_raw`
- `passthrough.canonical_ipa_raw`
- `normalization_profile: ipa_eng_broad_v1`

The fields used by scoring should be normalized:

- `pred[0].processed_transcript`
- `pred[0].predicted_transcript`, if present and IPA-like
- `passthrough.target`
- `passthrough.canonical_ipa`, if present

It is important to normalize both predictions and references. If only model
predictions are normalized, the evaluation can still count reference notation
differences as errors.

## Default Profile

Because the evaluation datasets are English, the default profile should be:

```text
ipa_eng_broad_v1 = strict IPA normalization + English broad normalization
```

Raw and normalized metrics should both be available. Normalized metrics should
be treated as the main evaluation numbers, while raw metrics should be retained
for transparency.

## Strict Normalization

Strict normalization removes notation differences that do not represent
meaningful phonetic error distinctions for this benchmark.

### `t ʃ -> t͡ʃ`

English /t͡ʃ/ is often written either as a tied affricate `t͡ʃ` or as a stop plus
fricative sequence `t ʃ`. Different IPA tokenizers and model vocabularies make
different choices here. For this benchmark, these should be treated as the same
phone when they correspond to the English affricate in words such as
`teacher`, `picture`, and `children`.

Counting `t ʃ` as two phones while the reference uses `t͡ʃ` creates artificial
insertions and substitutions. That inflates PER and can also distort phonetic
error detection: a model can correctly identify the child's /t͡ʃ/ realization
but still be penalized for using an untied spelling.

### `d ʒ -> d͡ʒ`

The same logic applies to the voiced English affricate /d͡ʒ/. It may be emitted
as `d͡ʒ` or as `d ʒ`, depending on the model vocabulary. In English evaluation,
these should be treated as the same affricate when they occur as a single
English phone.

This prevents the scorer from confusing an IPA representation difference with a
phonetic recognition error.

### `pʰ/tʰ/kʰ -> p/t/k`

English voiceless stops are commonly aspirated in stressed syllable onsets.
Some models output narrow allophonic transcriptions such as `pʰ`, `tʰ`, or
`kʰ`, while references in this benchmark are broad phone-level transcriptions
using `p`, `t`, and `k`.

For phonetic error detection, aspiration is usually not the target error being
measured. If the child says a recognizable /t/ in `tooth` and a model writes
`tʰ`, that should not be counted as a substitution. The relevant error is
whether the target stop was realized as the correct English phone, not whether
the model encoded predictable aspiration.

This rule should only remove the aspiration marker. It should not collapse
different stop places or voicing contrasts.

### Long vowels like `iː/uː/oː/ɑː -> i/u/o/ɑ`

Some systems output IPA length marks, especially speech-language models. The
English reference transcriptions in these eval sets generally use a broad
phone-like representation without length marks. For example, one model may
write `iː` where the reference has `i`.

Length marks should be removed for this English benchmark because vowel length
is not being evaluated as an independent phonemic contrast in the current
labels. Keeping the length mark causes a correct vowel category to be counted
as a phone substitution.

This rule should not merge different vowel qualities. For example, `i` and `ɪ`
should remain distinct, and `u` and `ʊ` should remain distinct.

### `r -> ɹ`

English rhotic consonants in these datasets are represented as the alveolar
approximant `ɹ`. Many models, however, use the more generic IPA symbol `r`.
In a broad English phone-recognition setting, `r` from a model is almost always
intended to mean English /ɹ/, not a trilled [r].

Normalizing `r` to `ɹ` avoids penalizing models for using a common simplified
IPA convention. This is especially important for words like `carrot`, `paper`,
`teacher`, and sentence-read prompts with rhotics.

This rule is justified by the English-only assumption. It would not be safe as
a universal multilingual rule because `r` and `ɹ` can be contrastive or
language-specific in other datasets.

### `ɫ -> l`

English /l/ has light and dark allophones. Some models output the velarized
symbol `ɫ`, especially in coda-like positions, while the benchmark references
use broad `l`.

For this task, `ɫ` should be normalized to `l` because the benchmark is not
designed to evaluate allophonic light/dark-l differences. Penalizing `ɫ` as a
substitution for `l` would count a narrow allophonic detail as a phonetic error.

### Dentalized symbols like `t̪/s̪/l̪ -> t/s/l`

Some models output dental diacritics, such as `t̪`, `s̪`, or `l̪`. These encode
fine-grained place detail that is not represented in the broad English target
labels.

For English phonetic error detection in this benchmark, these should be mapped
to their base consonants. The key question is whether the child produced the
intended broad English phone, not whether the model marked a subtle dentalized
variant.

This should be limited to diacritic removal on the same base phone. It should
not merge unrelated phones.

### `g -> ɡ`

Unicode contains the Latin letter `g` and the IPA symbol `ɡ`. They are visually
similar but distinct codepoints. PanPhon and other IPA tools generally expect
the IPA `ɡ`.

This is a pure Unicode normalization issue, not a phonetic decision. Mapping
`g` to `ɡ` prevents toolchain artifacts from becoming evaluation errors.

### `ɚ -> ə˞`, `ɝ -> ɜ˞`

Some IPA tools, including PanPhon in this codebase, handle decomposed rhotic
vowels more reliably than the precomposed symbols `ɚ` and `ɝ`. The forms
`ə˞` and `ɜ˞` preserve the same phonetic content while making segmentation and
feature lookup more robust.

This should be treated as representation normalization. It does not change the
intended rhotic vowel category.

## English Broad Normalization

English broad normalization is more language-specific than strict
normalization. It is appropriate here because all current evaluation datasets
are English and the target labels use broad English IPA categories.

### `t ɕ`, `t ɕʰ`, `ʈ ʂʰ -> t͡ʃ`

Some models, especially multilingual CTC models, output alveolo-palatal or
retroflex affricate-like symbols for English `ch`. Examples include `t ɕ`,
`t ɕʰ`, and `ʈ ʂʰ`. In English child-speech datasets, these outputs often
correspond to the model's vocabulary choice for the English affricate /t͡ʃ/,
not to a real intended contrast in the label space.

For example, if the target or child realization is `t͡ʃ` in `teacher` and a
model emits `t ɕʰ`, the model has likely identified the affricate region
correctly but used a non-English or narrower symbol inventory. Counting this as
multiple errors exaggerates the model's phonetic error.

Because the benchmark is English-only, mapping these affricate-like outputs to
`t͡ʃ` is reasonable for evaluation. This would not be safe in multilingual
evaluation, where `ɕ`, `ʂ`, and retroflex affricates may be contrastive.

### `ɕ`, `ʂ -> ʃ`

The symbols `ɕ` and `ʂ` are not broad English phoneme categories in the current
reference inventory. When English-only models or multilingual phone recognizers
emit these for English speech, they are often approximating the English
postalveolar fricative /ʃ/.

Normalizing them to `ʃ` reduces penalties caused by model vocabulary mismatch.
This is especially relevant for models trained with multilingual IPA
inventories, where English-like sibilants may be distributed over several
nearby IPA symbols.

This rule should be documented as English-specific. It should not be applied to
datasets where `ɕ`, `ʂ`, and `ʃ` are separate target categories.

### `d ʑ`, `ɖ ʐ -> d͡ʒ` if they appear

The voiced counterpart follows the same reasoning. If an English evaluation
item contains /d͡ʒ/ and a model emits `d ʑ` or `ɖ ʐ`, this is likely a
multilingual-vocabulary rendering of an English voiced affricate rather than a
separate English contrast.

These mappings should be included if such symbols appear in model outputs. They
should also be tracked in `normalization_report.csv` so we can verify that the
rule is actually being used in plausible English contexts.

## Contrasts That Should Not Be Normalized

The following distinctions should remain errors because they represent real
English phone contrasts or clinically relevant production differences:

- `i` vs `ɪ`
- `u` vs `ʊ`
- `ɔ` vs `ɑ`
- `f` vs `θ`
- `d` vs `ð`
- missing or extra `ɹ`
- stop place differences such as `p` vs `t` vs `k`
- voicing differences such as `t` vs `d`

These contrasts can change the identity of the English phone. Collapsing them
would hide real phonetic recognition errors and would make phonetic error
detection less useful.

## Recommended Evaluation Workflow

1. Run model inference as usual.
2. Merge per-worker outputs into `transcription_raw.json`.
3. Apply `ipa_eng_broad_v1` during the merge step to create
   `transcription_normalized.json`.
4. Write `transcription.json` as a compatibility alias of
   `transcription_normalized.json`.
5. Write `normalization_report.csv`.
6. Score normalized outputs by default.
7. Optionally score raw outputs for audit and ablation.

The normalized metrics should be used as the primary benchmark results because
they better reflect English phone recognition performance across heterogeneous
model vocabularies. The raw metrics should be retained to quantify how much of
the apparent error comes from IPA symbol mismatch.

## Audit Requirements

Every normalization pass should be auditable. The report should include:

- `utt_id`
- `field`
- `raw`
- `normalized`
- `profile`
- `changed`
- optionally, `rule_ids`

This makes it possible to inspect cases where normalization changes an apparent
error into a match. It also helps detect over-normalization.

## Summary

For English-only phonetic error detection, `ipa_eng_broad_v1` is appropriate as
the default because it removes predictable model-vocabulary and notation
differences while preserving real English phonemic contrasts. The key principle
is to normalize symbol choices that reflect the same English broad phone, while
preserving distinctions that matter for detecting child speech production
errors.
