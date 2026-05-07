# Three-Way DP Alignment for MDD Evaluation

## Motivation

In mispronunciation detection and diagnosis (MDD), we are not only asking
whether two phone sequences are equal. We are comparing three related sequences:

```text
canonical phone: what should have been said
uttered phone:   what the speaker actually said, based on human annotation
predicted phone: what the model recognized
```

For diagnosis metrics such as `CD` and `DE`, the model should be credited when
its predicted error matches the speaker's actual error. This requires the
uttered and predicted phones to be compared at the same human-intended error
location.

## Alignment Methods

The original strict metric aligned the sequences separately:

```text
canonical -> uttered
canonical -> predicted
then compare slot-by-slot
```

The literature-style hierarchical method is also mostly pairwise:

```text
canonical vs annotated
canonical vs recognized
annotated vs recognized
```

The problem is that each pairwise alignment can choose a different locally valid
gap placement. With insertions and deletions, multiple alignments can have the
same edit distance, but only some are linguistically sensible. For example, in
the `penguin` case, the model's extra `ɹ` can shift later phones and cause a
correctly recognized child error to be counted as wrong.

The current strict metric now follows a published pairwise-plus-correction
style instead:

```text
uttered -> predicted
prompted -> shared uttered/predicted row grid
then compute prompted/uttered and prompted/predicted correctness vectors
```

This is still pairwise/constrained because the uttered/predicted alignment is
chosen first, but it avoids the worst independent-slot mismatch by forcing both
correctness vectors onto one shared row grid.

## Joint Three-Way Alignment

The joint DP method aligns all three sequences at once:

```text
canonical | uttered | predicted
```

This creates one shared alignment grid. The predicted sequence can use a
predicted-only insertion row instead of being forced onto the wrong canonical
slot.

Conceptually, the DP state is:

```text
dp[i, j, k]
```

meaning the best alignment after consuming:

```text
i canonical phones
j uttered phones
k predicted phones
```

At each step, the algorithm emits one aligned row by consuming any non-empty
combination of the three sequences. There are seven possible row types:

```text
C U H
C U -
C - H
- U H
C - -
- U -
- - H
```

where `C` is canonical, `U` is uttered, `H` is hypothesis/predicted, and `-` is
a gap.

Example aligned rows:

```text
canonical | uttered | predicted
p         | p       | p
ɡ         | m       | m
w         | w       | w
```

## MDD Row Labels

Each aligned triplet is converted into an MDD label:

```text
TA: child correct, model says correct
FR: child correct, model says error
FA: child error, model says correct
CD: child error, model catches the same error
DE: child error, model catches an error but diagnoses the wrong phone
```

`TR` is then split into:

```text
TR = CD + DE
```

## Scoring Objective

The DP objective is ordered lexicographically, not treated as one vague scalar
score. In the current implementation, the alignment roughly follows this
priority order:

1. Preserve the human truth alignment by minimizing canonical-vs-uttered edit
   cost.
2. Among equally good human-truth alignments, prefer rows where uttered and
   predicted agree on a real, non-gap phone.
3. Preserve exact canonical-vs-predicted phone matches, so a model output that
   matches the canonical phone is not split into a predicted insertion plus a
   separate canonical deletion.
4. Prefer alignments that produce more sensible `TA` and `CD` rows.
5. Minimize canonical-vs-predicted edit cost.
6. Prefer compact alignments with fewer unnecessary rows and gap-only effects.

This ordering matters. The prediction is not allowed to distort the
canonical-vs-uttered truth alignment. It only breaks ties when the child
alignment is already equally good.

## Why It Works Better

The main reason the joint method works better is that it treats alignment as
part of the diagnostic decision, not as three independent preprocessing steps.

This is especially helpful for:

- insertions
- deletions
- nearby substitutions
- repeated phones in the same word
- model-only extra phones
- cases where several edit alignments have equal cost

The method is therefore more robust to alignment-induced false `DE`, `FA`, or
`FR` counts.

## Relation to Prior Literature

The general algorithm is not new. A three-sequence DP alignment is a classic
extension of Needleman-Wunsch and multiple sequence alignment. Prior literature
calls it three-way alignment, exact 3-way alignment, or 3D dynamic programming.

Relevant prior work:

- Gotoh, 1986, "Alignment of three biological sequences with an efficient
  traceback procedure": describes DP for aligning three sequences with
  `O(L^3)` forward computation.
  Source: https://doi.org/10.1016/S0022-5193(86)80112-6
- Kruspe and Stadler, 2007, "Progressive multiple sequence alignments from
  triplets": states that Needleman-Wunsch extends naturally to cubic time and
  space for three sequences and discusses exact DP for triples.
  Source: https://link.springer.com/article/10.1186/1471-2105-8-254
- Colbourn and Kumar, 2007, "Lower bounds on multiple sequence alignment using
  exact 3-way alignment": uses exact 2-way and 3-way alignments and discusses
  how independent pairwise alignments can conflict with the best 3-way
  alignment.
  Source: https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-8-140

However, I have not found the exact same method proposed in the MDD papers as an
evaluation metric for:

```text
canonical / human-annotated / model-predicted phones
```

Those MDD papers appear to use hierarchical or pairwise comparisons, while this
method produces one shared 3-way alignment. This is also consistent with recent
MDD criticism that dictation and alignment are often modeled independently.
For example, PeppaNet notes that conventional dictation-based MDD methods mostly
make the dictation and alignment processes independent.
Source: https://colab.ws/articles/10.1109%2Fslt54892.2023.10022472

## Suggested Framing

A defensible way to describe the method is:

```text
We adapt exact three-way dynamic-programming sequence alignment from the
multiple-sequence-alignment literature to MDD evaluation. Unlike prior MDD
evaluation protocols that rely on pairwise alignments among canonical,
annotated, and recognized phone sequences, our method produces a single shared
alignment and then computes TA/TR/FA/FR/CD/DE from aligned triplets.
```

The important claim is that the DP algorithm itself is not new, but the
MDD-specific alignment objective and its use as a diagnostic scoring metric may
be novel or at least less standard.

## Reporting Recommendation

Keep the original strict and paper-style metrics for comparability, but report
the joint metric as an improved joint-alignment diagnostic metric. This makes it
possible to show both historical comparability and robustness against
alignment-induced scoring errors.
