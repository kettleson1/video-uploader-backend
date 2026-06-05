# Dave Golden Dataset

This folder is the starter NFHS/high-school golden dataset for Dave (Digital Artificial Video Review).

## Purpose

Use these same high-school clips after every prompt, rule, backend, or model change to check whether Dave is improving or regressing under NFHS rules.

## Structure

```text
golden-dataset/
  videos/
  labels.csv
  candidate_sources.csv
  candidate_review.html
```

Put the actual `.mov` or `.mp4` files in `videos/` using the exact filenames in `labels.csv`.

`candidate_sources.csv` and `candidate_review.html` are working files for human review. They list NFHS/high-school-oriented source videos that may contain useful plays, but those rows are not approved golden clips until a human selects the exact play segment and confirms the NFHS ruling.

## Current Dataset Status

The starter golden dataset now contains 27 approved local clips:

- 16 foul clips
- 11 no-foul clips
- Defensive and offensive pass interference examples
- Holding and block-in-the-back examples
- Formation/motion/shift examples
- Personal-foul examples
- Kick-play examples

Include both obvious and borderline plays. Borderline plays are especially valuable because they expose whether the model understands NFHS rule details instead of only obvious contact.

## Label Fields

`labels.csv` uses one row per video:

```csv
id,filename,expected_label,expected_result,rule_reference,notes
```

- `id`: stable dataset id
- `filename`: video filename under `golden-dataset/videos/`
- `expected_label`: correct app label, or `None` for no foul
- `expected_result`: `FOUL` or `NO FOUL`
- `rule_reference`: supporting NFHS rule reference when applicable
- `notes`: human reason for the ruling and common model confusion to avoid

## Before Running Evals

1. Confirm the local video files are present in `golden-dataset/videos/`.
2. Keep filenames exactly aligned with `labels.csv`.
3. Keep the `id` stable once a clip is part of the dataset.
4. Score `FOUL` / `NO FOUL` separately from the specific foul label so false positives are easy to spot.
