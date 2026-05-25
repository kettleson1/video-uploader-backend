# Codex Work Log

## NFHS Golden Dataset

Codex helped create the starter NFHS/high-school golden dataset for DAVE evaluation.

The dataset lives in `golden-dataset/` and is intended to support repeatable regression checks after backend, prompt, model, or rule changes.

Tracked files:

- `golden-dataset/labels.csv`
- `golden-dataset/candidate_sources.csv`
- `golden-dataset/candidate_review.html`
- `golden-dataset/README.md`
- `golden-dataset/videos/.gitkeep`

Local-only files:

- `golden-dataset/videos/*.mov`
- `golden-dataset/videos/*.mp4`

Approved clips so far:

- `002`: Obvious defensive pass interference, DPI category `early_contact`, expected `FOUL`.
- `003`: Borderline defensive pass interference, DPI category `arm_bar/body_restrict`, expected `FOUL`.
- `004`: No pass-interference foul, expected `NO FOUL`.

For MVP 1, the eval should compare the main result and label only. DPI subcategories are captured in notes for future subtype scoring.
