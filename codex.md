# Codex Work Log

## NFHS Golden Dataset

Codex helped create the starter NFHS/high-school golden dataset for DAVE evaluation.

The dataset lives in `golden-dataset/` and is intended to support repeatable regression checks after backend, prompt, model, or rule changes.

Current status:

- 27 local video clips have been created.
- `golden-dataset/labels.csv` contains 27 approved label rows.
- `golden-dataset/candidate_sources.csv` marks all 27 rows as `approved`.
- The approved set includes 16 foul clips and 11 no-foul clips.
- The local video files are intentionally not committed to git because they are large media files.

Tracked files:

- `golden-dataset/labels.csv`
- `golden-dataset/candidate_sources.csv`
- `golden-dataset/candidate_review.html`
- `golden-dataset/README.md`
- `golden-dataset/videos/.gitkeep`
- `golden-dataset/reports/.gitkeep`
- `eval_golden_dataset.py`

Local-only files:

- `golden-dataset/videos/*.mov`
- `golden-dataset/videos/*.mp4`
- `golden-dataset/reports/*.csv`

Approved coverage:

- Defensive pass interference
- Offensive pass interference
- Holding
- Block in the back
- Illegal formation
- Personal fouls including illegal blindside block, targeting, and horse-collar tackle
- Kick plays including roughing the kicker and free kick out of bounds
- No-foul controls for pass interference, holding, block in the back, illegal motion/shift, and kick-catch interference

For MVP 1, the eval should compare the main result and label only. DPI subcategories are captured in notes for future subtype scoring.

Eval runner:

- `eval_golden_dataset.py` validates the local video files, uploads the clips one at a time, waits for backend processing, compares predictions to `golden-dataset/labels.csv`, prints misses, and writes a CSV report under `golden-dataset/reports/`.
