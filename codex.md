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

## Local Setup Progress - June 7, 2026

Current branch:

- `codex/golden-eval-runner`

Completed locally:

- Pulled the latest branch from GitHub.
- Cleared a stale `.git/HEAD.lock` that blocked Git updates.
- Added the actual 27 golden video files under `golden-dataset/videos/`.
- Confirmed `python3 eval_golden_dataset.py --validate-only` passes for all 27 clips.
- Created a fresh `.venv` because the old `venv/` folder was incomplete.
- Installed `requirements.txt`; `asyncpg` import now works in `.venv`.
- Started the backend successfully with `python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000`.
- Added `DAVE_API_KEY` to local environment configuration.
- Confirmed the eval request can authenticate and reach `/api/upload`.
- Confirmed S3 upload works for the first golden clip.

Remaining blocker:

- `/api/upload` fails after S3 upload when writing the upload row to Postgres.
- A direct local Postgres connection check to the configured AWS RDS database timed out.
- Next work should be completed on the AWS side: allow the current client IP to reach the RDS security group on PostgreSQL port `5432`, confirm the DB is publicly accessible for local testing, or run the backend from inside the AWS VPC.

Next command after AWS/RDS access is fixed:

```bash
source .venv/bin/activate
export DAVE_API_BASE_URL=http://127.0.0.1:8000
export DAVE_API_KEY=your-shared-api-key
python3 eval_golden_dataset.py --ids 001,027
```
