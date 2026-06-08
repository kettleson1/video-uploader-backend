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
- Added `check_postgres_connection.py` to diagnose local RDS/Postgres access.
- Improved database connection setup in `database.py` with a timeout, `pool_pre_ping`, configurable SQL logging, and clearer missing-`DATABASE_URL` failure.
- Improved `/api/upload` database error logging in `main.py` by printing `repr(...)` for DB exceptions.
- Fixed the AWS RDS/Postgres network blocker by allowing the current local public IP through the `default` security group (`sg-0f5593b9524adac92`) on PostgreSQL/TCP port `5432`.
- Confirmed `python check_postgres_connection.py` passes with TCP and SQLAlchemy `select 1` success.
- Completed a two-clip golden eval with `python3 eval_golden_dataset.py --ids 001,027`.
- Clip `001_york_gbs_dpi.mov` passed with expected label `pass_interference_defense`.
- Clip `027_free_kick_out_of_bounds.mov` passed with expected label `free_kick_out_of_bounds`.
- Eval summary: `2/2` correct, `100.0%` accuracy, no misses.
- Report written locally: `golden-dataset/reports/golden_eval_20260607T180954Z.csv`.
- Started the full 27-clip eval. Clips `001` through `020` passed, then the client timed out while uploading the large `021_no_foul_not_illegal_shift.mov` file.
- Updated `eval_golden_dataset.py` with `--request-timeout-seconds` and a longer default request timeout so large video uploads can complete.
- Resumed the golden eval with `python3 eval_golden_dataset.py --ids 021,022,023,024,025,026,027 --request-timeout-seconds 600`.
- Clips `021` through `027` all passed.
- Combined golden eval result across the partial and resumed reports: `27/27` correct, `100.0%` accuracy, no misses.
- Reports used for combined result: `golden_eval_20260607T182923Z.csv` and `golden_eval_20260608T202127Z.csv`.

Next command for continued local eval testing:

```bash
source .venv/bin/activate
export DAVE_API_BASE_URL=http://127.0.0.1:8000
export DAVE_API_KEY=your-shared-api-key
python3 eval_golden_dataset.py
```
