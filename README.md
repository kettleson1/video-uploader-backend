# DAVE Video Uploader Backend

Backend API for DAVE - Digital Artificial Video Evaluation. The service accepts high school football play videos, stores them in S3, extracts representative frames, uses OpenAI vision analysis to summarize the visible play action, retrieves likely NFHS rule snippets from Postgres/pgvector, and stores the AI prediction for human review.

## What This Service Does

- Accepts uploaded football video clips from the frontend.
- Stores original videos in AWS S3.
- Creates an upload record in PostgreSQL.
- Processes each upload in the background.
- Extracts video frames with `ffmpeg`.
- Sends the actual frames to an OpenAI vision-capable chat model.
- Embeds the play summary and searches the `rules` table with pgvector.
- Predicts the likely foul label, confidence, and explanation.
- Saves retrieved rules and prediction results for review.
- Supports human correction and retry workflows.

## Tech Stack

- FastAPI
- PostgreSQL
- pgvector
- SQLAlchemy
- asyncpg
- AWS S3
- OpenAI API
- ffmpeg

## Repository Layout

```text
main.py                 FastAPI app, upload routes, background processing, rule search
models.py               SQLAlchemy models for uploads and rules
database.py             Async database connection setup
check_postgres_connection.py  Local RDS/Postgres connectivity diagnostic
ingest_rules.py         Loads rule text files into Postgres with embeddings
video_processor.py      More advanced vision pipeline module, not fully wired into main.py yet
rules/                  Local rule snippet source files
golden-dataset/         NFHS golden dataset labels and candidate review metadata
eval_golden_dataset.py  Uploads golden clips and reports model misses
agents.md               Current agent/pipeline notes
requirements.txt        Runtime Python dependencies
requirements_additions.txt  Optional/additional video-processing dependency notes
```

## Required Environment Variables

Create a `.env` file or set these in the hosting environment:

```bash
AWS_REGION=us-east-2
AWS_S3_BUCKET=your-s3-bucket-name
DATABASE_URL=postgresql://user:password@host:5432/database
OPENAI_API_KEY=your-openai-api-key
DAVE_API_KEY=generate-a-long-random-shared-secret
```

Optional:

```bash
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBED_MODEL=text-embedding-3-large
OPENAI_IMAGE_DETAIL=high
CONFIDENCE_THRESHOLD=0.60
CORS_ALLOWED_ORIGINS=https://www.davesystemsinc.com,https://davesystemsinc.com,http://localhost:3000
```

`CORS_ALLOWED_ORIGINS` is a comma-separated list. If it is omitted, the backend allows the DAVE production domains and `http://localhost:3000` for local development.

The database must have pgvector available. `ingest_rules.py` creates the `rules` table and vector extension if permissions allow it.

## AWS RDS/Postgres Connectivity

For local Mac testing, the configured AWS RDS Postgres instance must be reachable from the current public IP address.

The June 7, 2026 local setup used:

- RDS database: `football-metadata-db`
- Engine: PostgreSQL
- Region: `us-east-2`
- Port: `5432`
- Security group: `default` / `sg-0f5593b9524adac92`
- Public access: enabled for local testing
- Inbound security group rule: PostgreSQL/TCP `5432` from the current local public IP as `/32`

Use the diagnostic script before running golden eval uploads:

```bash
source .venv/bin/activate
python check_postgres_connection.py
```

A correct local setup prints:

```text
tcp_5432=ok
sqlalchemy_select_1=ok value=1
```

If it times out, update the RDS security group inbound rule for PostgreSQL `5432` to the current public IP, or run the backend from inside the same AWS VPC.

## API Authentication

All application API routes require this request header:

```text
X-DAVE-API-Key: <DAVE_API_KEY>
```

The `/health` route remains public so uptime checks can work without credentials.

Generate a local key with:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

Use the same value in the frontend as `REACT_APP_DAVE_API_KEY`.

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install `ffmpeg` if it is not already available:

```bash
brew install ffmpeg
```

For Linux hosting:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

## Load Rule Embeddings

Run this whenever files in `rules/` change:

```bash
python ingest_rules.py
```

This reads every `rules/*.txt` file, creates an embedding from the rule title and body, and upserts it into PostgreSQL.

## Run the API

```bash
source .venv/bin/activate
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API docs are available at:

```text
http://localhost:8000/api/docs
```

## Main API Routes

| Route | Method | Purpose |
| --- | --- | --- |
| `/health` | GET | Basic health check |
| `/upload` | POST | Upload a video clip |
| `/api/upload` | POST | Alias for video upload |
| `/api/plays` | GET | List uploaded plays with pagination and filters |
| `/api/retry/{upload_id}` | POST | Re-run background processing for an upload |
| `/api/rules/list` | GET | Return rules for the frontend dropdown |
| `/api/rules/search` | GET | Search rule snippets with pgvector |
| `/api/plays/{upload_id}/feedback` | POST | Save official agree/disagree feedback |
| `/api/review/{upload_id}` | POST | Save human review and optional prediction override |
| `/api/plays/{upload_id}/review` | PATCH | Save or clear human review data |

## Upload Flow

`/api/plays` accepts `limit`, `offset`, `status`, `reviewed`, and `q` query parameters. It returns `items`, `total`, `limit`, `offset`, and `has_more` so the frontend can page through results without a hard display cap.

1. The frontend posts multipart form data with:
   - `file`
   - `foul_type`
   - `notes`
2. The backend uploads the video to S3.
3. A queued upload record is written to Postgres.
4. `_process_upload_bg` runs in the background.
5. The worker downloads the video from S3.
6. `ffmpeg` extracts representative JPEG frames.
7. OpenAI analyzes the actual frames and produces an evidence summary.
8. The summary, foul hint, and notes are embedded.
9. pgvector retrieves candidate NFHS rule snippets.
10. OpenAI selects the likely foul label and writes an explanation.
11. The upload record is marked `done`, `Uncertain`, or `error`.

## Rule Quality Notes

The current `rules/` files are usable for early retrieval testing, but many are short summaries. For better foul detection and rule-book alignment, each rule file should include:

- NFHS rule number and article.
- Foul name.
- Required elements of the foul.
- Exceptions and non-foul examples.
- Penalty and enforcement information.
- Closely related fouls that may be confused with it.

Better source chunks will improve retrieval more than prompt tuning alone.

## Golden Dataset

The `golden-dataset/` folder tracks the NFHS/high-school regression set for DAVE model and prompt changes.

Dataset status:

- 27 approved high-school/NFHS clips have been created locally.
- The label manifest contains 16 foul clips and 11 no-foul clips.
- The local video files match the filenames in `golden-dataset/labels.csv`.
- The large `.mov` files are intentionally kept out of git; the repo tracks the labels, sources, and review notes.

Tracked in git:

- `golden-dataset/labels.csv`: final expected labels/results for approved and planned clips.
- `golden-dataset/candidate_sources.csv`: human-review checklist with exact MIBT candidate source pages.
- `golden-dataset/candidate_review.html`: lightweight review index for browsing candidate sources.
- `golden-dataset/videos/.gitkeep`: placeholder for the local video folder.
- `golden-dataset/reports/.gitkeep`: placeholder for generated eval reports.

Not tracked in git:

- Actual `.mov` and `.mp4` clips under `golden-dataset/videos/`.
- Generated eval CSV reports under `golden-dataset/reports/`.

Current approved label coverage:

| Expected label | Clip count |
| --- | --- |
| `None` / no foul | 11 |
| `pass_interference_defense` | 4 |
| `pass_interference_offense` | 2 |
| `holding` | 3 |
| `illegal_block_in_back` | 1 |
| `illegal_formation` | 1 |
| `illegal_blindside_block` | 1 |
| `targeting` | 1 |
| `horse_collar_tackle` | 1 |
| `roughing_the_kicker` | 1 |
| `free_kick_out_of_bounds` | 1 |

When a human approves a new clip:

1. Save the local clip in `golden-dataset/videos/` using the planned filename from `candidate_sources.csv`.
2. Mark the row in `candidate_sources.csv` as `approved`.
3. Update the matching row in `labels.csv` with the final `expected_label`, `expected_result`, rule reference, and notes.
4. Keep the video file local unless there is a separate approved storage location for large/private clips.

Run the golden dataset eval:

```bash
export DAVE_API_BASE_URL=https://your-backend-host
export DAVE_API_KEY=your-shared-api-key
python3 eval_golden_dataset.py
```

For a local file check without uploading:

```bash
python3 eval_golden_dataset.py --validate-only
```

To run only specific clips while debugging:

```bash
python3 eval_golden_dataset.py --ids 001,007,025
```

If a large clip upload times out, rerun only the remaining clip IDs. The eval runner uses a longer request timeout by default, and it can be increased if needed:

```bash
python3 eval_golden_dataset.py --ids 021,022,023,024,025,026,027 --request-timeout-seconds 600
```

The script uploads each local clip, waits for processing to finish, compares DAVE's prediction against `golden-dataset/labels.csv`, prints misses, and writes a CSV report under `golden-dataset/reports/`.

### Local Eval Setup Progress

As of June 7, 2026, local setup has been advanced through these checks:

- Git branch `codex/golden-eval-runner` was fetched and checked out locally.
- A stale `.git/HEAD.lock` was cleared so Git could update normally.
- The 27 local golden video files were placed under `golden-dataset/videos/`.
- `python3 eval_golden_dataset.py --validate-only` passed with 27 clips.
- A fresh `.venv` was created because the old `venv/` folder was incomplete.
- Dependencies from `requirements.txt` installed successfully, including `asyncpg`.
- The FastAPI backend starts locally with `python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000`.
- `DAVE_API_KEY` was added to `.env` and the eval request reached `/api/upload`.
- S3 upload succeeded for `001_york_gbs_dpi.mov`.
- AWS RDS/Postgres connectivity was fixed by enabling local access through the `default` security group for PostgreSQL port `5432`.
- `python check_postgres_connection.py` now passes with `tcp_5432=ok` and `sqlalchemy_select_1=ok value=1`.
- A two-clip eval run with `--ids 001,027` passed end to end.
- The run uploaded both clips, wrote upload rows to Postgres, processed frames, retrieved rules, predicted the expected labels, and produced `100.0%` accuracy.
- Report written locally: `golden-dataset/reports/golden_eval_20260607T180954Z.csv`.

To continue local eval testing, restart the backend and run:

```bash
source .venv/bin/activate
export DAVE_API_BASE_URL=http://127.0.0.1:8000
export DAVE_API_KEY=your-shared-api-key
python3 eval_golden_dataset.py --ids 001,027
```

## Development Checks

Run a basic syntax check:

```bash
python3 -m py_compile main.py models.py database.py ingest_rules.py video_processor.py
```

## Current Improvement Priorities

- Replace short rule snippets with fuller NFHS rule-book chunks.
- Store the generated video summary on each upload for auditability.
- Add automated tests for `/api/rules/search`, upload creation, retry, and review.
- Move background processing to a real worker queue if upload volume increases.
- Add structured logging instead of `print` statements.
- Consider wiring the richer `video_processor.py` pipeline into `main.py` once the current flow is stable.
