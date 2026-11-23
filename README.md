# DAVE Video Uploader Backend

Backend API for DAVE - Digital Artificial Video Evaluation. The service accepts high school football play videos, stores them in S3, extracts representative frames, uses OpenAI vision analysis to summarize the visible play action, retrieves likely NFHS rule snippets from Postgres/pgvector, and stores the AI prediction for human review.

## Coaches-film MVP status

On November 19, 2026, the coaches-film development branch has completed a local end-to-end test. The React upload of Leyden–Morton created Game 5. The detector proposed 12 candidates. Removing a setup-only sequence and an incomplete final entry produced the same 10 boundaries as the official's saved review. The worker resumed queued work after restart and completed all 44 jobs. Every extracted clip played in the review screen.

The official judged all 10 plays legal. DAVE returned eight no-foul results and flagged plays 4 and 9 as uncertain. Both involved forward handoffs. The quarterback also carried out a fake on play 9. The subsequent analyzer update asks DAVE to distinguish a hand-to-hand transfer from an airborne shovel pass and to mark a hidden exchange unclear. The genuine short forward pass control is still pending.

On November 20, two WordPress subscriber sessions verified uploader-private access against the local API. Member Game 6 used the dedicated S3 test bucket. Shadow mode initially selected four legacy windows. From its twelve play-first candidates we submitted the ten boundaries matching the official reference. All 44 jobs completed and the uploader could retrieve every clip. A second member could neither open the timeline nor change findings. The model returned nine no-foul results, including plays 4 and 9. It flagged play 8 as uncertain for offensive pass interference despite the official's no-foul review. Full-length game capacity and accuracy on unseen film still need testing.

The separate 16-clip blind evaluation with focused review and 83 high-school rule snippets matched four foul/no-foul outcomes and two exact labels. It flagged two of seven clean controls as fouls. Officials retain responsibility for rulings and enforcement. The branch passed 34 backend tests at the handoff-guidance checkpoint. See the [backend agent guide](agents.md) for the analysis paths and verification commands.

The play-first detector treats a padded clip over 60 seconds as needing boundary review. If those clips cover more than half of a timeline derived from black transitions, it tries motion detection. If motion yields no candidates, the transition timeline stays available for correction. The review screen identifies oversized clips, and game analysis can proceed once they have been shortened, split, or removed. The same limit applies to saved games when their timelines are reopened.

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
PLAY_DETECTOR_MODE=shadow_compare
PLAY_FIRST_ACCEPTED=false
LEGACY_GAME_SEGMENT_SECONDS=35
LEGACY_GAME_SEGMENT_GAP_SECONDS=5
FULL_GAME_WORKER_CONCURRENCY=4
FULL_GAME_CLAIM_BATCH_SIZE=8
FULL_GAME_POLL_SECONDS=2
FULL_GAME_STALE_SECONDS=900
FULL_GAME_RETRY_SECONDS=15
```

`CORS_ALLOWED_ORIGINS` is a comma-separated list. If it is omitted, the backend allows the DAVE production domains and `http://localhost:3000` for local development.

The database must have pgvector available. `ingest_rules.py` creates the `rules` table and vector extension if permissions allow it.

## API Authentication

WordPress members use a short-lived bearer token signed by the plugin. The API records its signed user ID on each new game or standalone upload. Game lists, playback links, findings, and edits are scoped to that member. Historical records with unknown ownership stay available to the server administrator. A playback URL gives its holder temporary access until it expires.

The server-side `DAVE_API_KEY` also grants administrative access through the `X-DAVE-API-Key` header. Use a different random secret of at least 32 characters for each WordPress installation and its paired API. Keep it out of browser bundles. The public health check is `/health`.

For an isolated staging installation, configure the staging database, private bucket, server keys, and WordPress origin. Populate the high-school rules with `ingest_rules.py`. Start both `uvicorn main:app` and `python full_game_worker.py` against the same staging configuration. Set `PLAY_DETECTOR_MODE=coaches_activity` on the staging worker to exercise the play-first timeline. The default shadow comparison selects legacy windows until play-first is accepted. Size the upload proxy limit, timeout, temporary disk, and worker memory against a representative full-length game.

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
uvicorn main:app --reload --host 0.0.0.0 --port 8000
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
| `/api/games/{game_id}/timeline` | GET | Return migrated plays, findings, and processing-job state for a full game |
| `/api/games/{game_id}/segmentation-comparison` | GET | Return fixed-window/play-first overlap and human-correction metrics |
| `/api/migration/readiness` | GET | Audit migration integrity and legacy-retirement blockers |
| `/api/play-findings/{finding_id}/review` | PATCH | Agree with, correct, dismiss, or clear review of one foul finding |
| `/api/plays/{play_id}/findings` | POST | Add an official finding to a full-game play |
| `/api/games/{game_id}/plays/{play_id}/boundary` | PATCH | Adjust one play boundary and queue selective re-analysis |
| `/api/games/{game_id}/plays/{play_id}/split` | POST | Split one candidate into two plays |
| `/api/games/{game_id}/plays/{play_id}/merge` | POST | Merge two consecutive plays |
| `/api/games/{game_id}/plays/{play_id}/reanalyze` | POST | Re-analyze one play without changing its boundary |
| `/api/games/{game_id}/plays/{play_id}` | DELETE | Remove one candidate and renumber later plays |

Full-game play analysis may return multiple independent foul or uncertain
findings. A completed no-foul review is represented by `result: "NO FOUL"` and
an empty `findings` array. Enforcement, acceptance, declination, offsetting
administration, and disqualification remain outside the AI finding schema.

## Full-Game Compatibility Migration

Backend startup creates the additive `plays`, `play_findings`,
`processing_jobs`, and `segmentation_comparisons` tables and backfills existing
batch uploads without changing or deleting rows in `uploads`. Existing
single-play and batch APIs continue to use the legacy fields while the timeline
API exposes the new model.

`PLAY_DETECTOR_MODE` supports:

- `shadow_compare` (default): run both detectors, persist candidates and overlap
  metrics, and continue selecting fixed windows until `PLAY_FIRST_ACCEPTED=true`.
- `coaches_activity_with_fallback`: use coaches-film activity detection
  and fall back to legacy fixed windows when no candidates are found.
- `coaches_activity`: use only the new detector and surface detection failures.
- `fixed_window`: use the retained legacy segmentation path.

The play-first detector checks first for repeated, brief black transitions used
by many exported coaches-film cut-ups. When present, those transitions produce
ordered, gap-free source clips (`coaches_cutup_transitions`). Continuous film
without repeated transitions continues through formation/activity detection.
Low-activity cut-up intervals remain visible and flagged for review rather than
being silently discarded.

Fixed-window comparisons are a baseline, not labeled ground truth. The API calls
non-overlapping candidates `unmatched`; it does not call them missed plays. Once
an official edits a timeline, correction counts and total boundary movement are
reported separately. Elapsed time to the first correction is available as an
operational proxy, but is not active review labor time.

Promote play-first detection only after reviewed test games meet the team's
acceptance targets, then set `PLAY_FIRST_ACCEPTED=true` while retaining
`fixed_window` for one release. `/api/migration/readiness` deliberately keeps
`ready_to_remove_legacy` false until that release and rollback check are complete.

## Full-Game Worker

Full-game processing runs outside the FastAPI process through PostgreSQL-backed,
idempotently keyed jobs. Start at least one worker process alongside the API:

```bash
python full_game_worker.py
```

Workers claim jobs with `FOR UPDATE SKIP LOCKED`, so multiple worker processes
may run safely. `FULL_GAME_WORKER_CONCURRENCY` bounds concurrent play stages
inside each process. Interrupted running jobs return to the queue after
`FULL_GAME_STALE_SECONDS`; failed jobs retry up to their stored `max_attempts`.

The single-play `/upload` workflow retains its existing background path. The
durable queue applies to full-game processing only.

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

The script uploads each local clip, waits for processing to finish, compares DAVE's prediction against `golden-dataset/labels.csv`, prints misses, and writes a CSV report under `golden-dataset/reports/`.

## Development Checks

Run a basic syntax check:

```bash
python3 -m py_compile main.py models.py database.py ingest_rules.py video_processor.py
```

## Current Improvement Priorities

- Replace short rule snippets with fuller NFHS rule-book chunks.
- Store the generated video summary on each upload for auditability.
- Add automated tests for `/api/rules/search`, upload creation, retry, and review.
- Exercise the durable full-game queue on an unseen, full-length game after checking the handoff regression.
- Add structured logging instead of `print` statements.
- Consider wiring the richer `video_processor.py` pipeline into `main.py` once the current flow is stable.
