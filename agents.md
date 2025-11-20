## Agent Overview

This backend currently leans on three lightweight "agents" (small, single-responsibility LLM interactions) that orchestrate the automated review of uploaded football clips. They are defined in `main.py` and stitched together inside the `_process_upload_bg` pipeline.

1. **Frame Summarizer (`_summarize_frames_async`)**
   - **Input**: Up to six JPEG frames that `_extract_frames` pulls from each uploaded MP4 (1 FPS cap).
   - **Model**: `OPENAI_CHAT_MODEL` (defaults to `gpt-4o-mini`) via `AsyncOpenAI`.
   - **Prompting**: System prompt asks for a concise description of formation, motion, contact, and obvious infractions. The user message simply states how many frames were captured; raw images are *not* transmitted, so nothing sensitive leaves the box.
   - **Output**: A ~70 word natural-language play summary. Downstream components treat this as ground truth context.

2. **Embedding Generator (`_embed_text`)**
   - **Input**: Arbitrary text string.
   - **Model**: `OPENAI_EMBED_MODEL` (defaults to `text-embedding-3-large`, 3,072 dims) via the embeddings endpoint. Used by both the runtime retrieval (`_retrieve_rules_async`) and the `ingest_rules.py` bootstrapper.
   - **Output**: List of floats that represent the vectorized text. The async variant stores vectors in Postgres `vector(3072)` columns.

3. **Rule-Aligned Predictor (`_predict_with_rules`)**
   - **Input**: The play summary plus the `top_k` rule snippets returned from pgvector search. Snippets are formatted as bulleted text before prompting.
   - **Model**: Same `OPENAI_CHAT_MODEL` configured above.
   - **Prompting**: System prompt constrains behavior to “high school football rule assistant,” asking for JSON containing `{label, confidence, explanation}`. Confidence is a float in `[0,1]`.
   - **Output**: Parsed JSON. The label is later thresholded (default `CONFIDENCE_THRESHOLD=0.60`) to optionally fall back to `Uncertain`. The explanation string is saved for UI transparency.

## Data Flow Summary

```
Upload -> S3 -> DB row (queued)
            \
             -> `_process_upload_bg` task
                    1. `_extract_frames`
                    2. `_summarize_frames_async`  (Agent 1)
                    3. `_retrieve_rules_async`
                       └─ `_embed_text`          (Agent 2)
                    4. `_predict_with_rules`      (Agent 3)
                    5. Persist prediction + retrieved rule snippets
```

All long-running work stays off the request/response path thanks to `asyncio.create_task` inside `/upload`. Retries call the same worker via `/api/retry/{upload_id}`.

## Supporting Scripts

- `ingest_rules.py` ingests every `rules/*.txt` file, calls `_embed_text`, and upserts the vectors into Postgres (requires the `vector` extension and `uq_rules_title_section` unique index). Run it whenever rule text changes.
- `rules/` contains the canonical snippets used both for retrieval and for the predictor to cite in explanations.

## Extending the Agent Layer

- **Adding new models**: Introduce new env vars (e.g., `OPENAI_VISION_MODEL`) and thread them through the helper responsible for that step. Keep prompts alongside code for easy auditing.
- **Changing prompts**: Update the static strings inside `_summarize_frames_async` or `_predict_with_rules`. Because those strings live in code, remember to re-run regression tests or at minimum submit a QA clip before deploying.
- **More context**: If you want to feed audio transcripts or telemetry into the predictor, enrich the summary string before the retrieval step—no DB changes needed unless storing the extra context.
- **Observability**: Each stage logs an emoji-tagged line; expand this with structured logging if you need dashboards.

## Operational Notes

- Secrets: `AWS_S3_BUCKET`, `DATABASE_URL`, and `OPENAI_API_KEY` must be defined (see `.env`). Missing values will raise during app startup.
- Confidence fallback: The user-facing `prediction_label` becomes `Uncertain` when `confidence < CONFIDENCE_THRESHOLD`. Adjust via env var rather than editing code so that Airflow/infra can override per environment.
- Human review: `/api/review/{upload_id}` and `/api/plays/{upload_id}/review` let reviewers correct predictions. Those handlers never call an agent; they simply patch DB columns (`human_label`, `human_notes`, `reviewed_at`).

Use this document as the canonical reference before tweaking prompts, swapping models, or inserting additional LLM calls.
