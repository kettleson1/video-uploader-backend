# DAVE backend agent guide

## Film analysis

The single-play upload path in `main.py` uses `_build_visual_evidence_async` to screen timestamped frames across a clip. When the screen nominates contact, a second pass samples around the event. Actual JPEG frames and prompt text go to OpenAI. Rule retrieval embeds the resulting description and searches PostgreSQL with pgvector. The description is a model observation that an official can correct.

For a full game, run `full_game_worker.py` alongside the API. The worker claims durable jobs from PostgreSQL, detects candidate boundaries, and waits for the timeline to be accepted for analysis. It extracts clips, screens up to eight timestamped frames per play, retrieves rules, evaluates multiple possible fouls, stores findings, and finalizes game counts. Preserve job keys, retry behavior, existing single-play records, and human reviews when changing these stages.

## Ball exchanges

`ball_exchange.py` supplies guidance to broad screening, focused visual review, and both prediction paths. A direct handoff can move forward. For a short shovel pass, look for the ball traveling through the air. Trace the ball carrier separately from a quarterback carrying out a fake. If the transfer is hidden, record the exchange as unclear. An unclear exchange alone cannot support pass interference.

## Member access

Use the signed WordPress subject as the owner of a new game or standalone upload. A game clip inherits access from its parent game. Enforce ownership before returning clips, signing playback URLs, or changing any review or job. Older rows without an owner are accessible with the server administrator key. Keep the key on the server. Run the two-member HTTP regression by setting `DAVE_PRIVACY_TEST_DATABASE_URL` to a disposable PostgreSQL database initialized with the DAVE schema.

## Verification

Run `python -m unittest discover -q` in the backend environment. For an end-to-end check, use the isolated local database and dedicated test bucket, submit film from the React screen, compare boundaries against the saved review, reload, and verify clip playback and completed jobs.

Use `eval_golden_dataset.py --direct-local --blind` with a configured test database for blind clip analysis. This path reads local videos and still calls OpenAI. Ensure the intended ruleset has populated rule embeddings before comparing predictions. Score the foul disposition and exact label separately. Keep official notes and expected outcomes out of the model inputs.

See the [backend status](README.md#coaches-film-mvp-status) for the measured results and the pending film regression.
