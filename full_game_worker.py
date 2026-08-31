"""Durable staged worker for DAVE full-game processing.

Run independently from FastAPI:
    python full_game_worker.py
"""

import asyncio
import hashlib
import json
import os
import signal
from datetime import datetime

from sqlalchemy import delete, func, select

from database import async_session
from job_queue import (
    claim_jobs,
    complete_job,
    enqueue_job,
    heartbeat_job,
    recover_stale_jobs,
    retry_or_fail_job,
)
from main import (
    BUCKET_NAME,
    _detect_play_segments,
    _extract_clip_bytes,
    _extract_frames,
    _predict_batch_with_rules,
    _probe_video_duration,
    _retrieve_rules_async,
    _s3_key_from_url,
    _safe_err_text,
    _summarize_frames_async,
    ensure_ruleset_columns,
    s3_client,
)
from models import Game, Play, PlayFinding, ProcessingJob, Upload


WORKER_CONCURRENCY = max(1, int(os.getenv("FULL_GAME_WORKER_CONCURRENCY", "4")))
CLAIM_BATCH_SIZE = max(WORKER_CONCURRENCY, int(os.getenv("FULL_GAME_CLAIM_BATCH_SIZE", "8")))
POLL_SECONDS = max(0.25, float(os.getenv("FULL_GAME_POLL_SECONDS", "2")))
STALE_SECONDS = max(60, int(os.getenv("FULL_GAME_STALE_SECONDS", "900")))
RETRY_SECONDS = max(1, int(os.getenv("FULL_GAME_RETRY_SECONDS", "15")))
ANALYSIS_VERSION = "staged-multi-finding-v1"


def now_utc():
    return datetime.utcnow()


async def enqueue_finalize_if_ready(db, game_id: int):
    remaining = await db.scalar(
        select(func.count()).select_from(Play).where(
            Play.game_id == game_id,
            Play.analysis_status.notin_(["done", "error"]),
        )
    )
    if remaining:
        return
    ids_result = await db.execute(
        select(Play.id).where(Play.game_id == game_id).order_by(Play.id)
    )
    play_ids = list(ids_result.scalars().all())
    signature = hashlib.sha256(
        ",".join(str(play_id) for play_id in play_ids).encode("ascii")
    ).hexdigest()[:16]
    await enqueue_job(
        db,
        job_key=f"game:{game_id}:finalize:multi-v1:{signature}",
        game_id=game_id,
        stage="finalize_game",
    )


async def read_s3(url: str) -> bytes:
    key = _s3_key_from_url(url)
    response = await asyncio.to_thread(s3_client.get_object, Bucket=BUCKET_NAME, Key=key)
    return await asyncio.to_thread(response["Body"].read)


async def put_s3(key: str, body: bytes) -> str:
    await asyncio.to_thread(
        s3_client.put_object,
        Bucket=BUCKET_NAME,
        Key=key,
        Body=body,
        ContentType="video/mp4",
    )
    return f"https://{BUCKET_NAME}.s3.{os.getenv('AWS_REGION', 'us-east-2')}.amazonaws.com/{key}"


async def handle_validate_upload(job: ProcessingJob):
    async with async_session() as db:
        game = await db.get(Game, job.game_id)
        if not game:
            raise RuntimeError("Game not found")
        video_bytes = await read_s3(game.s3_url)
        duration = await asyncio.to_thread(_probe_video_duration, video_bytes)
        if duration <= 0:
            raise RuntimeError("Uploaded file is not a readable video")
        game.source_duration = duration
        game.status = "queued_for_detection"
        game.progress = 0.03
        game.error_message = None
        await enqueue_job(
            db,
            job_key=f"game:{game.id}:detect_plays:v1",
            game_id=game.id,
            stage="detect_plays",
        )
        await db.commit()


async def handle_detect_plays(job: ProcessingJob):
    async with async_session() as db:
        game = await db.get(Game, job.game_id)
        if not game:
            raise RuntimeError("Game not found")
        existing_count = await db.scalar(
            select(func.count()).select_from(Play).where(Play.game_id == game.id)
        )
        if existing_count:
            game.status = "segments_ready"
            game.segment_count = existing_count
            game.progress = 0.25
            await db.commit()
            return

        game.status = "detecting_plays"
        game.progress = 0.08
        await db.commit()
        video_bytes = await read_s3(game.s3_url)
        segments = await asyncio.to_thread(_detect_play_segments, video_bytes)
        if not segments:
            raise RuntimeError("No playable segments could be detected")

        for segment in segments:
            upload = Upload(
                game_id=game.id,
                play_number=segment["play_number"],
                start_time=segment["start_time"],
                end_time=segment["end_time"],
                segment_confidence=segment["segment_confidence"],
                s3_url=game.s3_url,
                foul_type="",
                notes="",
                ruleset=game.ruleset,
                timestamp=now_utc(),
                status="segmented",
            )
            db.add(upload)
            await db.flush()
            db.add(Play(
                game_id=game.id,
                legacy_upload_id=upload.id,
                play_number=upload.play_number,
                source_start_time=upload.start_time,
                source_end_time=upload.end_time,
                snap_time=segment.get("snap_time"),
                live_ball_end_time=segment.get("live_ball_end_time"),
                clip_start_time=upload.start_time,
                clip_end_time=upload.end_time,
                detection_method=(
                    "legacy_fixed_window"
                    if upload.segment_confidence == 0.35
                    else "coaches_activity"
                ),
                detection_confidence=upload.segment_confidence,
                boundary_flags=segment.get("boundary_flags") or [],
                boundary_review_status="unreviewed",
                analysis_status="segmented",
                notes="",
                created_at=now_utc(),
                updated_at=now_utc(),
            ))
        game.status = "segments_ready"
        game.segment_count = len(segments)
        game.progress = 0.25
        game.segmented_at = now_utc()
        await db.commit()


async def handle_extract_clips(job: ProcessingJob):
    async with async_session() as db:
        game = await db.get(Game, job.game_id)
        if not game:
            raise RuntimeError("Game not found")
        plays_result = await db.execute(
            select(Play).where(Play.game_id == game.id).order_by(Play.play_number)
        )
        plays = plays_result.scalars().all()
        if not plays:
            raise RuntimeError("No detected plays are available")
        game.status = "extracting_clips"
        game.progress = 0.28
        await db.commit()
        video_bytes = await read_s3(game.s3_url)
        for play in plays:
            if not play.clip_s3_url:
                clip = await asyncio.to_thread(
                    _extract_clip_bytes,
                    video_bytes,
                    play.clip_start_time or play.source_start_time,
                    play.clip_end_time or play.source_end_time,
                )
                key = f"games/{game.id}/plays/{play.play_number}_{play.id}.mp4"
                play.clip_s3_url = await put_s3(key, clip)
            play.analysis_status = "queued_for_screening"
            play.updated_at = now_utc()
            upload = await db.get(Upload, play.legacy_upload_id)
            if upload:
                upload.clip_s3_url = play.clip_s3_url
                upload.s3_url = play.clip_s3_url
                upload.status = "queued"
            await enqueue_job(
                db,
                job_key=f"play:{play.id}:screen:multi-v1",
                game_id=game.id,
                play_id=play.id,
                stage="screen_play",
            )
        game.status = "analyzing"
        game.progress = 0.35
        await db.commit()


async def handle_screen_play(job: ProcessingJob):
    async with async_session() as db:
        play = await db.get(Play, job.play_id)
        game = await db.get(Game, job.game_id)
        if not play or not game or not play.clip_s3_url:
            raise RuntimeError("Play clip is unavailable")
        if not play.broad_screen:
            play.analysis_status = "screening"
            await db.commit()
            clip = await read_s3(play.clip_s3_url)
            frames = await asyncio.to_thread(_extract_frames, clip, 1, 8)
            play.broad_screen = await _summarize_frames_async(
                frames, notes=play.notes or "", ruleset=game.ruleset
            )
        play.analysis_status = "queued_for_retrieval"
        play.updated_at = now_utc()
        await enqueue_job(
            db,
            job_key=f"play:{play.id}:retrieve_rules:multi-v1",
            game_id=game.id,
            play_id=play.id,
            stage="retrieve_rules",
        )
        await db.commit()


async def handle_retrieve_rules(job: ProcessingJob):
    async with async_session() as db:
        play = await db.get(Play, job.play_id)
        game = await db.get(Game, job.game_id)
        if not play or not game or not play.broad_screen:
            raise RuntimeError("Broad visual screening is unavailable")
        if not play.retrieved_rules:
            play.analysis_status = "retrieving_rules"
            await db.commit()
            query = "\n".join(filter(None, [play.broad_screen, play.notes or ""]))
            play.retrieved_rules = await _retrieve_rules_async(query, top_k=6, ruleset=game.ruleset)
        play.analysis_status = "queued_for_focused_analysis"
        play.updated_at = now_utc()
        await enqueue_job(
            db,
            job_key=f"play:{play.id}:analyze_foul:multi-v1",
            game_id=game.id,
            play_id=play.id,
            stage="analyze_foul",
        )
        await db.commit()


async def handle_analyze_foul(job: ProcessingJob):
    async with async_session() as db:
        play = await db.get(Play, job.play_id)
        game = await db.get(Game, job.game_id)
        if not play or not game or not play.broad_screen:
            raise RuntimeError("Play evidence is unavailable")
        if not play.analysis_payload or "findings" not in play.analysis_payload:
            play.analysis_status = "focused_analysis"
            await db.commit()
            play.analysis_payload = await _predict_batch_with_rules(
                play.broad_screen,
                play.retrieved_rules or [],
                notes=play.notes or "",
                ruleset=game.ruleset,
                start_time=play.source_start_time,
                end_time=play.source_end_time,
            )
        play.analysis_status = "queued_for_storage"
        play.updated_at = now_utc()
        await enqueue_job(
            db,
            job_key=f"play:{play.id}:store_findings:multi-v1",
            game_id=game.id,
            play_id=play.id,
            stage="store_findings",
        )
        await db.commit()


async def handle_store_findings(job: ProcessingJob):
    async with async_session() as db:
        play = await db.get(Play, job.play_id)
        if not play or not play.analysis_payload:
            raise RuntimeError("Focused analysis result is unavailable")
        prediction = play.analysis_payload
        await db.execute(delete(PlayFinding).where(PlayFinding.play_id == play.id))
        for finding in prediction.get("findings") or []:
            db.add(PlayFinding(
                play_id=play.id,
                disposition=finding["disposition"],
                foul_label=finding["foul_label"],
                committing_team=finding["committing_team"],
                player_number=finding["player_number"],
                event_time=finding["event_time"],
                confidence=finding["confidence"],
                evidence_summary=finding["evidence_summary"],
                reasoning=finding["reasoning"],
                rule_citations=finding["rule_citations"],
                uncertainty_notes=finding["uncertainty_notes"],
                review_status="unreviewed",
                created_at=now_utc(),
                updated_at=now_utc(),
            ))
        play.result = prediction["result"]
        play.prediction_label = prediction["label"]
        play.confidence = prediction["confidence"]
        play.analysis_status = "done"
        play.analysis_version = ANALYSIS_VERSION
        play.analyzed_at = now_utc()
        play.updated_at = now_utc()
        upload = await db.get(Upload, play.legacy_upload_id)
        if upload:
            upload.status = "done"
            upload.result = play.result
            upload.prediction_label = play.prediction_label
            upload.confidence = play.confidence
            upload.explanation = json.dumps({
                "rule_citations": prediction["rule_citations"],
                "evidence_summary": prediction["evidence_summary"],
                "why_foul_or_no_foul": prediction["why_foul_or_no_foul"],
                "uncertainty_notes": prediction["uncertainty_notes"],
                "no_foul_reason": prediction.get("no_foul_reason", ""),
                "findings": prediction.get("findings") or [],
            })
            upload.retrieved_rules = play.retrieved_rules
            upload.processed_at = now_utc()
        await db.flush()
        game = await db.get(Game, play.game_id)
        terminal_count = await db.scalar(
            select(func.count()).select_from(Play).where(
                Play.game_id == play.game_id,
                Play.analysis_status.in_(["done", "error"]),
            )
        )
        total_count = await db.scalar(
            select(func.count()).select_from(Play).where(Play.game_id == play.game_id)
        )
        if game and total_count:
            game.analyzed_count = terminal_count
            game.progress = 0.35 + (0.60 * terminal_count / total_count)
        await enqueue_finalize_if_ready(db, play.game_id)
        await db.commit()


async def handle_finalize_game(job: ProcessingJob):
    async with async_session() as db:
        game = await db.get(Game, job.game_id)
        if not game:
            raise RuntimeError("Game not found")
        total = await db.scalar(select(func.count()).select_from(Play).where(Play.game_id == game.id))
        completed = await db.scalar(select(func.count()).select_from(Play).where(
            Play.game_id == game.id, Play.analysis_status == "done"
        ))
        failed = await db.scalar(select(func.count()).select_from(Play).where(
            Play.game_id == game.id, Play.analysis_status == "error"
        ))
        game.segment_count = total
        game.analyzed_count = completed + failed
        game.progress = 1.0
        game.status = "done" if failed == 0 and completed == total else "error"
        game.error_message = None if failed == 0 else f"{failed} of {total} play analyses failed."
        game.analyzed_at = now_utc()
        await db.commit()


HANDLERS = {
    "validate_upload": handle_validate_upload,
    "detect_plays": handle_detect_plays,
    "extract_clips": handle_extract_clips,
    "screen_play": handle_screen_play,
    "retrieve_rules": handle_retrieve_rules,
    "analyze_foul": handle_analyze_foul,
    "store_findings": handle_store_findings,
    "finalize_game": handle_finalize_game,
}


async def process_job(job: ProcessingJob):
    handler = HANDLERS.get(job.stage)
    if not handler:
        await retry_or_fail_job(job.id, f"Unknown stage: {job.stage}", RETRY_SECONDS)
        return
    heartbeat_interval = max(10, STALE_SECONDS // 3)

    async def keep_alive():
        while True:
            await asyncio.sleep(heartbeat_interval)
            await heartbeat_job(job.id)

    heartbeat = asyncio.create_task(keep_alive())
    try:
        await handler(job)
        await complete_job(job.id)
    except Exception as error:
        exhausted = await retry_or_fail_job(job.id, _safe_err_text(error), RETRY_SECONDS)
        if exhausted:
            async with async_session() as db:
                if job.play_id:
                    play = await db.get(Play, job.play_id)
                    if play:
                        play.analysis_status = "error"
                        play.error_message = _safe_err_text(error)
                        play.updated_at = now_utc()
                        await db.flush()
                        await enqueue_finalize_if_ready(db, play.game_id)
                game = await db.get(Game, job.game_id)
                if game and not job.play_id:
                    game.status = "error"
                    game.error_message = _safe_err_text(error)
                await db.commit()
    finally:
        heartbeat.cancel()
        try:
            await heartbeat
        except asyncio.CancelledError:
            pass


async def run_worker():
    ensure_ruleset_columns()
    await recover_stale_jobs(STALE_SECONDS)
    stopping = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signal_name in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signal_name, stopping.set)
    semaphore = asyncio.Semaphore(WORKER_CONCURRENCY)

    async def bounded(job):
        async with semaphore:
            await process_job(job)

    while not stopping.is_set():
        jobs = await claim_jobs(min(CLAIM_BATCH_SIZE, WORKER_CONCURRENCY))
        if not jobs:
            try:
                await asyncio.wait_for(stopping.wait(), timeout=POLL_SECONDS)
            except asyncio.TimeoutError:
                pass
            continue
        await asyncio.gather(*(bounded(job) for job in jobs))


if __name__ == "__main__":
    asyncio.run(run_worker())
