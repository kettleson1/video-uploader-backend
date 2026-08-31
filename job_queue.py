"""PostgreSQL-backed durable job queue primitives for full-game processing."""

from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import select, text as sqltext
from sqlalchemy.dialects.postgresql import insert

from database import async_session
from models import ProcessingJob


def now_utc():
    return datetime.utcnow()


async def enqueue_job(db, *, job_key: str, game_id: int, stage: str, play_id: Optional[int] = None):
    statement = insert(ProcessingJob).values(
        job_key=job_key,
        game_id=game_id,
        play_id=play_id,
        stage=stage,
        status="queued",
        attempts=0,
        max_attempts=3,
        available_at=now_utc(),
        created_at=now_utc(),
        updated_at=now_utc(),
    ).on_conflict_do_nothing(index_elements=["job_key"])
    await db.execute(statement)


async def recover_stale_jobs(timeout_seconds: int):
    cutoff = now_utc() - timedelta(seconds=timeout_seconds)
    async with async_session() as db:
        await db.execute(sqltext("""
            UPDATE processing_jobs
            SET status = 'queued', started_at = NULL, available_at = :now,
                error_message = 'Recovered after worker interruption', updated_at = :now
            WHERE status = 'running'
              AND COALESCE(updated_at, started_at, created_at) < :cutoff
        """), {"now": now_utc(), "cutoff": cutoff})
        await db.commit()


async def claim_jobs(limit: int):
    async with async_session() as db:
        result = await db.execute(sqltext("""
            WITH claimed AS (
                SELECT id
                FROM processing_jobs
                WHERE status = 'queued'
                  AND attempts < max_attempts
                  AND (available_at IS NULL OR available_at <= :now)
                ORDER BY id
                FOR UPDATE SKIP LOCKED
                LIMIT :limit
            )
            UPDATE processing_jobs AS job
            SET status = 'running', attempts = job.attempts + 1,
                started_at = :now, updated_at = :now, error_message = NULL
            FROM claimed
            WHERE job.id = claimed.id
            RETURNING job.id
        """), {"now": now_utc(), "limit": limit})
        job_ids = [row[0] for row in result.all()]
        await db.commit()
        if not job_ids:
            return []
        jobs_result = await db.execute(
            select(ProcessingJob).where(ProcessingJob.id.in_(job_ids)).order_by(ProcessingJob.id)
        )
        return jobs_result.scalars().all()


async def complete_job(job_id: int):
    async with async_session() as db:
        job = await db.get(ProcessingJob, job_id)
        if job:
            job.status = "completed"
            job.finished_at = now_utc()
            job.updated_at = now_utc()
            await db.commit()


async def heartbeat_job(job_id: int):
    async with async_session() as db:
        job = await db.get(ProcessingJob, job_id)
        if job and job.status == "running":
            job.updated_at = now_utc()
            await db.commit()


async def retry_or_fail_job(job_id: int, error_message: str, retry_delay_seconds: int):
    async with async_session() as db:
        job = await db.get(ProcessingJob, job_id)
        if not job:
            return False
        exhausted = int(job.attempts or 0) >= int(job.max_attempts or 3)
        job.status = "failed" if exhausted else "queued"
        job.error_message = error_message[:2000]
        job.available_at = now_utc() + timedelta(seconds=retry_delay_seconds)
        job.finished_at = now_utc() if exhausted else None
        job.updated_at = now_utc()
        await db.commit()
        return exhausted
