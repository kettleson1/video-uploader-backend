#!/usr/bin/env python3
"""Diagnose local connectivity to the configured Postgres database."""

from __future__ import annotations

import asyncio
import os
import socket
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine


def async_database_url(value: str) -> str:
    return value.replace("postgresql://", "postgresql+asyncpg://", 1)


async def check_sqlalchemy(url: str, timeout_seconds: float) -> None:
    engine = create_async_engine(
        async_database_url(url),
        pool_pre_ping=True,
        connect_args={"timeout": timeout_seconds},
    )
    try:
        async with engine.connect() as conn:
            result = await conn.execute(text("select 1"))
            print(f"sqlalchemy_select_1=ok value={result.scalar_one()}")
    finally:
        await engine.dispose()


def main() -> int:
    load_dotenv(dotenv_path=Path(".env"))
    database_url = os.getenv("DATABASE_URL")
    timeout_seconds = float(os.getenv("DB_CONNECT_TIMEOUT", "8"))

    if not database_url:
        print("DATABASE_URL=missing")
        return 2

    parsed = urlparse(database_url)
    host = parsed.hostname
    port = parsed.port or 5432
    database = parsed.path.lstrip("/")

    print(f"database_host={host}")
    print(f"database_port={port}")
    print(f"database_name={database}")
    print(f"database_query={parsed.query or 'none'}")

    sock = socket.socket()
    sock.settimeout(timeout_seconds)
    try:
        sock.connect((host, port))
        print("tcp_5432=ok")
    except Exception as exc:
        print(f"tcp_5432=failed {type(exc).__name__}: {exc}")
        print("next_step=Allow this computer's public IP in the RDS security group for PostgreSQL port 5432, or run the backend inside the same AWS VPC.")
        return 1
    finally:
        sock.close()

    try:
        asyncio.run(check_sqlalchemy(database_url, timeout_seconds))
    except Exception as exc:
        print(f"sqlalchemy_select_1=failed {type(exc).__name__}: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
