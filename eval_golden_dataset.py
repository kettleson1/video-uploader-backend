#!/usr/bin/env python3
"""Run the DAVE golden dataset against a backend API."""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, parse, request


DONE_STATUSES = {"done", "completed", "complete", "processed"}
FAILED_STATUSES = {"error", "failed", "failure"}
NO_FOUL_LABELS = {"", "none", "no_foul", "no foul", "no-foul", "no_penalty", "no penalty"}
VALID_RESULTS = {"FOUL", "NO FOUL"}


@dataclass
class GoldenClip:
    clip_id: str
    filename: str
    expected_label: str
    expected_result: str
    rule_reference: str
    notes: str
    video_path: Path


def normalize_label(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    text = text.lower().replace("-", "_").replace(" ", "_")
    if text in {"", "none", "no_foul", "no_penalty"}:
        return "none"
    return text


def result_from_label(value: Any) -> str:
    text = "" if value is None else str(value).strip().lower().replace("_", " ")
    return "NO FOUL" if text in NO_FOUL_LABELS else "FOUL"


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def default_videos_dir() -> Path:
    root = repo_root()
    candidates = [
        root / "golden-dataset" / "videos",
        root.parent / "golden-dataset" / "videos",
    ]
    for candidate in candidates:
        if any(candidate.glob("*.mov")) or any(candidate.glob("*.mp4")):
            return candidate
    return candidates[0]


def load_labels(labels_path: Path, videos_dir: Path) -> list[GoldenClip]:
    clips: list[GoldenClip] = []
    with labels_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "filename", "expected_label", "expected_result", "rule_reference", "notes"}
        missing_columns = required - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"labels.csv is missing columns: {', '.join(sorted(missing_columns))}")

        for row in reader:
            video_path = videos_dir / row["filename"]
            clips.append(
                GoldenClip(
                    clip_id=row["id"],
                    filename=row["filename"],
                    expected_label=row["expected_label"],
                    expected_result=row["expected_result"],
                    rule_reference=row["rule_reference"],
                    notes=row["notes"],
                    video_path=video_path,
                )
            )
    return clips


def validate_clips(clips: list[GoldenClip]) -> None:
    seen_ids: set[str] = set()
    seen_files: set[str] = set()
    duplicate_ids: list[str] = []
    duplicate_files: list[str] = []
    bad_results: list[str] = []

    for clip in clips:
        if clip.clip_id in seen_ids:
            duplicate_ids.append(clip.clip_id)
        seen_ids.add(clip.clip_id)

        if clip.filename in seen_files:
            duplicate_files.append(clip.filename)
        seen_files.add(clip.filename)

        if clip.expected_result.strip().upper() not in VALID_RESULTS:
            bad_results.append(f"{clip.clip_id} {clip.filename}: {clip.expected_result}")

    missing = [clip.filename for clip in clips if not clip.video_path.exists()]
    problems: list[str] = []

    if duplicate_ids:
        problems.append("Duplicate ids:\n" + "\n".join(f"  - {value}" for value in duplicate_ids))
    if duplicate_files:
        problems.append("Duplicate filenames:\n" + "\n".join(f"  - {value}" for value in duplicate_files))
    if bad_results:
        problems.append("Invalid expected_result values:\n" + "\n".join(f"  - {value}" for value in bad_results))
    if missing:
        preview = "\n".join(f"  - {name}" for name in missing[:20])
        extra = "" if len(missing) <= 20 else f"\n  ... and {len(missing) - 20} more"
        problems.append(f"Missing video files:\n{preview}{extra}")

    if problems:
        raise ValueError("\n\n".join(problems))


def http_json(
    method: str,
    url: str,
    api_key: str,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout_seconds: int = 60,
) -> Any:
    req_headers = {"X-DAVE-API-Key": api_key}
    if headers:
        req_headers.update(headers)
    req = request.Request(url, data=body, headers=req_headers, method=method)
    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            payload = response.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed with HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"{method} {url} failed: {exc.reason}") from exc
    return json.loads(payload) if payload else {}


def multipart_body(fields: dict[str, str], file_field: str, file_path: Path) -> tuple[bytes, str]:
    boundary = f"----dave-golden-eval-{uuid.uuid4().hex}"
    chunks: list[bytes] = []

    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                str(value).encode(),
                b"\r\n",
            ]
        )

    content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    chunks.extend(
        [
            f"--{boundary}\r\n".encode(),
            (
                f'Content-Disposition: form-data; name="{file_field}"; '
                f'filename="{file_path.name}"\r\n'
            ).encode(),
            f"Content-Type: {content_type}\r\n\r\n".encode(),
            file_path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def upload_clip(base_url: str, api_key: str, clip: GoldenClip, run_id: str, request_timeout_seconds: int) -> int:
    notes = f"[golden-eval:{run_id}] {clip.clip_id} {clip.filename}. Expected: {clip.expected_result} {clip.expected_label}. {clip.notes}"
    body, content_type = multipart_body(
        fields={"foul_type": clip.expected_label, "notes": notes},
        file_field="file",
        file_path=clip.video_path,
    )
    response = http_json(
        "POST",
        f"{base_url.rstrip('/')}/api/upload",
        api_key,
        body=body,
        headers={"Content-Type": content_type, "Content-Length": str(len(body))},
        timeout_seconds=request_timeout_seconds,
    )
    upload_id = response.get("id")
    if not upload_id:
        raise RuntimeError(f"Upload response did not include an id: {response}")
    return int(upload_id)


def get_play(base_url: str, api_key: str, upload_id: int, request_timeout_seconds: int) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/api/plays?limit=200&offset=0"
    response = http_json("GET", url, api_key, timeout_seconds=request_timeout_seconds)
    if response.get("ok") is False:
        raise RuntimeError(f"Play list failed: {response}")
    for item in response.get("items", []):
        if int(item.get("id")) == upload_id:
            return item
    raise RuntimeError(f"Upload id {upload_id} was not found in the latest /api/plays response")


def wait_for_result(
    base_url: str,
    api_key: str,
    upload_id: int,
    timeout_seconds: int,
    poll_seconds: int,
    request_timeout_seconds: int,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_play: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        play = get_play(base_url, api_key, upload_id, request_timeout_seconds)
        last_play = play
        status = str(play.get("status") or "").lower()
        if status in DONE_STATUSES or status in FAILED_STATUSES:
            return play
        time.sleep(poll_seconds)
    raise TimeoutError(f"Timed out waiting for upload {upload_id}; last status was {last_play}")


def score_clip(clip: GoldenClip, play: dict[str, Any]) -> dict[str, Any]:
    predicted_label = play.get("prediction_label")
    expected_label_norm = normalize_label(clip.expected_label)
    predicted_label_norm = normalize_label(predicted_label)
    expected_result = clip.expected_result.strip().upper()
    predicted_result = result_from_label(predicted_label)

    result_correct = expected_result == predicted_result
    label_correct = expected_label_norm == predicted_label_norm

    return {
        "clip_id": clip.clip_id,
        "filename": clip.filename,
        "upload_id": play.get("id"),
        "status": play.get("status"),
        "expected_result": expected_result,
        "predicted_result": predicted_result,
        "result_correct": result_correct,
        "expected_label": expected_label_norm,
        "predicted_label": predicted_label_norm,
        "label_correct": label_correct,
        "overall_correct": result_correct and label_correct,
        "rule_reference": clip.rule_reference,
        "confidence": play.get("confidence"),
        "error_message": play.get("error_message"),
        "explanation": play.get("explanation"),
        "notes": clip.notes,
    }


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "clip_id",
        "filename",
        "upload_id",
        "status",
        "expected_result",
        "predicted_result",
        "result_correct",
        "expected_label",
        "predicted_label",
        "label_correct",
        "overall_correct",
        "rule_reference",
        "confidence",
        "error_message",
        "explanation",
        "notes",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, Any]], report_path: Path) -> None:
    total = len(rows)
    overall_hits = sum(1 for row in rows if row["overall_correct"])
    result_hits = sum(1 for row in rows if row["result_correct"])
    label_hits = sum(1 for row in rows if row["label_correct"])
    misses = [row for row in rows if not row["overall_correct"]]

    def pct(value: int) -> str:
        return f"{(value / total * 100):.1f}%" if total else "0.0%"

    print("\nGolden dataset eval complete")
    print(f"Total plays: {total}")
    print(f"Correct: {overall_hits}")
    print(f"Accuracy: {pct(overall_hits)}")
    print(f"Result accuracy: {result_hits}/{total} ({pct(result_hits)})")
    print(f"Label accuracy:  {label_hits}/{total} ({pct(label_hits)})")
    print(f"Report: {report_path}")
    if misses:
        print("\nMisses:")
        for row in misses:
            print(
                f"- {row['clip_id']} {row['filename']}: "
                f"expected {row['expected_result']} / {row['expected_label']}, "
                f"got {row['predicted_result']} / {row['predicted_label']}"
            )
    else:
        print("\nNo misses.")


def parse_args() -> argparse.Namespace:
    default_report = repo_root() / "golden-dataset" / "reports" / (
        f"golden_eval_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.csv"
    )
    parser = argparse.ArgumentParser(description="Upload and score the DAVE golden dataset.")
    parser.add_argument("--base-url", default=os.getenv("DAVE_API_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--api-key", default=os.getenv("DAVE_API_KEY"))
    parser.add_argument("--labels", type=Path, default=repo_root() / "golden-dataset" / "labels.csv")
    parser.add_argument("--videos-dir", type=Path, default=default_videos_dir())
    parser.add_argument("--report", type=Path, default=default_report)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument(
        "--request-timeout-seconds",
        type=int,
        default=300,
        help="HTTP request timeout for large video uploads and polling requests.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N clips.")
    parser.add_argument("--ids", default="", help="Comma-separated clip ids to run, for example: 001,007,025.")
    parser.add_argument("--validate-only", action="store_true", help="Check labels/videos without uploading.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.api_key and not args.validate_only:
        print("DAVE_API_KEY is required. Set it in the environment or pass --api-key.", file=sys.stderr)
        return 2

    clips = load_labels(args.labels, args.videos_dir)
    if args.ids:
        requested = {value.strip() for value in args.ids.split(",") if value.strip()}
        clips = [clip for clip in clips if clip.clip_id in requested]
        missing_ids = requested - {clip.clip_id for clip in clips}
        if missing_ids:
            print(f"Unknown clip ids: {', '.join(sorted(missing_ids))}", file=sys.stderr)
            return 2
    if args.limit:
        clips = clips[: args.limit]
    validate_clips(clips)

    print(f"Labels: {args.labels}")
    print(f"Videos: {args.videos_dir}")
    print(f"Clips:  {len(clips)}")

    if args.validate_only:
        print("Validation passed. No uploads were run.")
        return 0

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rows: list[dict[str, Any]] = []
    for index, clip in enumerate(clips, start=1):
        print(f"\n[{index}/{len(clips)}] Uploading {clip.filename}")
        upload_id = upload_clip(args.base_url, args.api_key, clip, run_id, args.request_timeout_seconds)
        print(f"Upload id {upload_id}; waiting for result...")
        play = wait_for_result(
            args.base_url,
            args.api_key,
            upload_id,
            args.timeout_seconds,
            args.poll_seconds,
            args.request_timeout_seconds,
        )
        scored = score_clip(clip, play)
        rows.append(scored)
        mark = "PASS" if scored["result_correct"] and scored["label_correct"] else "MISS"
        print(
            f"{mark}: expected {scored['expected_result']} / {scored['expected_label']}; "
            f"got {scored['predicted_result']} / {scored['predicted_label']}"
        )
        write_report(args.report, rows)

    print_summary(rows, args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
