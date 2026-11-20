"""HTTP integration check. Point DAVE_PRIVACY_TEST_DATABASE_URL at a disposable
Postgres database containing the DAVE schema. Storage and AI use no network.
"""
import base64
import hashlib
import hmac
import json
import os
import secrets
import time
import unittest
from unittest.mock import patch


def member_headers(subject, key):
    def encode(value):
        return base64.urlsafe_b64encode(value).decode().rstrip("=")

    payload = encode(json.dumps({"v": 1, "sub": subject, "exp": int(time.time()) + 900}).encode())
    signature = encode(hmac.digest(key.encode(), payload.encode(), hashlib.sha256))
    return {"Authorization": f"Bearer {payload}.{signature}"}


@unittest.skipUnless(os.getenv("DAVE_PRIVACY_TEST_DATABASE_URL"), "requires a disposable PostgreSQL database")
class MemberFilmTests(unittest.TestCase):
    def test_only_uploader_can_see_and_review_game(self):
        from fastapi.testclient import TestClient

        key = secrets.token_hex(32)
        config = dict(
            DATABASE_URL=os.environ["DAVE_PRIVACY_TEST_DATABASE_URL"],
            DAVE_API_KEY=key, OPENAI_API_KEY="test", AWS_S3_BUCKET="member-test",
            AWS_ACCESS_KEY_ID="local", AWS_SECRET_ACCESS_KEY="local",
        )
        with patch.dict(os.environ, config):
            import main
            self.assertEqual(main.DATABASE_URL, config["DATABASE_URL"])
            with TestClient(main.app) as client, patch.object(main.s3_client, "upload_fileobj"):
                from database import engine

                try:
                    self.check_reviews(client, key)
                finally:
                    client.portal.call(engine.dispose)

    def check_reviews(self, client, key):
        owner, other = [member_headers(secrets.token_hex(16), key) for _ in range(2)]
        games = "/api/games"
        self.assertEqual(client.get(games).status_code, 401)
        for headers in (owner, other):
            response = client.post(games + "/upload", headers=headers, files={"file": ("film.mp4", b"fixture")})
            self.assertEqual(response.status_code, 200)
            listing = client.get(games, headers=headers)
            self.assertIn("no-store", listing.headers["Cache-Control"])
            self.assertEqual(listing.json()["total"], 1)
            self.assertEqual(listing.json()["items"][0]["id"], response.json()["id"])
        game = client.get(games, headers=owner).json()["items"][0]
        path = f"{games}/{game['id']}"
        segments = {"segments": [{"start_time": 0, "end_time": 10}]}
        self.assertEqual(client.patch(path + "/segments", headers=owner, json=segments).status_code, 200)
        play = client.get(path + "/timeline", headers=owner).json()["items"][0]
        findings = f"/api/plays/{play['id']}/findings"
        label = {"disposition": "UNCERTAIN", "foul_label": "holding"}
        finding = client.post(findings, headers=owner, json=label).json()["finding"]
        review = f"/api/play-findings/{finding['id']}/review"
        for route in (path, path + "/timeline", path + "/plays", path + "/segmentation-comparison"):
            self.assertEqual(client.get(route, headers=other).status_code, 404)
        self.assertEqual(client.get("/api/plays", headers=other).json()["total"], 0)
        actions = [
            ("PATCH", path + "/segments", segments),
            ("POST", path + "/analyze", {}),
            ("POST", findings, label),
            ("PATCH", review, {"review_status": "dismissed"}),
            ("PATCH", f"/api/plays/{play['legacy_upload_id']}/review", {"reviewed": True}),
        ]
        for method, route, payload in actions:
            with self.subTest(route=route):
                response = client.request(method, route, headers=other, json=payload)
                self.assertEqual(response.status_code, 404)
        response = client.patch(review, headers=owner, json={"review_status": "dismissed"})
        self.assertEqual(response.status_code, 200)
        saved = client.get(path + "/timeline", headers=owner).json()["items"][0]
        self.assertEqual(saved["findings"][0]["review_status"], "dismissed")
