"""HTTP smoke/acceptance checks against a running FastAPI instance.

Usage:
    python backend/tools/smoke_http.py --base http://127.0.0.1:8000
    python backend/tools/smoke_http.py --base http://127.0.0.1:8001 --partial
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import httpx
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VALIDATION_REPORT = os.path.join(
    REPO_ROOT, "artifacts", "validation", "phase5_report.json"
)

CASES = {
    "sparse_fire": "eval_00_000027",
    "dense_fire": "train_11_000326",
    "no_fire": "train_00_000001",
    "all_unknown": "train_00_000060",
    "partial_unknown_dense": "train_06_000286",
}


def check(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)
    print(f"PASS {label}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    base = args.base.rstrip("/")

    with httpx.Client(base_url=base, timeout=60.0) as client:
        health = client.get("/api/health")
        check(health.status_code == 200, "health returns 200")
        body = health.json()
        check(body["data"]["sample_count"] == 18_545, "health reports 18,545 samples")
        check(body["data"]["split_counts"] == {"train": 14_979, "eval": 1_877, "test": 1_689},
              "health split counts")
        check(body["models"]["M1"]["sha256"]
              == "3ec708fa1bca276f4b4d23778a4d5cb2b0e56c14913facac8c06bb589047f15f",
              "M1 checkpoint hash")
        check("validation-selected" in body["threshold_provenance"].lower(),
              "threshold provenance is labelled validation-selected")

        if args.partial:
            check(body["status"] == "degraded", "partial health is degraded")
            check(body["models"]["M1"]["status"] == "ready", "M1 ready")
            check(body["models"]["M2"]["status"] == "unavailable", "M2 unavailable")
            check(body["models"]["M3"]["status"] == "ready", "M3 ready")

            partial = client.post(
                "/api/predictions",
                json={"sample_id": "eval_00_000027", "models": ["M1", "M2"]},
            )
            check(partial.status_code == 200, "partial prediction returns 200")
            payload = partial.json()
            check(payload["status"] == "partial", "partial response status")
            check(payload["models"][0]["status"] == "ok", "partial keeps M1")
            check(payload["models"][0]["metrics"]["available"] is True,
                  "partial keeps M1 metrics")
            check(payload["models"][1]["status"] == "unavailable", "partial marks M2")
            check(payload["models"][1]["metrics"] is None, "partial M2 has no metrics")

            only_bad = client.post(
                "/api/predictions",
                json={"sample_id": "eval_00_000027", "models": ["M2"]},
            )
            check(only_bad.status_code == 503, "all-unavailable request returns 503")
            print("\nPARTIAL CHECKS PASSED")
            return 0

        check(body["status"] == "ok", "health status ok")

        listing = client.get("/api/samples", params={"limit": 50, "offset": 0}).json()
        check(listing["total"] == 18_545 and len(listing["items"]) == 50,
              "samples pagination")
        searched = client.get("/api/samples", params={"search": "eval_00_000027"}).json()
        check(searched["total"] == 1 and searched["items"][0]["sample_id"] == "eval_00_000027",
              "sample search")
        eval_page = client.get("/api/samples", params={"split": "eval"}).json()
        check(eval_page["total"] == 1_877, "split filter")

        detail = client.get("/api/samples/eval_00_000027")
        check(detail.status_code == 200, "sample detail 200")
        detail_body = detail.json()
        check(len(detail_body["previous_fire_mask"]) == 64
              and len(detail_body["target_valid_mask"]) == 64,
              "detail includes 64x64 masks")

        report = json.load(open(VALIDATION_REPORT, encoding="utf-8"))

        for case_name, sample_id in CASES.items():
            response = client.post("/api/predictions", json={"sample_id": sample_id})
            check(response.status_code == 200, f"{case_name}: predictions 200")
            payload = response.json()
            check(payload["sample_id"] == sample_id, f"{case_name}: echoed sample id")
            check(len(payload["models"]) == 3, f"{case_name}: three models")

            expected = report["cases"][case_name]
            check(payload["ground_truth"]["valid_pixels"] == expected["target_valid_pixels"],
                  f"{case_name}: valid pixel count")
            check(payload["ground_truth"]["fire_pixels"] == expected["target_fire_pixels"],
                  f"{case_name}: fire pixel count")

            for model in payload["models"]:
                model_id = model["model_id"]
                probs = np.asarray(model["probabilities"], dtype=np.float32)
                check(probs.shape == (64, 64), f"{case_name}/{model_id}: 64x64 probabilities")
                digest = hashlib.sha256(probs.tobytes()).hexdigest()
                check(digest == expected["models"][model_id]["probabilities_sha256"],
                      f"{case_name}/{model_id}: probabilities match validation report")
                if case_name == "all_unknown":
                    check(model["metrics"]["available"] is False,
                          f"{case_name}/{model_id}: metrics unavailable")
                    check(model["metrics"]["dice"] is None,
                          f"{case_name}/{model_id}: dice is null")
                else:
                    check(model["metrics"]["available"] is True,
                          f"{case_name}/{model_id}: metrics available")
                    check(model["metrics"]["valid_pixels"] == expected["target_valid_pixels"],
                          f"{case_name}/{model_id}: metrics use valid pixels only")

        check(client.get("/api/samples/not-an-id").status_code == 422,
              "malformed sample id returns 422")
        check(client.get("/api/samples/eval_99_000000").status_code == 404,
              "unknown sample id returns 404")
        check(client.post("/api/predictions", json={
            "sample_id": "eval_00_000027", "models": ["M4"]
        }).status_code == 422, "unsupported model returns 422")
        check(client.post("/api/predictions", json={
            "sample_id": "eval_00_000027", "models": []
        }).status_code == 422, "empty model list returns 422")

        cors = client.options(
            "/api/predictions",
            headers={
                "Origin": "http://127.0.0.1:5173",
                "Access-Control-Request-Method": "POST",
            },
        )
        check(cors.headers.get("access-control-allow-origin") == "http://127.0.0.1:5173",
              "CORS allows local dev origin")
        denied = client.options(
            "/api/predictions",
            headers={
                "Origin": "http://example.invalid",
                "Access-Control-Request-Method": "POST",
            },
        )
        check(denied.headers.get("access-control-allow-origin") is None,
              "CORS denies foreign origin")

    print("\nALL HTTP CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
