"""Measure actual model/inference latency. No caching is added by this script.

Usage (repository root):
    python backend/tools/benchmark_inference.py --runs 5
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(BACKEND_DIR)
sys.path.insert(0, BACKEND_DIR)

from app.config import Settings  # noqa: E402
from app.services.model_registry import MODEL_IDS, ModelRegistry  # noqa: E402
from app.services.preprocessing import (  # noqa: E402
    build_model_input,
    preprocess_features,
)
from app.services.sample_catalog import load_catalog  # noqa: E402


def summary(samples_ms: list[float]) -> str:
    return (
        f"median={statistics.median(samples_ms):7.1f}ms "
        f"mean={statistics.mean(samples_ms):7.1f}ms "
        f"min={min(samples_ms):7.1f}ms max={max(samples_ms):7.1f}ms"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--sample", default="eval_00_000027")
    args = parser.parse_args(argv)

    settings = Settings.default()
    print(f"device           : {'cuda' if __import__('torch').cuda.is_available() else 'cpu'}")
    print(f"sample           : {args.sample}")
    print(f"runs per model   : {args.runs} (after one warmup)")

    load_started = time.perf_counter()
    registry = ModelRegistry(str(settings.checkpoint_dir))
    load_ms = (time.perf_counter() - load_started) * 1000
    print(f"checkpoint load  : {load_ms:.1f}ms ({', '.join(registry.ready_model_ids)})")

    catalog = load_catalog(str(settings.manifest_path), str(settings.data_root))

    read_started = time.perf_counter()
    raw = catalog.load_features(args.sample)
    read_ms = (time.perf_counter() - read_started) * 1000
    print(f"TFRecord read    : {read_ms:.1f}ms (includes manifest integrity check)")

    prep_started = time.perf_counter()
    features = preprocess_features(raw)
    prep_ms = (time.perf_counter() - prep_started) * 1000
    print(f"preprocessing    : {prep_ms:.2f}ms (all 13 features)")

    for model_id in MODEL_IDS:
        if model_id not in registry.ready_model_ids:
            print(f"{model_id} forward       : unavailable")
            continue
        model_input = build_model_input(features, model_id)
        registry.predict(model_id, model_input)  # warmup
        timings = []
        for _ in range(args.runs):
            started = time.perf_counter()
            registry.predict(model_id, model_input)
            timings.append((time.perf_counter() - started) * 1000)
        print(f"{model_id} forward       : {summary(timings)}")

    # End-to-end request timing (in-process TestClient: includes validation and JSON).
    from fastapi.testclient import TestClient

    from app.main import create_app

    with TestClient(create_app(settings)) as client:
        payload = {"sample_id": args.sample, "models": list(MODEL_IDS)}
        client.post("/api/predictions", json=payload)  # warmup
        request_timings = []
        for _ in range(max(1, args.runs - 2)):
            started = time.perf_counter()
            response = client.post("/api/predictions", json=payload)
            request_timings.append((time.perf_counter() - started) * 1000)
            if response.status_code != 200:
                raise SystemExit(f"benchmark request failed: {response.status_code}")
        print(f"POST /predictions: {summary(request_timings)} (3 models, JSON in/out)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
