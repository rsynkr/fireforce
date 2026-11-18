"""Build the versioned sample manifest from the canonical TFRecord shards.

Usage (from the repository root):
    python backend/tools/build_manifest.py --data-root archive \
        --out artifacts/sample_manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(BACKEND_DIR)
sys.path.insert(0, BACKEND_DIR)

from app.services.sample_catalog import (  # noqa: E402
    PIXELS_PER_RECORD,
    SPLITS,
    build_manifest,
    save_manifest,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the deterministic sample manifest.")
    parser.add_argument(
        "--data-root",
        default=os.path.join(REPO_ROOT, "archive"),
        help="directory containing next_day_wildfire_spread_*.tfrecord shards",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(REPO_ROOT, "artifacts", "sample_manifest.json"),
        help="manifest output path",
    )
    parser.add_argument(
        "--dataset-name",
        default="next-day-wildfire-spread",
        help="dataset_name recorded in the manifest",
    )
    args = parser.parse_args(argv)

    started = time.perf_counter()
    manifest = build_manifest(args.data_root, dataset_name=args.dataset_name)
    elapsed = time.perf_counter() - started

    save_manifest(manifest, args.out)

    shard_counts = {split: 0 for split in SPLITS}
    record_counts = {split: 0 for split in SPLITS}
    for shard in manifest["shards"]:
        shard_counts[shard["split"]] += 1
        record_counts[shard["split"]] += shard["record_count"]

    print(f"data root : {os.path.abspath(args.data_root)}")
    print(f"manifest  : {os.path.abspath(args.out)}")
    print(f"scan time : {elapsed:.2f}s")
    for split in SPLITS:
        print(
            f"{split:5s}     : {shard_counts[split]} shards, "
            f"{record_counts[split]} records"
        )
    print(f"total     : {len(manifest['shards'])} shards, "
          f"{len(manifest['samples'])} samples")
    print(f"fingerprint: {manifest['dataset_fingerprint']}")
    print(f"output size: {os.path.getsize(args.out) / 1e6:.2f} MB")
    print(f"sample rows: {len(manifest['samples'])} x 6 fields "
          f"({PIXELS_PER_RECORD} pixels per record)")
    print(json.dumps(manifest["samples"][0], indent=2))
    print(json.dumps(manifest["samples"][-1], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
