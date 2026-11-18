"""Run the deterministic comparative evaluation (default: 100 test samples).

Usage (repository root):
    python backend/tools/run_evaluation.py
    python backend/tools/run_evaluation.py --count 100 --seed 20261118
"""

from __future__ import annotations

import argparse
import os
import sys

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(BACKEND_DIR)
sys.path.insert(0, BACKEND_DIR)

from app.config import Settings  # noqa: E402
from app.services.evaluation import (  # noqa: E402
    DEFAULT_COUNT,
    DEFAULT_SEED,
    DEFAULT_SPLIT,
    evaluate_samples,
    select_evaluation_samples,
    write_evaluation_outputs,
)
from app.services.model_registry import MODEL_IDS, ModelRegistry  # noqa: E402
from app.services.sample_catalog import load_catalog  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministic model evaluation.")
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--out",
        default=None,
        help="default: artifacts/evaluation/<count>-<split>-v1",
    )
    parser.add_argument("--version", default=None)
    parser.add_argument(
        "--allow-non-test",
        action="store_true",
        help="permit a split other than test (not used for reported results)",
    )
    args = parser.parse_args(argv)

    if args.split != "test" and not args.allow_non_test:
        raise SystemExit("Refusing a non-test split without --allow-non-test.")

    version = args.version or f"{args.count}-{args.split}-v1"
    out_dir = args.out or os.path.join(
        REPO_ROOT, "artifacts", "evaluation", version
    )

    settings = Settings.default()
    catalog = load_catalog(str(settings.manifest_path), str(settings.data_root))
    registry = ModelRegistry(str(settings.checkpoint_dir))

    sample_ids = select_evaluation_samples(
        catalog, split=args.split, count=args.count, seed=args.seed
    )
    print(f"split={args.split} selected={len(sample_ids)} seed={args.seed}")
    print(f"first={sample_ids[0]} last={sample_ids[-1]}")

    result = evaluate_samples(
        catalog,
        registry,
        sample_ids,
        version=version,
        seed=args.seed,
    )
    paths = write_evaluation_outputs(result, out_dir)

    summary = result["summary"]
    print("\nMODEL  MICRO precision / recall / F1 / IoU / accuracy | TP / FP / FN / TN | predicted")
    for model_id in MODEL_IDS:
        model = summary["models"][model_id]
        micro = model["micro"]
        counts = model["counts"]
        print(
            f"{model_id:>4}  {micro['precision']:.4f} / {micro['recall']:.4f} / "
            f"{micro['f1_dice']:.4f} / {micro['iou']:.4f} / {micro['accuracy']:.4f} | "
            f"{counts['true_positive']} / {counts['false_positive']} / "
            f"{counts['false_negative']} / {counts['true_negative']} | "
            f"{model['predicted']['total']}"
        )

    print("\nRANKINGS")
    for name, ranking in summary["rankings"].items():
        print(f"  {name}: {', '.join(ranking['winners'])}")

    print("\nSUMMARY")
    for sentence in summary["summary"]:
        print(f"  {sentence}")

    print(f"\nvalid samples={summary['counts']['valid_samples']} "
          f"all_unknown={summary['counts']['all_unknown_samples']}")
    print(f"elapsed: {summary['elapsed_seconds']:.2f}s")
    for label, path in paths.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
