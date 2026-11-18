import os
import sys
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(BACKEND_DIR)
sys.path.insert(0, BACKEND_DIR)

from app.services.sample_catalog import build_manifest  # noqa: E402


def make_fire_mask(fire_pixels: int, unknown_pixels: int = 0) -> np.ndarray:
    """64x64 FireMask with fire at the start and unknowns at the end."""
    assert 0 <= fire_pixels <= 4096 - unknown_pixels
    flat = np.zeros(4096, dtype=np.float32)
    if fire_pixels:
        flat[:fire_pixels] = 1.0
    if unknown_pixels:
        flat[-unknown_pixels:] = -1.0
    return flat.reshape(64, 64)


def write_shard(path: str, masks: list[np.ndarray]) -> None:
    with tf.io.TFRecordWriter(str(path)) as writer:
        for mask in masks:
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "FireMask": tf.train.Feature(
                            float_list=tf.train.FloatList(
                                value=mask.reshape(-1).tolist()
                            )
                        )
                    }
                )
            )
            writer.write(example.SerializeToString())


@pytest.fixture()
def dataset(tmp_path):
    """Tiny two-shard dataset plus its manifest."""
    root = tmp_path / "data"
    root.mkdir()

    write_shard(
        str(root / "next_day_wildfire_spread_train_00.tfrecord"),
        [
            make_fire_mask(0),
            make_fire_mask(2),
            make_fire_mask(0, unknown_pixels=96),
            make_fire_mask(1, unknown_pixels=4095),
            make_fire_mask(3),
        ],
    )
    write_shard(
        str(root / "next_day_wildfire_spread_eval_00.tfrecord"),
        [make_fire_mask(0), make_fire_mask(1, unknown_pixels=10), make_fire_mask(4)],
    )

    manifest = build_manifest(str(root))
    return {"root": root, "manifest": manifest}


@pytest.fixture()
def shard_writer():
    """Helper pair for rewriting a fixture shard in place."""
    return write_shard, make_fire_mask


@pytest.fixture(scope="session")
def real_paths():
    return {
        "root": os.path.join(REPO_ROOT, "archive"),
        "manifest": os.path.join(REPO_ROOT, "artifacts", "sample_manifest.json"),
        "phase0_report": os.path.join(
            REPO_ROOT, "artifacts", "reference", "phase0_report.json"
        ),
        "phase0_npz": os.path.join(
            REPO_ROOT, "artifacts", "reference", "phase0_reference.npz"
        ),
    }


@pytest.fixture(scope="session")
def real_settings(real_paths):
    from app.config import Settings

    for path in (real_paths["manifest"],):
        if not os.path.isfile(path):
            pytest.skip(f"required artifact missing: {path}")
    if not os.path.isdir(real_paths["root"]):
        pytest.skip(f"required data root missing: {real_paths['root']}")

    return Settings(
        repo_root=Path(REPO_ROOT),
        data_root=Path(real_paths["root"]),
        manifest_path=Path(real_paths["manifest"]),
        checkpoint_dir=Path(REPO_ROOT),
    )


@pytest.fixture(scope="module")
def api_client(real_settings):
    from fastapi.testclient import TestClient

    from app.main import create_app

    with TestClient(create_app(real_settings)) as client:
        yield client
