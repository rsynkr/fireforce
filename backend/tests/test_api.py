import json
import os
import shutil
from dataclasses import replace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


@pytest.fixture(scope="module")
def reference(real_paths):
    if not os.path.isfile(real_paths["phase0_npz"]):
        pytest.skip("Phase 0 reference not available")
    with open(real_paths["phase0_report"], encoding="utf-8") as handle:
        report = json.load(handle)
    return report, np.load(real_paths["phase0_npz"])


@pytest.fixture(scope="module")
def partial_client(real_settings, tmp_path_factory):
    checkpoint_dir = tmp_path_factory.mktemp("partial-checkpoints")
    shutil.copy2(
        real_settings.checkpoint_dir / "model_1_best.pth",
        checkpoint_dir / "model_1_best.pth",
    )
    settings = replace(real_settings, checkpoint_dir=checkpoint_dir)
    with TestClient(create_app(settings)) as client:
        yield client


# ------------------------------------------------------------------- health


def test_health_reports_data_models_and_device(api_client):
    response = api_client.get("/api/health")
    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "ok"
    assert body["device"] in {"cpu", "cuda"}
    assert body["data"]["ready"] is True
    assert body["data"]["manifest_version"] == 1
    assert body["data"]["shard_count"] == 19
    assert body["data"]["sample_count"] == 18_545
    assert body["data"]["split_counts"] == {
        "train": 14_979,
        "eval": 1_877,
        "test": 1_689,
    }
    assert set(body["models"]) == {"M1", "M2", "M3"}
    assert {m["threshold"] for m in body["models"].values()} == {0.93, 0.94}
    assert "validation-selected" in body["threshold_provenance"].lower()
    assert "not re-run" in body["threshold_provenance"].lower()


# ---------------------------------------------------------------- samples


def test_list_samples_defaults_and_pagination(api_client):
    response = api_client.get("/api/samples")
    assert response.status_code == 200
    body = response.json()

    assert body["total"] == 18_545
    assert body["limit"] == 50
    assert body["offset"] == 0
    assert len(body["items"]) == 50
    assert body["items"][0]["sample_id"] == "train_00_000000"
    assert body["items"][0]["target_unknown_pixels"] == 0


def test_list_samples_split_filter_and_search(api_client):
    eval_page = api_client.get("/api/samples", params={"split": "eval"}).json()
    assert eval_page["total"] == 1_877
    assert eval_page["items"][0]["split"] == "eval"

    page = api_client.get(
        "/api/samples",
        params={"split": "eval", "offset": 27, "limit": 1},
    ).json()
    assert page["total"] == 1_877
    assert page["items"][0]["sample_id"] == "eval_00_000027"

    search = api_client.get(
        "/api/samples", params={"search": "eval_00_000027"}
    ).json()
    assert search["total"] == 1
    assert search["items"][0]["sample_id"] == "eval_00_000027"

    empty = api_client.get("/api/samples", params={"search": "no-such-id"}).json()
    assert empty["total"] == 0
    assert empty["items"] == []


def test_list_samples_fire_filter_combines_with_split_search_pagination(api_client):
    response = api_client.get(
        "/api/samples", params={"min_fire_pixels": 101, "limit": 10, "offset": 0}
    )
    body = response.json()
    assert response.status_code == 200
    assert body["total"] == 2377
    assert len(body["items"]) == 10
    assert all(item["target_fire_pixels"] > 100 for item in body["items"])

    for split, expected in (("train", 1799), ("eval", 328), ("test", 250)):
        split_body = api_client.get(
            "/api/samples", params={"split": split, "min_fire_pixels": 101}
        ).json()
        assert split_body["total"] == expected
        assert all(item["split"] == split for item in split_body["items"])
        assert all(item["target_fire_pixels"] > 100 for item in split_body["items"])

    # Exact boundary: 100 excluded, 101 included.
    at_100 = api_client.get(
        "/api/samples",
        params={"search": "train_00_000160", "min_fire_pixels": 101},
    ).json()
    assert at_100["total"] == 0
    without_filter = api_client.get(
        "/api/samples", params={"search": "train_00_000160"}
    ).json()
    assert without_filter["total"] == 1

    at_101 = api_client.get(
        "/api/samples",
        params={"search": "train_00_000844", "min_fire_pixels": 101},
    ).json()
    assert at_101["total"] == 1
    assert at_101["items"][0]["target_fire_pixels"] == 101

    # Pagination is applied after filtering.
    last_page = api_client.get(
        "/api/samples", params={"min_fire_pixels": 101, "limit": 10, "offset": 2370}
    ).json()
    assert last_page["total"] == 2377
    assert len(last_page["items"]) == 7
    assert all(item["target_fire_pixels"] > 100 for item in last_page["items"])

    # Combined split + search + filter.
    combined = api_client.get(
        "/api/samples",
        params={
            "split": "eval",
            "search": "eval_00_000014",
            "min_fire_pixels": 101,
        },
    ).json()
    assert combined["total"] == 1
    assert combined["items"][0]["target_fire_pixels"] == 319


def test_list_samples_fire_filter_rejects_negative(api_client):
    assert (
        api_client.get("/api/samples", params={"min_fire_pixels": -1}).status_code
        == 422
    )


@pytest.mark.parametrize(
    "params",
    [
        {"limit": 0},
        {"limit": 501},
        {"offset": -1},
        {"split": "validation"},
    ],
)
def test_list_samples_rejects_bad_query(api_client, params):
    assert api_client.get("/api/samples", params=params).status_code == 422


def test_sample_detail(api_client):
    body = api_client.get("/api/samples/eval_00_000027").json()
    assert body["sample_id"] == "eval_00_000027"
    assert body["split"] == "eval"
    assert body["record_index"] == 27
    assert body["target_valid_pixels"] == 4096
    assert body["target_fire_pixels"] == 21
    assert body["target_unknown_pixels"] == 0
    assert body["all_unknown"] is False
    assert len(body["available_features"]) == 13
    assert body["models"] == ["M1", "M2", "M3"]

    previous_fire = np.array(body["previous_fire_mask"])
    target = np.array(body["target_mask"])
    valid = np.array(body["target_valid_mask"], dtype=bool)
    assert previous_fire.shape == (64, 64)
    assert target.shape == (64, 64)
    assert valid.shape == (64, 64)
    assert set(np.unique(previous_fire)) <= {-1, 0, 1}
    assert set(np.unique(target)) <= {0, 1}
    # Values cross-checked against phase0_report.json for record 27.
    assert int((previous_fire == 1).sum()) == 5
    assert int((previous_fire == -1).sum()) == 13
    assert int(target.sum()) == body["target_fire_pixels"] == 21
    assert int(valid.sum()) == body["target_valid_pixels"] == 4096

    unknown = api_client.get("/api/samples/eval_00_000009").json()
    assert unknown["target_valid_pixels"] == 0
    assert unknown["target_unknown_pixels"] == 4096
    assert unknown["all_unknown"] is True
    assert not any(
        any(row) for row in unknown["target_valid_mask"]
    )
    assert not any(any(row) for row in unknown["target_mask"])


def test_sample_detail_errors(api_client):
    assert api_client.get("/api/samples/not-an-id").status_code == 422
    assert api_client.get("/api/samples/eval_99_000000").status_code == 404


# -------------------------------------------------------------- predictions


def _predict(client, sample_id, models=None):
    payload = {"sample_id": sample_id}
    if models is not None:
        payload["models"] = models
    return client.post("/api/predictions", json=payload)


def test_predictions_all_models_match_phase0(api_client, reference):
    report, ref = reference
    response = _predict(api_client, "eval_00_000027")
    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "ok"
    assert body["sample_id"] == "eval_00_000027"
    assert body["device"] in {"cpu", "cuda"}
    assert body["duration_ms"] > 0
    assert len(body["models"]) == 3

    assert body["ground_truth"]["available"] is True
    assert body["ground_truth"]["valid_pixels"] == 4096
    assert body["ground_truth"]["fire_pixels"] == 21
    assert body["ground_truth"]["unknown_pixels"] == 0
    assert np.array(body["ground_truth"]["valid_mask"], dtype=bool).all()

    for entry in body["models"]:
        model_id = entry["model_id"]
        assert entry["status"] == "ok"
        assert entry["threshold"] == report["samples"]["fire"]["models"][model_id]["threshold"]
        assert entry["checkpoint"]["sha256"] == report["checkpoints"][model_id]["sha256"]
        assert entry["inference_ms"] is not None and entry["inference_ms"] >= 0

        probabilities = np.array(entry["probabilities"], dtype=np.float32)
        assert probabilities.shape == (64, 64)
        assert np.isfinite(probabilities).all()
        assert np.array_equal(probabilities, ref[f"fire_{model_id}_probs"])

        expected_mask = probabilities >= entry["threshold"]
        assert np.array_equal(np.array(entry["mask"], dtype=bool), expected_mask)
        assert entry["predicted_fire_pixels"] == int(expected_mask.sum())

        expected = report["samples"]["fire"]["models"][model_id]["metrics_masked"]
        metrics = entry["metrics"]
        assert metrics["available"] is True
        assert metrics["valid_pixels"] == 4096
        assert metrics["true_positive"] == expected["TP"]
        assert metrics["false_positive"] == expected["FP"]
        assert metrics["false_negative"] == expected["FN"]
        assert metrics["true_negative"] == expected["TN"]
        for key in ("dice", "iou", "precision", "recall"):
            assert metrics[key] == pytest.approx(expected[key], rel=0, abs=1e-12)


def test_predictions_are_deterministic(api_client):
    first = _predict(api_client, "eval_00_000027", ["M2"]).json()
    second = _predict(api_client, "eval_00_000027", ["M2"]).json()

    assert np.array_equal(
        np.array(first["models"][0]["probabilities"], dtype=np.float32),
        np.array(second["models"][0]["probabilities"], dtype=np.float32),
    )
    assert first["models"][0]["mask"] == second["models"][0]["mask"]


def test_predictions_all_unknown_target(api_client):
    body = _predict(api_client, "eval_00_000009").json()
    assert body["status"] == "ok"
    assert body["ground_truth"]["available"] is False
    assert body["ground_truth"]["valid_pixels"] == 0
    assert body["ground_truth"]["unknown_pixels"] == 4096

    for entry in body["models"]:
        assert entry["status"] == "ok"
        assert entry["metrics"] == {
            "available": False,
            "reason": "no_valid_target_pixels",
            "valid_pixels": 0,
            "true_positive": None,
            "false_positive": None,
            "false_negative": None,
            "true_negative": None,
            "precision": None,
            "recall": None,
            "dice": None,
            "iou": None,
        }


def test_predictions_single_model(api_client):
    body = _predict(api_client, "eval_00_000027", ["M1"]).json()
    assert len(body["models"]) == 1
    assert body["models"][0]["model_id"] == "M1"


@pytest.mark.parametrize(
    "payload",
    [
        {"sample_id": "not-an-id"},
        {"sample_id": "eval_00_000027", "models": ["M4"]},
        {"sample_id": "eval_00_000027", "models": []},
        {"sample_id": "eval_00_000027", "models": ["M1", "M1"]},
    ],
)
def test_predictions_reject_invalid_payload(api_client, payload):
    assert api_client.post("/api/predictions", json=payload).status_code == 422


def test_predictions_unknown_sample_is_404(api_client):
    assert _predict(api_client, "eval_99_000000").status_code == 404
    assert _predict(api_client, "train_99_000000").status_code == 404


# ---------------------------------------------------------- failure modes


def test_partial_model_failure_is_explicit(partial_client):
    health = partial_client.get("/api/health").json()
    assert health["status"] == "degraded"
    assert health["models"]["M1"]["status"] == "ready"
    assert health["models"]["M2"]["status"] == "unavailable"
    assert "not found" in health["models"]["M2"]["error"].lower()

    body = _predict(partial_client, "eval_00_000027", ["M1", "M2"]).json()
    assert body["status"] == "partial"
    assert body["models"][0]["status"] == "ok"
    assert body["models"][0]["probabilities"] is not None
    assert body["models"][1]["status"] == "unavailable"
    assert body["models"][1]["probabilities"] is None
    assert body["models"][1]["metrics"] is None
    assert body["models"][1]["error"]


def test_all_models_unavailable_is_503(partial_client):
    response = _predict(partial_client, "eval_00_000027", ["M2"])
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["models"][0]["status"] == "unavailable"


def test_missing_manifest_is_service_unavailable(real_settings, tmp_path):
    settings = replace(real_settings, manifest_path=tmp_path / "missing.json")
    with TestClient(create_app(settings)) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["status"] == "unavailable"
        assert health.json()["data"]["ready"] is False

        assert client.get("/api/samples").status_code == 503
        assert client.get("/api/samples/eval_00_000027").status_code == 503
        assert _predict(client, "eval_00_000027").status_code == 503


def test_train_00_000000_is_a_genuine_model_miss(api_client):
    """Regression guard for the investigated sample: correct data, real miss."""
    detail = api_client.get("/api/samples/train_00_000000").json()
    assert detail["target_fire_pixels"] == 7
    assert detail["target_valid_pixels"] == 4096

    body = _predict(api_client, "train_00_000000").json()
    assert body["ground_truth"]["fire_pixels"] == 7
    assert body["ground_truth"]["valid_pixels"] == 4096
    expected_predicted = {"M1": 51, "M2": 16, "M3": 16}
    for model in body["models"]:
        assert model["status"] == "ok"
        assert model["predicted_fire_pixels"] == expected_predicted[model["model_id"]]
        assert model["metrics"]["true_positive"] == 0
        assert model["metrics"]["dice"] == 0.0


def test_evaluation_summary_endpoint(api_client):
    response = api_client.get("/api/evaluation/summary")
    assert response.status_code == 200
    body = response.json()
    assert body["split"] == "test"
    assert body["sampling"]["count"] == 100
    assert set(body["models"]) == {"M1", "M2", "M3"}
    assert body["summary"]
    assert body["rankings"]["best_f1_dice"]["winners"]


def test_missing_evaluation_summary_is_503(real_settings, tmp_path):
    settings = replace(
        real_settings, evaluation_summary_path=tmp_path / "missing-summary.json"
    )
    with TestClient(create_app(settings)) as client:
        response = client.get("/api/evaluation/summary")
    assert response.status_code == 503
    assert "run_evaluation" in response.json()["detail"]


def test_cors_allows_only_the_local_frontend_origin(api_client):
    allowed = api_client.options(
        "/api/predictions",
        headers={
            "Origin": "http://127.0.0.1:5173",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert allowed.status_code == 200
    assert (
        allowed.headers.get("access-control-allow-origin")
        == "http://127.0.0.1:5173"
    )

    denied = api_client.options(
        "/api/predictions",
        headers={
            "Origin": "http://example.invalid",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert denied.headers.get("access-control-allow-origin") is None
