import json
import os

import numpy as np
import pytest

from app.services.preprocessing import (
    CHANNEL_COUNTS,
    MODEL_CHANNELS,
    MODEL_FEATURES,
    NORMALIZATION_EPSILON,
    PIXELS_PER_RECORD,
    TRAIN_STATS,
    PreprocessingError,
    build_model_input,
    prepare_sample,
    preprocess_features,
)
from app.services.sample_catalog import load_catalog

BASE_FEATURES = {
    "tmmn": 260.0,
    "tmmx": 330.0,
    "pr": 5.0,
    "NDVI": 0.0,
    "pdsi": 10.0,
    "erc": 120.0,
    "elevation": 3000.0,
    "population": 1000.0,
    "sph": 0.02,
    "vs": 8.0,
    "th": 45.0,
}


def make_raw(prev_fire=0.0, fire=0.0, **overrides):
    values = {**BASE_FEATURES, **overrides}
    raw = {
        name: np.full(PIXELS_PER_RECORD, value, dtype=np.float32)
        for name, value in values.items()
    }
    raw["PrevFireMask"] = np.full(PIXELS_PER_RECORD, prev_fire, dtype=np.float32)
    raw["FireMask"] = np.full(PIXELS_PER_RECORD, fire, dtype=np.float32)
    return raw


def z(name, value):
    mean, std = TRAIN_STATS[name]
    return (np.float32(value) - mean) / (std + NORMALIZATION_EPSILON)


# --------------------------------------------------------------- shapes/order


@pytest.mark.parametrize("model,channels", [("M1", 1), ("M2", 8), ("M3", 13)])
def test_shapes_dtype_and_finiteness(model, channels):
    prepared = prepare_sample(make_raw(prev_fire=1.0, fire=1.0), model)

    assert prepared.model == model
    assert prepared.input.shape == (channels, 64, 64)
    assert prepared.input.dtype == np.float32
    assert np.isfinite(prepared.input).all()
    assert prepared.target.shape == (64, 64)
    assert prepared.target.dtype == np.float32
    assert set(np.unique(prepared.target)) <= {0.0, 1.0}
    assert prepared.valid.shape == (64, 64)
    assert prepared.valid.dtype == np.bool_


def test_channel_counts_match_checkpoint_contract():
    assert CHANNEL_COUNTS == {"M1": 1, "M2": 8, "M3": 13}
    assert len(MODEL_CHANNELS["M2"]) == 8
    assert len(MODEL_CHANNELS["M3"]) == 13
    assert MODEL_CHANNELS["M2"][-2:] == ("th_sin", "th_cos")


def test_m2_channel_order_and_values():
    raw = make_raw(prev_fire=1.0, fire=1.0)
    features = preprocess_features(raw)
    x = build_model_input(features, "M2")

    expected = [
        1.0,
        z("tmmn", 260.0),
        z("tmmx", 330.0),
        z("pr", 5.0),
        z("sph", 0.02),
        z("vs", 8.0),
        np.sin(np.deg2rad(np.float32(45.0))),
        np.cos(np.deg2rad(np.float32(45.0))),
    ]
    for channel, value in enumerate(expected):
        np.testing.assert_allclose(x[channel], value, rtol=1e-6, atol=1e-7)


def test_m3_channel_order_and_values():
    raw = make_raw(prev_fire=1.0, fire=1.0)
    x = build_model_input(preprocess_features(raw), "M3")

    theta = np.deg2rad(np.float32(45.0))
    expected = [
        1.0,
        z("tmmn", 260.0),
        z("tmmx", 330.0),
        z("pr", 5.0),
        z("sph", 0.02),
        z("vs", 8.0),
        np.sin(theta),
        np.cos(theta),
        z("NDVI", 0.0),
        z("pdsi", 10.0),
        z("erc", 120.0),
        z("elevation", 3000.0),
        z("population", 1000.0),
    ]
    for channel, value in enumerate(expected):
        np.testing.assert_allclose(x[channel], value, rtol=1e-6, atol=1e-7)


def test_m1_is_previous_fire_only():
    raw = make_raw(prev_fire=1.0)
    x = build_model_input(preprocess_features(raw), "M1")
    assert x.shape == (1, 64, 64)
    assert np.array_equal(x[0], np.ones((64, 64), dtype=np.float32))


def test_m2_only_uses_its_feature_group():
    # Changing a channel that belongs to M3 only must not affect M2.
    raw_a = make_raw(prev_fire=1.0)
    raw_b = make_raw(prev_fire=1.0, NDVI=9999.0, population=9999.0)
    np.testing.assert_array_equal(
        build_model_input(preprocess_features(raw_a), "M2"),
        build_model_input(preprocess_features(raw_b), "M2"),
    )


# ------------------------------------------------------------------ unknowns


def test_prev_fire_minus_one_maps_to_zero_only_in_input():
    prev = np.zeros(PIXELS_PER_RECORD, dtype=np.float32)
    prev[:1000] = -1.0
    prev[1000:2000] = 1.0
    raw = make_raw()
    raw["PrevFireMask"] = prev

    features = preprocess_features(raw)
    assert int((features["PrevFireMask"] == -1).sum()) == 1000

    x = build_model_input(features, "M1")
    assert int((x == -1).sum()) == 0
    assert np.all(x[0].reshape(-1)[:1000] == 0.0)
    assert np.all(x[0].reshape(-1)[1000:2000] == 1.0)


def test_target_unknowns_are_excluded_not_treated_as_non_fire():
    fire = np.zeros(PIXELS_PER_RECORD, dtype=np.float32)
    fire[:10] = 1.0
    fire[10:20] = -1.0
    raw = make_raw(fire=0.0)
    raw["FireMask"] = fire

    prepared = prepare_sample(raw, "M1")
    flat_target = prepared.target.reshape(-1)
    flat_valid = prepared.valid.reshape(-1)

    assert np.all(flat_target[:10] == 1.0)
    assert np.all(flat_target[10:20] == 0.0)
    assert not flat_valid[10:20].any()
    assert flat_valid.sum() == PIXELS_PER_RECORD - 10
    # Raw target is untouched (unknown stays -1).
    assert np.all(raw["FireMask"][10:20] == -1.0)


def test_all_unknown_target():
    raw = make_raw(fire=-1.0)
    prepared = prepare_sample(raw, "M3")
    assert not prepared.valid.any()
    assert np.all(prepared.target == 0.0)
    assert int(prepared.target.sum()) == 0


def test_all_unknown_previous_fire():
    raw = make_raw(prev_fire=-1.0)
    x = build_model_input(preprocess_features(raw), "M1")
    assert np.array_equal(x[0], np.zeros((64, 64), dtype=np.float32))


# -------------------------------------------------------- clipping/normalizing


def test_negative_precipitation_is_clipped_to_zero():
    features = preprocess_features(make_raw(pr=-5.0))
    np.testing.assert_allclose(features["pr"], z("pr", 0.0), rtol=0, atol=1e-7)
    # The clip happens before z-scoring, so the stored value is z(0), not 0.
    assert float(features["pr"][0]) < 0.0


def test_ndvi_clipping_and_boundaries():
    low = preprocess_features(make_raw(NDVI=-3000.0))["NDVI"]
    high = preprocess_features(make_raw(NDVI=20000.0))["NDVI"]
    at_low = preprocess_features(make_raw(NDVI=-2000.0))["NDVI"]
    at_high = preprocess_features(make_raw(NDVI=10000.0))["NDVI"]

    np.testing.assert_allclose(low, z("NDVI", -2000.0), rtol=0, atol=1e-7)
    np.testing.assert_allclose(high, z("NDVI", 10000.0), rtol=0, atol=1e-7)
    np.testing.assert_allclose(low, at_low, rtol=0, atol=0)
    np.testing.assert_allclose(high, at_high, rtol=0, atol=0)


def test_wind_direction_clipping_and_circular_encoding():
    clipped = preprocess_features(make_raw(th=400.0))
    at_bound = preprocess_features(make_raw(th=360.0))
    np.testing.assert_allclose(clipped["th_sin"], at_bound["th_sin"], rtol=0, atol=1e-7)
    np.testing.assert_allclose(clipped["th_cos"], at_bound["th_cos"], rtol=0, atol=1e-7)
    # 360 degrees is one full turn: cos ~ 1, sin ~ 0.
    assert abs(float(at_bound["th_sin"][0])) < 1e-6
    assert abs(float(at_bound["th_cos"][0]) - 1.0) < 1e-6


@pytest.mark.parametrize(
    "degrees,expected_sin,expected_cos",
    [(0.0, 0.0, 1.0), (90.0, 1.0, 0.0), (-90.0, -1.0, 0.0), (180.0, 0.0, -1.0)],
)
def test_wind_uses_raw_degrees(degrees, expected_sin, expected_cos):
    features = preprocess_features(make_raw(th=degrees))
    assert abs(float(features["th_sin"][0]) - expected_sin) < 1e-6
    assert abs(float(features["th_cos"][0]) - expected_cos) < 1e-6


def test_normalization_uses_std_plus_epsilon():
    assert NORMALIZATION_EPSILON == 1e-8
    features = preprocess_features(make_raw(tmmn=260.0))
    expected = (np.float32(260.0) - 281.852110) / (18.497169 + 1e-8)
    np.testing.assert_allclose(features["tmmn"], expected, rtol=0, atol=1e-7)
    # th stays in raw degrees; it is not z-scored before sin/cos.
    th_features = preprocess_features(make_raw(th=90.0))
    np.testing.assert_allclose(th_features["th"], 90.0, rtol=0, atol=0)


# ----------------------------------------------------------- safety/repeatability


def test_inputs_are_not_mutated():
    raw = make_raw(prev_fire=-1.0, fire=-1.0)
    raw["PrevFireMask"] = np.full(PIXELS_PER_RECORD, -1.0, dtype=np.float32)
    raw["FireMask"][:5] = 1.0
    before = {name: values.copy() for name, values in raw.items()}

    for model in ("M1", "M2", "M3"):
        prepare_sample(raw, model)

    for name, values in raw.items():
        np.testing.assert_array_equal(values, before[name], err_msg=name)


def test_repeated_preparation_is_bit_identical():
    raw = make_raw(prev_fire=1.0, fire=1.0)
    for model in ("M1", "M2", "M3"):
        first = prepare_sample(raw, model)
        second = prepare_sample(raw, model)
        np.testing.assert_array_equal(first.input, second.input)
        np.testing.assert_array_equal(first.target, second.target)
        np.testing.assert_array_equal(first.valid, second.valid)


def test_missing_feature_is_rejected():
    raw = make_raw()
    del raw["NDVI"]
    with pytest.raises(PreprocessingError, match="missing raw features"):
        preprocess_features(raw)


def test_wrong_length_is_rejected():
    raw = make_raw()
    raw["tmmn"] = np.zeros(10, dtype=np.float32)
    with pytest.raises(PreprocessingError, match="4096"):
        preprocess_features(raw)


def test_non_finite_input_is_rejected():
    raw = make_raw()
    raw["vs"][0] = np.nan
    with pytest.raises(PreprocessingError, match="non-finite"):
        preprocess_features(raw)


def test_unknown_model_is_rejected():
    with pytest.raises(PreprocessingError, match="unsupported model"):
        prepare_sample(make_raw(), "M4")


def test_channel_contract_has_no_duplicates():
    for model, channels in MODEL_CHANNELS.items():
        assert len(channels) == len(set(channels)), model
    assert MODEL_FEATURES["M2"] == (
        "PrevFireMask", "tmmn", "tmmx", "pr", "sph", "vs", "th"
    )


# ------------------------------------------------- Phase 0 parity (exact)


@pytest.mark.parametrize("sample_name", ["fire", "all_unknown"])
def test_matches_phase0_reference_exactly(real_paths, sample_name):
    for path in (real_paths["manifest"], real_paths["phase0_npz"]):
        if not os.path.isfile(path):
            pytest.skip(f"{path} not available")
    if not os.path.isdir(real_paths["root"]):
        pytest.skip("archive/ shards not present")

    with open(real_paths["phase0_report"], encoding="utf-8") as handle:
        phase0 = json.load(handle)

    reference = np.load(real_paths["phase0_npz"])
    record_index = phase0["samples"][sample_name]["record_index"]
    sample_id = f"eval_00_{record_index:06d}"

    catalog = load_catalog(real_paths["manifest"], real_paths["root"])
    raw = catalog.load_features(sample_id)

    for model in ("M1", "M2", "M3"):
        prepared = prepare_sample(raw, model)
        expected = reference[f"{sample_name}_{model}_input"]

        assert prepared.input.dtype == expected.dtype == np.float32
        assert prepared.input.shape == expected.shape
        max_abs_diff = float(np.abs(prepared.input - expected).max())
        print(f"PARITY {sample_name}/{model}: shape={prepared.input.shape} "
              f"dtype={prepared.input.dtype} max_abs_diff={max_abs_diff}")
        assert np.array_equal(prepared.input, expected), (
            f"{sample_name}/{model}: max_abs_diff={max_abs_diff}"
        )

    np.testing.assert_array_equal(
        prepared.target, reference[f"{sample_name}_target"]
    )
    np.testing.assert_array_equal(
        prepared.valid, reference[f"{sample_name}_valid"]
    )
