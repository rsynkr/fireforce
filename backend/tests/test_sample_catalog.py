import copy
import json
import os

import numpy as np
import pytest

from app.services.sample_catalog import (
    ArtifactMissingError,
    InvalidSampleIdError,
    ManifestIntegrityError,
    ManifestValidationError,
    PIXELS_PER_RECORD,
    RecordIndexError,
    SampleCatalog,
    SampleNotFoundError,
    build_manifest,
    load_catalog,
    make_sample_id,
    save_manifest,
    validate_manifest,
)

TRAIN_SHARD = "next_day_wildfire_spread_train_00.tfrecord"
EVAL_SHARD = "next_day_wildfire_spread_eval_00.tfrecord"


def catalog_for(dataset) -> SampleCatalog:
    return SampleCatalog(dataset["manifest"], str(dataset["root"]))


# ---------------------------------------------------------------- determinism


def test_manifest_is_deterministic(dataset):
    first = build_manifest(str(dataset["root"]))
    second = build_manifest(str(dataset["root"]))
    assert first == second
    assert first["manifest_version"] == 1
    assert first["dataset_fingerprint"] == second["dataset_fingerprint"]


def test_ids_are_derived_from_split_shard_index(dataset):
    ids = [sample["sample_id"] for sample in dataset["manifest"]["samples"]]
    assert ids == [
        "train_00_000000",
        "train_00_000001",
        "train_00_000002",
        "train_00_000003",
        "train_00_000004",
        "eval_00_000000",
        "eval_00_000001",
        "eval_00_000002",
    ]
    assert make_sample_id("train", TRAIN_SHARD, 42) == "train_00_000042"
    assert make_sample_id("eval", EVAL_SHARD, 27) == "eval_00_000027"


def test_fingerprint_changes_when_content_changes(dataset, shard_writer):
    original = dataset["manifest"]["dataset_fingerprint"]
    write_shard, make_fire_mask = shard_writer

    write_shard(
        str(dataset["root"] / TRAIN_SHARD),
        [make_fire_mask(0), make_fire_mask(2), make_fire_mask(7), make_fire_mask(1), make_fire_mask(3)],
    )
    rebuilt = build_manifest(str(dataset["root"]))
    assert rebuilt["dataset_fingerprint"] != original


# ------------------------------------------------------------ record contents


def test_first_middle_and_last_records(dataset):
    catalog = catalog_for(dataset)

    first = catalog.get_sample("train_00_000000")
    middle = catalog.get_sample("train_00_000002")
    last = catalog.get_sample("train_00_000004")

    assert (first.entry.target_valid_pixels, first.entry.target_fire_pixels) == (4096, 0)
    assert (middle.entry.target_valid_pixels, middle.entry.target_fire_pixels) == (4000, 0)
    assert (last.entry.target_valid_pixels, last.entry.target_fire_pixels) == (4096, 3)
    assert last.fire_mask.shape == (64, 64)
    assert int((last.fire_mask == 1).sum()) == 3
    assert int((middle.fire_mask == -1).sum()) == 96

    counts = catalog.split_counts()
    assert counts == {"train": 5, "eval": 3, "test": 0}
    assert catalog.entry_count == 8
    assert catalog.shard_count == 2


def test_repeated_lookup_is_identical(dataset):
    catalog = catalog_for(dataset)
    first = catalog.get_sample("eval_00_000001")
    second = catalog.get_sample("eval_00_000001")

    assert first.to_dict() == second.to_dict()
    assert first.record_sha256 == second.record_sha256
    assert np.array_equal(first.fire_mask, second.fire_mask)


def test_all_unknown_target_is_preserved(dataset):
    catalog = catalog_for(dataset)
    sample = catalog.get_sample("train_00_000003")
    assert sample.entry.target_valid_pixels == 1
    assert sample.entry.target_fire_pixels == 1
    assert int((sample.fire_mask == -1).sum()) == 4095


# ------------------------------------------------------------- invalid inputs


@pytest.mark.parametrize(
    "sample_id",
    [
        "",
        "train_00_1",
        "train_00_0000000",
        "train_0_000000",
        "train_000_000000",
        "TRAIN_00_000000",
        "train-00-000000",
        "train_00_000000\n",
        "../train_00_000000",
        "train_99_999999x",
    ],
)
def test_malformed_ids_are_rejected(dataset, sample_id):
    catalog = catalog_for(dataset)
    with pytest.raises(InvalidSampleIdError):
        catalog.get_entry(sample_id)


def test_well_formed_unknown_id_is_not_found(dataset):
    catalog = catalog_for(dataset)
    with pytest.raises(SampleNotFoundError):
        catalog.get_sample("train_09_000000")
    with pytest.raises(SampleNotFoundError):
        catalog.get_sample("eval_00_000003")


def test_duplicate_ids_are_rejected(dataset):
    manifest = copy.deepcopy(dataset["manifest"])
    manifest["samples"].append(copy.deepcopy(manifest["samples"][0]))
    with pytest.raises(ManifestValidationError, match="duplicate sample_id"):
        validate_manifest(manifest)
    with pytest.raises(ManifestValidationError, match="duplicate sample_id"):
        SampleCatalog(manifest, str(dataset["root"]))


def test_out_of_range_record_index_in_manifest_is_rejected(dataset):
    manifest = copy.deepcopy(dataset["manifest"])
    entry = manifest["samples"][0]
    entry["record_index"] = 5  # shard has exactly 5 records
    entry["sample_id"] = "train_00_000005"
    with pytest.raises(ManifestValidationError, match="outside shard record_count"):
        validate_manifest(manifest)


def test_reader_rejects_out_of_range_and_negative_indices(dataset):
    catalog = catalog_for(dataset)
    with pytest.raises(RecordIndexError):
        catalog.read_record(TRAIN_SHARD, 5)
    with pytest.raises(RecordIndexError):
        catalog.read_record(TRAIN_SHARD, -1)


def test_id_beyond_six_digits_is_rejected():
    with pytest.raises(ManifestValidationError):
        make_sample_id("train", TRAIN_SHARD, 1_000_000)


def test_missing_shard_is_reported(dataset):
    catalog = catalog_for(dataset)
    os.remove(dataset["root"] / EVAL_SHARD)

    with pytest.raises(ArtifactMissingError, match="shard not found"):
        catalog.get_sample("eval_00_000000")
    # A shard that still exists keeps working.
    assert catalog.get_sample("train_00_000000").entry.split == "train"


def test_missing_manifest_is_reported(tmp_path, dataset):
    with pytest.raises(ArtifactMissingError, match="manifest not found"):
        load_catalog(str(tmp_path / "nope.json"), str(dataset["root"]))


def test_manifest_counts_are_checked_against_the_shard(dataset):
    manifest = copy.deepcopy(dataset["manifest"])
    entry = next(s for s in manifest["samples"] if s["sample_id"] == "eval_00_000001")
    entry["target_valid_pixels"] = 4000  # actual is 4086
    catalog = SampleCatalog(manifest, str(dataset["root"]))
    with pytest.raises(ManifestIntegrityError, match="valid=4000"):
        catalog.get_sample("eval_00_000001")


def test_fire_pixels_cannot_exceed_valid_pixels(dataset):
    manifest = copy.deepcopy(dataset["manifest"])
    manifest["samples"][0]["target_fire_pixels"] = PIXELS_PER_RECORD
    manifest["samples"][0]["target_valid_pixels"] = 0
    with pytest.raises(ManifestValidationError, match="exceeds target_valid_pixels"):
        validate_manifest(manifest)


def test_unsafe_shard_paths_are_rejected(dataset):
    manifest = copy.deepcopy(dataset["manifest"])
    manifest["samples"][0]["shard"] = "../train.tfrecord"
    with pytest.raises(ManifestValidationError, match="bare filename"):
        validate_manifest(manifest)


def test_sample_identity_must_match_id(dataset):
    manifest = copy.deepcopy(dataset["manifest"])
    manifest["samples"][0]["split"] = "eval"
    with pytest.raises(ManifestValidationError, match="does not match sample_id"):
        validate_manifest(manifest)


def test_unexpected_tfrecord_name_fails_the_scan(dataset):
    with open(dataset["root"] / "stray.tfrecord", "wb") as handle:
        handle.write(b"")
    with pytest.raises(ManifestValidationError, match="unexpected TFRecord filename"):
        build_manifest(str(dataset["root"]))


def test_manifest_round_trip(dataset, tmp_path):
    path = tmp_path / "sample_manifest.json"
    save_manifest(dataset["manifest"], str(path))
    loaded = load_catalog(str(path), str(dataset["root"]))

    assert loaded.fingerprint == dataset["manifest"]["dataset_fingerprint"]
    assert loaded.entry_count == 8
    assert loaded.get_entry("eval_00_000002").to_dict() == {
        "sample_id": "eval_00_000002",
        "split": "eval",
        "shard": EVAL_SHARD,
        "record_index": 2,
        "target_valid_pixels": 4096,
        "target_fire_pixels": 4,
    }
    with open(path, encoding="utf-8") as handle:
        assert json.load(handle) == dataset["manifest"]


# ------------------------------------------------------------- real artifacts


def test_real_manifest_statistics_and_reference_parity(real_paths):
    if not os.path.isfile(real_paths["manifest"]):
        pytest.skip("artifacts/sample_manifest.json not built yet")
    if not os.path.isdir(real_paths["root"]):
        pytest.skip("archive/ shards not present")

    catalog = load_catalog(real_paths["manifest"], real_paths["root"])

    assert catalog.entry_count == 18_545
    assert catalog.shard_count == 19
    assert catalog.split_counts() == {"train": 14_979, "eval": 1_877, "test": 1_689}

    first = catalog.get_sample("eval_00_000000")
    middle = catalog.get_sample("eval_00_000027")
    last = catalog.get_sample("eval_00_000999")
    assert first.entry.record_index == 0
    assert middle.entry.record_index == 27
    assert last.entry.record_index == 999

    # Phase 0 reference was produced from this exact record; do not alter that
    # artifact, only compare against it.
    with open(real_paths["phase0_report"], encoding="utf-8") as handle:
        phase0 = json.load(handle)
    reference = phase0["samples"]["fire"]
    assert middle.record_sha256 == reference["record_sha256"]
    assert middle.entry.target_valid_pixels == reference["valid_pixels"] == 4096
    assert middle.entry.target_fire_pixels == reference["fire_pixels"] == 21

    all_unknown = catalog.get_sample("eval_00_000009")
    assert all_unknown.entry.target_valid_pixels == 0
    assert all_unknown.entry.target_fire_pixels == 0
