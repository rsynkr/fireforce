"""Deterministic sample catalog over the next-day-wildfire TFRecord shards.

Phase 1 scope: stable IDs, a versioned manifest, and a strict read-only loader.
No preprocessing, inference, or HTTP surface lives here.

ID format: ``<split>_<shard:02d>_<record_index:06d>`` e.g. ``eval_00_000027``.
The ID encodes only split, shard number, and zero-based record index. It never
encodes or implies a date, location, or event.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import tensorflow as tf

from app.services.preprocessing import RAW_FEATURES

MANIFEST_VERSION = 1
PIXELS_PER_RECORD = 64 * 64
SPLITS = ("train", "eval", "test")
ID_FORMAT = "<split>_<shard:02d>_<record_index:06d>"

SAMPLE_ID_RE = re.compile(r"^(train|eval|test)_(\d{2})_(\d{6})$")
SHARD_RE = re.compile(
    r"^next_day_wildfire_spread_(train|eval|test)_(\d{2})\.tfrecord$"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_FIRE_MASK_DESCRIPTION = {
    "FireMask": tf.io.FixedLenFeature([PIXELS_PER_RECORD], tf.float32)
}

_ENTRY_KEYS = (
    "sample_id",
    "split",
    "shard",
    "record_index",
    "target_valid_pixels",
    "target_fire_pixels",
)
_SHARD_KEYS = ("split", "shard", "record_count", "size_bytes", "sha256")


class SampleCatalogError(Exception):
    """Base class for catalog failures."""


class ManifestValidationError(SampleCatalogError):
    """The manifest is structurally invalid or internally inconsistent."""


class InvalidSampleIdError(SampleCatalogError):
    """The sample ID is malformed."""


class SampleNotFoundError(SampleCatalogError):
    """The sample ID is well formed but absent from the manifest."""


class ArtifactMissingError(SampleCatalogError):
    """A shard referenced by the manifest is not present on disk."""


class ManifestIntegrityError(SampleCatalogError):
    """A record does not match the counts recorded for it in the manifest."""


class RecordIndexError(SampleCatalogError):
    """A record index is outside the contents of its shard."""


@dataclass(frozen=True)
class SampleEntry:
    sample_id: str
    split: str
    shard: str
    record_index: int
    target_valid_pixels: int
    target_fire_pixels: int

    def to_dict(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in _ENTRY_KEYS}


@dataclass(frozen=True)
class Sample:
    entry: SampleEntry
    record_sha256: str
    fire_mask: np.ndarray  # (64, 64) float32, values in {-1.0, 0.0, 1.0}

    def to_dict(self) -> dict[str, Any]:
        data = self.entry.to_dict()
        data["record_sha256"] = self.record_sha256
        return data


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ManifestValidationError(message)


def _require_sha256(value: Any, where: str) -> None:
    _require(
        isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
        f"{where} must be a 64-character lowercase hex sha256",
    )


def split_shard_name(shard: str) -> tuple[str, str]:
    """Return (split, two-digit suffix) for a canonical shard filename."""
    if not isinstance(shard, str):
        raise ManifestValidationError("shard name must be a string")
    match = SHARD_RE.fullmatch(shard)
    if match is None:
        raise ManifestValidationError(
            f"shard name {shard!r} does not match "
            "next_day_wildfire_spread_<split>_<NN>.tfrecord"
        )
    return match.group(1), match.group(2)


def make_sample_id(split: str, shard: str, record_index: int) -> str:
    """Derive the stable sample ID. Raises ManifestValidationError on bad input."""
    _require(split in SPLITS, f"unsupported split {split!r}")
    shard_split, suffix = split_shard_name(shard)
    _require(
        shard_split == split,
        f"shard {shard!r} belongs to split {shard_split!r}, not {split!r}",
    )
    _require(
        _is_int(record_index) and 0 <= record_index <= 999_999,
        f"record_index {record_index!r} is outside the 6-digit ID range",
    )
    return f"{split}_{suffix}_{record_index:06d}"


def parse_sample_id(sample_id: Any) -> tuple[str, str, int]:
    """Split a sample ID into (split, suffix, record_index)."""
    if not isinstance(sample_id, str):
        raise InvalidSampleIdError("sample_id must be a string")
    match = SAMPLE_ID_RE.fullmatch(sample_id)
    if match is None:
        raise InvalidSampleIdError(
            f"invalid sample_id {sample_id!r}; expected {ID_FORMAT}"
        )
    return match.group(1), match.group(2), int(match.group(3))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def scan_shard(shard_path: str, split: str) -> dict[str, Any]:
    """Read one shard once, hashing record payloads and counting valid/fire pixels.

    Returns shard identity plus per-record target counts. Content hashing uses
    record payloads (not file mtimes), so the digest is stable across copies.
    """
    shard = os.path.basename(shard_path)
    shard_split, _ = split_shard_name(shard)
    _require(
        shard_split == split,
        f"shard {shard!r} belongs to split {shard_split!r}, not {split!r}",
    )

    digest = hashlib.sha256()
    records: list[dict[str, int]] = []
    for index, raw in enumerate(tf.data.TFRecordDataset(shard_path)):
        payload = raw.numpy()
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
        example = tf.io.parse_single_example(payload, _FIRE_MASK_DESCRIPTION)
        fire_mask = example["FireMask"].numpy()
        records.append(
            {
                "record_index": index,
                "target_valid_pixels": int((fire_mask != -1).sum()),
                "target_fire_pixels": int((fire_mask == 1).sum()),
            }
        )

    _require(len(records) > 0, f"shard {shard!r} contains no records")

    return {
        "split": split,
        "shard": shard,
        "record_count": len(records),
        "size_bytes": os.path.getsize(shard_path),
        "sha256": digest.hexdigest(),
        "records": records,
    }


def discover_shards(data_root: str) -> list[tuple[str, str]]:
    """Return canonical (path, split) pairs beneath data_root, deterministically ordered."""
    if not os.path.isdir(data_root):
        raise ArtifactMissingError(f"data root does not exist: {data_root}")

    found: list[tuple[str, str]] = []
    for name in os.listdir(data_root):
        path = os.path.join(data_root, name)
        if not os.path.isfile(path):
            continue
        match = SHARD_RE.fullmatch(name)
        if match is not None:
            found.append((path, match.group(1)))
        elif name.endswith(".tfrecord"):
            raise ManifestValidationError(
                f"unexpected TFRecord filename {name!r}; expected "
                "next_day_wildfire_spread_<split>_<NN>.tfrecord"
            )

    if not found:
        raise ArtifactMissingError(f"no canonical TFRecord shards found in {data_root}")

    order = {split: i for i, split in enumerate(SPLITS)}
    found.sort(key=lambda item: (order[item[1]], os.path.basename(item[0])))
    return found


def build_manifest(
    data_root: str,
    dataset_name: str = "next-day-wildfire-spread",
    generator_version: str = "1.0.0",
) -> dict[str, Any]:
    """Scan every canonical shard and return a deterministic manifest dict."""
    shards: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []

    for path, split in discover_shards(data_root):
        scanned = scan_shard(path, split)
        shards.append({key: scanned[key] for key in _SHARD_KEYS})
        for record in scanned["records"]:
            samples.append(
                {
                    "sample_id": make_sample_id(
                        split, scanned["shard"], record["record_index"]
                    ),
                    "split": split,
                    "shard": scanned["shard"],
                    "record_index": record["record_index"],
                    "target_valid_pixels": record["target_valid_pixels"],
                    "target_fire_pixels": record["target_fire_pixels"],
                }
            )

    fingerprint_payload = json.dumps(
        shards, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "dataset_name": dataset_name,
        "dataset_fingerprint": _sha256_bytes(fingerprint_payload),
        "id_format": ID_FORMAT,
        "generator_version": generator_version,
        "shards": shards,
        "samples": samples,
    }
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: Any) -> None:
    """Validate structure, ID derivation, uniqueness, and count bounds."""
    _require(isinstance(manifest, dict), "manifest must be a JSON object")
    _require(
        _is_int(manifest.get("manifest_version"))
        and manifest["manifest_version"] == MANIFEST_VERSION,
        f"manifest_version must be {MANIFEST_VERSION}",
    )
    _require(
        isinstance(manifest.get("dataset_name"), str)
        and manifest["dataset_name"].strip() != "",
        "dataset_name must be a non-empty string",
    )
    _require_sha256(manifest.get("dataset_fingerprint"), "dataset_fingerprint")

    if "id_format" in manifest:
        _require(
            manifest["id_format"] == ID_FORMAT,
            f"id_format must be {ID_FORMAT!r}",
        )

    shards = manifest.get("shards")
    _require(isinstance(shards, list) and len(shards) > 0, "shards must be a non-empty list")
    shard_table: dict[str, dict[str, Any]] = {}
    for position, shard in enumerate(shards):
        where = f"shards[{position}]"
        _require(isinstance(shard, dict), f"{where} must be an object")
        for key in _SHARD_KEYS:
            _require(key in shard, f"{where} is missing {key!r}")
        _require(shard["split"] in SPLITS, f"{where}.split is invalid")
        _require(isinstance(shard["shard"], str), f"{where}.shard must be a string")
        _require(
            os.path.basename(shard["shard"]) == shard["shard"],
            f"{where}.shard must be a bare filename",
        )
        shard_split, _ = split_shard_name(shard["shard"])
        _require(
            shard_split == shard["split"],
            f"{where}.shard split does not match .split",
        )
        _require(
            _is_int(shard["record_count"]) and shard["record_count"] > 0,
            f"{where}.record_count must be a positive integer",
        )
        _require(
            _is_int(shard["size_bytes"]) and shard["size_bytes"] > 0,
            f"{where}.size_bytes must be a positive integer",
        )
        _require_sha256(shard["sha256"], f"{where}.sha256")
        _require(shard["shard"] not in shard_table, f"duplicate shard {shard['shard']!r}")
        shard_table[shard["shard"]] = shard

    samples = manifest.get("samples")
    _require(isinstance(samples, list) and len(samples) > 0, "samples must be a non-empty list")
    seen_ids: set[str] = set()
    for position, sample in enumerate(samples):
        where = f"samples[{position}]"
        _require(isinstance(sample, dict), f"{where} must be an object")
        for key in _ENTRY_KEYS:
            _require(key in sample, f"{where} is missing {key!r}")
        sample_id = sample["sample_id"]
        try:
            split, suffix, record_index = parse_sample_id(sample_id)
        except InvalidSampleIdError as exc:
            raise ManifestValidationError(f"{where}: {exc}") from exc
        _require(sample_id not in seen_ids, f"duplicate sample_id {sample_id!r}")
        seen_ids.add(sample_id)

        _require(sample["split"] in SPLITS, f"{where}.split is invalid")
        _require(sample["split"] == split, f"{where}.split does not match sample_id")
        _require(
            isinstance(sample["shard"], str)
            and os.path.basename(sample["shard"]) == sample["shard"],
            f"{where}.shard must be a bare filename",
        )
        shard_split, shard_suffix = split_shard_name(sample["shard"])
        _require(shard_split == split, f"{where}.shard does not match sample_id split")
        _require(shard_suffix == suffix, f"{where}.shard does not match sample_id suffix")
        _require(sample["shard"] in shard_table, f"{where} references unknown shard")
        _require(
            _is_int(sample["record_index"]) and sample["record_index"] == record_index,
            f"{where}.record_index does not match sample_id",
        )
        _require(
            0 <= sample["record_index"] < shard_table[sample["shard"]]["record_count"],
            f"{where}.record_index is outside shard record_count",
        )
        for key in ("target_valid_pixels", "target_fire_pixels"):
            _require(
                _is_int(sample[key]) and 0 <= sample[key] <= PIXELS_PER_RECORD,
                f"{where}.{key} must be an integer in [0, {PIXELS_PER_RECORD}]",
            )
        _require(
            sample["target_fire_pixels"] <= sample["target_valid_pixels"],
            f"{where}.target_fire_pixels exceeds target_valid_pixels",
        )


class SampleCatalog:
    """Strict, read-only view over a validated manifest plus the shard directory."""

    def __init__(self, manifest: dict[str, Any], data_root: str):
        validate_manifest(manifest)
        self._manifest = manifest
        self.data_root = os.path.abspath(data_root)
        self._shards = {item["shard"]: item for item in manifest["shards"]}
        self._entries = {
            item["sample_id"]: SampleEntry(**{key: item[key] for key in _ENTRY_KEYS})
            for item in manifest["samples"]
        }

    @property
    def manifest(self) -> dict[str, Any]:
        return self._manifest

    @property
    def fingerprint(self) -> str:
        return self._manifest["dataset_fingerprint"]

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    @property
    def shard_count(self) -> int:
        return len(self._shards)

    def split_counts(self) -> dict[str, int]:
        counts = {split: 0 for split in SPLITS}
        for entry in self._entries.values():
            counts[entry.split] += 1
        return counts

    def list_entries(self) -> tuple[SampleEntry, ...]:
        """All entries in deterministic manifest order."""
        return tuple(
            self._entries[sample["sample_id"]] for sample in self._manifest["samples"]
        )

    def get_entry(self, sample_id: str) -> SampleEntry:
        parse_sample_id(sample_id)
        entry = self._entries.get(sample_id)
        if entry is None:
            raise SampleNotFoundError(f"unknown sample_id {sample_id!r}")
        return entry

    def _shard_path(self, shard: str) -> str:
        if os.path.basename(shard) != shard:
            raise ManifestValidationError(f"unsafe shard name {shard!r}")
        path = os.path.abspath(os.path.join(self.data_root, shard))
        if os.path.commonpath([path, self.data_root]) != self.data_root:
            raise ManifestValidationError(f"shard {shard!r} escapes the data root")
        return path

    def read_record(self, shard: str, record_index: int) -> bytes:
        if not _is_int(record_index) or record_index < 0:
            raise RecordIndexError(f"record_index {record_index!r} is invalid")
        path = self._shard_path(shard)
        if not os.path.isfile(path):
            raise ArtifactMissingError(f"shard not found: {path}")

        for raw in tf.data.TFRecordDataset(path).skip(record_index).take(1):
            return raw.numpy()
        raise RecordIndexError(
            f"shard {shard!r} has no record at index {record_index}"
        )

    def get_sample(self, sample_id: str) -> Sample:
        """Resolve an ID and verify the record still matches the manifest."""
        entry, payload = self._read_verified(sample_id)

        example = tf.io.parse_single_example(payload, _FIRE_MASK_DESCRIPTION)
        fire_mask = example["FireMask"].numpy()

        return Sample(
            entry=entry,
            record_sha256=_sha256_bytes(payload),
            fire_mask=fire_mask.reshape(64, 64).astype(np.float32),
        )

    def _read_verified(self, sample_id: str) -> tuple[SampleEntry, bytes]:
        entry = self.get_entry(sample_id)
        payload = self.read_record(entry.shard, entry.record_index)

        example = tf.io.parse_single_example(payload, _FIRE_MASK_DESCRIPTION)
        fire_mask = example["FireMask"].numpy()
        valid = int((fire_mask != -1).sum())
        fire = int((fire_mask == 1).sum())
        if (valid, fire) != (entry.target_valid_pixels, entry.target_fire_pixels):
            raise ManifestIntegrityError(
                f"{sample_id}: manifest records valid={entry.target_valid_pixels} "
                f"fire={entry.target_fire_pixels} but shard has valid={valid} fire={fire}"
            )
        return entry, payload

    def load_features(self, sample_id: str) -> dict[str, np.ndarray]:
        """Return all 13 raw float32 features for a sample, in RAW_FEATURES order.

        This is the input to the shared preprocessing pipeline. The same
        manifest/data integrity check as get_sample applies.
        """
        entry, payload = self._read_verified(sample_id)
        description = {
            name: tf.io.FixedLenFeature([PIXELS_PER_RECORD], tf.float32)
            for name in RAW_FEATURES
        }
        example = tf.io.parse_single_example(payload, description)
        return {name: example[name].numpy() for name in RAW_FEATURES}


def save_manifest(manifest: dict[str, Any], path: str) -> None:
    validate_manifest(manifest)
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def load_catalog(manifest_path: str, data_root: str) -> SampleCatalog:
    try:
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except FileNotFoundError as exc:
        raise ArtifactMissingError(f"manifest not found: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestValidationError(f"manifest is not valid JSON: {exc}") from exc
    return SampleCatalog(manifest, data_root)
