"""Atomic benchmark artifact helpers."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

ARTIFACT_SCHEMA_VERSION = 1


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _temporary_path(path, suffix):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=suffix
    )
    os.close(descriptor)
    return Path(temporary)


def _sync_file(path):
    with open(path, "rb") as stream:
        os.fsync(stream.fileno())


def _replace(source, destination):
    os.replace(str(source), str(destination))
    descriptor = os.open(str(Path(destination).parent), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_json(path, document):
    path = Path(path)
    temporary = _temporary_path(path, ".json.tmp")
    try:
        with open(temporary, "w") as stream:
            json.dump(
                document,
                stream,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        _replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def atomic_savez(path, payload):
    path = Path(path)
    temporary = _temporary_path(path, ".npz")
    try:
        np.savez_compressed(str(temporary), **payload)
        _sync_file(temporary)
        _replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def atomic_promote(source, destination):
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _sync_file(source)
    _replace(source, destination)
    return destination


def relative_artifact_path(owner_path, artifact_path):
    owner_path = Path(owner_path).resolve()
    artifact_path = Path(artifact_path).resolve()
    return Path(os.path.relpath(str(artifact_path), str(owner_path.parent))).as_posix()


def artifact_record(owner_path, artifact_path):
    artifact_path = Path(artifact_path)
    stat = artifact_path.stat()
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "path": relative_artifact_path(owner_path, artifact_path),
        "sha256": sha256_file(artifact_path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def resolve_artifact_path(owner_path, record):
    owner_path = Path(owner_path)
    recorded_path = record["path"] if isinstance(record, dict) else record
    path = Path(recorded_path)
    if not path.is_absolute():
        return owner_path.parent / path
    if path.is_file():
        return path
    return owner_path.parent / path.name


def validate_artifact_record(owner_path, record):
    if not isinstance(record, dict):
        raise TypeError("artifact record must be an object")
    required = {"schema_version", "path", "sha256", "size_bytes"}
    if not required.issubset(record):
        missing = sorted(required.difference(record))
        raise ValueError("artifact record is missing {}".format(", ".join(missing)))
    if record["schema_version"] != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unsupported artifact schema")
    if Path(record["path"]).is_absolute():
        raise ValueError("canonical artifact paths must be relative")
    path = resolve_artifact_path(owner_path, record)
    if not path.is_file():
        raise FileNotFoundError(f"artifact does not exist: {path}")
    stat = path.stat()
    if int(record["size_bytes"]) != int(stat.st_size):
        raise ValueError(f"artifact size mismatch: {path}")
    if "mtime_ns" in record and int(record["mtime_ns"]) != int(stat.st_mtime_ns):
        raise ValueError(f"artifact modification time mismatch: {path}")
    if record["sha256"] != sha256_file(path):
        raise ValueError(f"artifact checksum mismatch: {path}")
    return path


def precision_metrics_path(result_path, consistency):
    record = consistency.get("precision_metrics")
    if record is not None:
        return validate_artifact_record(result_path, record)
    legacy = consistency.get("precision_metrics_file")
    if legacy:
        path = resolve_artifact_path(result_path, legacy)
        if not path.is_file():
            raise FileNotFoundError(f"precision metrics do not exist: {path}")
        return path
    path = Path(result_path).with_suffix(".metrics.npz")
    if not path.is_file():
        raise FileNotFoundError(f"precision metrics do not exist: {path}")
    return path
