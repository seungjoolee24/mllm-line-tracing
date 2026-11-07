"""Export a complete 10-case collection into GitHub Release sized ZIPs.

Each case ZIP contains its whole sample folder. A separate metadata ZIP contains
every non-.pt file in the run. All archive paths begin with the original run
folder name, so extracting the archives into one destination restores that run.
No model or GPU is loaded. Tensor verification is performed on the CPU.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import tempfile
import zipfile


PACKAGE_PROTOCOL = "qwen3vl_trace_case_zip_v1"
EXPECTED_CASES = 10
EXPECTED_TENSORS_PER_CASE = 206
MAX_ARCHIVE_BYTES = 2 * 1024**3
CHUNK_BYTES = 1024 * 1024


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _contained_file(root, relative):
    relative = Path(relative)
    if relative.is_absolute():
        raise ValueError(f"absolute file entry in source metadata: {relative}")
    target = (root / relative).resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ValueError(f"source file escapes its run folder: {relative}") from error
    if not target.is_file():
        raise ValueError(f"missing source file: {target}")
    return target


def _source_files(root):
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlinks are not exported: {path}")
        if path.is_file():
            path.resolve().relative_to(root)
            files.append(path)
    return files


def _source_state(files, root):
    return {path.relative_to(root).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns)
            for path in files}


def _load_verifier():
    path = Path(__file__).with_name("collect.py")
    spec = importlib.util.spec_from_file_location("trace_export_collector", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.SCHEMA_VERSION != "1.1":
        raise ValueError("export requires the schema 1.1 collector verifier")
    return module.verify_completed_sample


def validate_complete_run(run):
    """Reject partial runs and verify every recorded tensor's hash/shape/dtype."""
    run = Path(run).resolve()
    if not run.is_dir():
        raise ValueError(f"run directory does not exist: {run}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run.name):
        raise ValueError("run folder name must be portable ASCII letters, digits, dots, dashes, or underscores")
    summary = _read_json(run / "run_summary.json")
    config = _read_json(run / "run_config.json")
    if summary.get("schema_version") != "1.1" or config.get("schema_version") != "1.1":
        raise ValueError("only schema 1.1 collection runs can be packaged")
    if summary.get("status") != "complete" or summary.get("failures"):
        raise ValueError("only a complete run with no failures can be packaged")
    if summary.get("intended_cases") != EXPECTED_CASES or summary.get("completed_cases") != EXPECTED_CASES:
        raise ValueError(f"expected a completed {EXPECTED_CASES}-case run")
    if config.get("intended_dataset_count") != EXPECTED_CASES:
        raise ValueError("run config and expected dataset count disagree")
    config_hash = config.get("run_config_sha256")
    if not isinstance(config_hash, str) or summary.get("run_config_sha256") != config_hash:
        raise ValueError("run config and summary hashes disagree")
    cases = summary.get("cases")
    if not isinstance(cases, list) or len(cases) != EXPECTED_CASES:
        raise ValueError("run summary is missing its completed case inventory")
    identities = [row.get("sample_id") for row in cases]
    if len(set(identities)) != EXPECTED_CASES or any(
        not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value) for value in identities
    ):
        raise ValueError("case IDs must be unique safe strings")
    verify = _load_verifier()
    verified, tensor_hashes = [], {}
    for identity in identities:
        sample_dir = run / "samples" / identity
        metadata = verify(sample_dir, config_hash, check_shapes=True)
        if metadata.get("sample_id") != identity:
            raise ValueError(f"sample ID mismatch in {identity}")
        records = metadata["tensors"]
        if len(records) != EXPECTED_TENSORS_PER_CASE:
            raise ValueError(f"{identity}: expected {EXPECTED_TENSORS_PER_CASE} tensor files, got {len(records)}")
        for record in records:
            path = _contained_file(sample_dir.resolve(), record["file"])
            relative = path.relative_to(run).as_posix()
            if not relative.endswith(".pt") or relative in tensor_hashes:
                raise ValueError(f"invalid or duplicate tensor file: {relative}")
            tensor_hashes[relative] = record["sha256"]
        verified.append({"sample_id": identity, "metadata": metadata, "directory": sample_dir})
        print(f"Verified {identity}: {len(records)} tensor files", flush=True)
    files = _source_files(run)
    actual_tensor_files = {path.relative_to(run).as_posix() for path in files if path.suffix == ".pt"}
    if actual_tensor_files != set(tensor_hashes):
        missing = set(tensor_hashes) - actual_tensor_files
        extra = actual_tensor_files - set(tensor_hashes)
        raise ValueError(f"run tensor inventory mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
    expected_total = EXPECTED_CASES * EXPECTED_TENSORS_PER_CASE
    if len(tensor_hashes) != expected_total:
        raise ValueError(f"expected {expected_total} tensors, got {len(tensor_hashes)}")
    return {"run": run, "summary": summary, "config": config, "cases": verified,
            "files": files, "tensor_hashes": tensor_hashes,
            "source_state": _source_state(files, run)}


def write_streaming_zip(target, run, files, expected_hashes, compression_level=1):
    """Compress and hash one source file at a time, never buffering its contents."""
    target, run = Path(target), Path(run)
    uncompressed = 0
    try:
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=compression_level, allowZip64=True) as archive:
            for path in files:
                path = Path(path)
                relative = path.relative_to(run).as_posix()
                archive_name = f"{run.name}/{relative}"
                digest = hashlib.sha256()
                file_bytes = 0
                info = zipfile.ZipInfo.from_file(path, arcname=archive_name)
                info.compress_type = zipfile.ZIP_DEFLATED
                info._compresslevel = compression_level
                with path.open("rb") as source, archive.open(info, "w", force_zip64=True) as destination:
                    for chunk in iter(lambda: source.read(CHUNK_BYTES), b""):
                        digest.update(chunk)
                        destination.write(chunk)
                        file_bytes += len(chunk)
                if digest.hexdigest() != expected_hashes[relative]:
                    raise ValueError(f"source changed while archiving: {relative}")
                uncompressed += file_bytes
                if target.stat().st_size >= MAX_ARCHIVE_BYTES:
                    raise ValueError(f"archive exceeds GitHub Release size limit: {target.name}")
        size = target.stat().st_size
        if size >= MAX_ARCHIVE_BYTES:
            raise ValueError(f"archive must be smaller than 2 GiB: {target.name}")
        return {"file": target.name, "bytes": size, "sha256": sha256_file(target),
                "source_file_count": len(files), "uncompressed_bytes": uncompressed}
    except BaseException:
        target.unlink(missing_ok=True)
        raise


def export_run(run, output, compression_level=1):
    run, output = Path(run).resolve(), Path(output).resolve()
    try:
        output.relative_to(run)
    except ValueError:
        pass
    else:
        raise ValueError("output must be outside the source run directory")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}; choose a new export directory")
    if type(compression_level) is not int or not 0 <= compression_level <= 9:
        raise ValueError("compression level must be an integer from 0 to 9")
    validated = validate_complete_run(run)
    hashes = dict(validated["tensor_hashes"])
    metadata_files = [path for path in validated["files"] if path.suffix != ".pt"]
    for path in metadata_files:
        hashes[path.relative_to(run).as_posix()] = sha256_file(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        archives = []
        # Publish its local path/hash immediately in stdout so metadata analysis
        # can start while the ten large archives are still being compressed.
        metadata = write_streaming_zip(temporary / f"{run.name}_metadata.zip", run,
                                      metadata_files, hashes, compression_level)
        metadata.update({"kind": "metadata", "sample_id": None, "tensor_file_count": 0,
                         "archive_path_prefix": f"{run.name}/"})
        archives.append(metadata)
        print(json.dumps({"event": "metadata_archive_ready",
                          "path": str(temporary / metadata["file"]),
                          "final_path": str(output / metadata["file"]),
                          "file": metadata["file"], "bytes": metadata["bytes"],
                          "sha256": metadata["sha256"]}), flush=True)
        for case in validated["cases"]:
            identity = case["sample_id"]
            files = [path for path in validated["files"] if case["directory"] in path.parents]
            if not files:
                raise ValueError(f"case folder is empty: {identity}")
            artifact = write_streaming_zip(temporary / f"{run.name}_{identity}.zip", run, files,
                                           hashes, compression_level)
            artifact.update({"kind": "case", "sample_id": identity,
                             "tensor_file_count": EXPECTED_TENSORS_PER_CASE,
                             "archive_path_prefix": f"{run.name}/samples/{identity}/"})
            archives.append(artifact)
            print(f"Packaged {artifact['file']}: {artifact['bytes']} bytes", flush=True)
        current_files = _source_files(run)
        if _source_state(current_files, run) != validated["source_state"]:
            raise ValueError("source run changed during export; no export was published")
        manifest = {
            "schema_version": "1.0", "package_protocol": PACKAGE_PROTOCOL,
            "collection_schema_version": "1.1", "run_name": run.name, "status": "complete",
            "run_config_sha256": validated["config"]["run_config_sha256"],
            "model_id": validated["summary"].get("model_id"),
            "model_revision": validated["summary"].get("model_revision"),
            "exporter_sha256": sha256_file(__file__),
            "compression": {"method": "ZIP_DEFLATED", "level": compression_level,
                            "stream_chunk_bytes": CHUNK_BYTES},
            "case_count": len(validated["cases"]), "expected_case_count": EXPECTED_CASES,
            "tensor_file_count": len(validated["tensor_hashes"]),
            "expected_tensor_file_count": EXPECTED_CASES * EXPECTED_TENSORS_PER_CASE,
            "source_file_count": len(validated["files"]),
            "source_non_tensor_file_count": len(metadata_files),
            "source_tensor_bytes": sum(path.stat().st_size for path in validated["files"] if path.suffix == ".pt"),
            "archive_count": len(archives), "archives": archives,
            "total_archive_bytes": sum(artifact["bytes"] for artifact in archives),
            "maximum_archive_bytes_exclusive": MAX_ARCHIVE_BYTES,
            "verification": {"completed_sample_verifier": "collect.verify_completed_sample",
                             "source_tensor_sha256_shapes_dtypes": True,
                             "source_hashes_checked_during_compression": True,
                             "source_inventory_unchanged": True},
            "restore": {"destination": "One common parent directory",
                        "steps": ["Verify every archive's SHA-256 against this manifest.",
                                  "Extract the metadata ZIP and all ten case ZIPs into the same parent directory."],
                        "result_folder": run.name,
                        "overlap_note": "Case metadata appears in both archive types; overlapping files are identical.",
                        "contains_original_collection_logs": any(path.name == "collection.log" for path in metadata_files)},
        }
        manifest_path = temporary / "artifact_manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                                 encoding="utf-8")
        if output.exists():
            raise FileExistsError(f"output appeared during export: {output}")
        temporary.rename(output)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Completed schema 1.1 run directory")
    parser.add_argument("--output", required=True, help="New export directory outside the run")
    parser.add_argument("--compression-level", type=int, default=1, choices=range(10))
    args = parser.parse_args(argv)
    try:
        manifest = export_run(args.run, args.output, args.compression_level)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps({"output": str(Path(args.output).resolve()),
                      "package_protocol": manifest["package_protocol"],
                      "case_count": manifest["case_count"],
                      "tensor_file_count": manifest["tensor_file_count"],
                      "archive_count": manifest["archive_count"],
                      "total_archive_bytes": manifest["total_archive_bytes"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
