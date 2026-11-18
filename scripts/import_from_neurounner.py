#!/usr/bin/env python3
"""Import the immutable Neurounner DRIV3R staging snapshot into this new repo.

Scope: source files and provenance only. Original CyberSand/Neurounner are never written.
This script is intentionally pinned and refuses to overwrite any target file.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_REPOSITORY = "https://github.com/techrote/neurounner.git"
SOURCE_HEAD = "eb4e437f3aff7053b2b459d3c599ff3bf9e76c29"
SOURCE_PR = "https://github.com/techrote/neurounner/pull/252"

SOURCE_PREFIXES = (
    "experiments/driv3r/",
    "experiments/agent_painter/",
    "docs/driv3r-migration/",
)
HISTORICAL_IMPORT_TOOLS = (
    "scripts/migrate_cybersand_driv3r.py",
    "scripts/migrate_cybersand_auxiliary.py",
    "scripts/migrate_cybersand_historical_mz.py",
    "scripts/migrate_cybersand_stage_d.py",
)
MANIFEST_NAMES = (
    "MANIFEST.json",
    "AUXILIARY_MANIFEST.json",
    "HISTORICAL_MANIFEST.json",
    "STAGE_D_MANIFEST.json",
)


def git(cwd: Path, *args: str) -> bytes:
    p = subprocess.run(
        ["git", *args], cwd=cwd, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False
    )
    if p.returncode:
        raise RuntimeError(
            f"git {args[0]} failed: {p.stderr.decode(errors='replace')[:1800]}"
        )
    return p.stdout


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(
        b"blob " + str(len(data)).encode() + b"\0" + data
    ).hexdigest()


def import_files(source: Path) -> list[dict]:
    paths = tuple(p.removesuffix("/") for p in SOURCE_PREFIXES) + HISTORICAL_IMPORT_TOOLS
    data = git(source, "ls-files", "-s", "-z", "--", *paths)
    records: list[dict] = []
    for raw in data.split(b"\0"):
        if not raw:
            continue
        metadata, path_bytes = raw.split(b"\t", 1)
        mode, original_sha, stage = metadata.decode().split()
        rel = path_bytes.decode()
        if mode not in {"100644", "100755"} or stage != "0":
            raise RuntimeError(f"Refusing irregular Git entry: {rel}: {mode} {stage}")
        if not (rel.startswith(SOURCE_PREFIXES) or rel in HISTORICAL_IMPORT_TOOLS):
            raise RuntimeError(f"Outside strict source allowlist: {rel}")
        destination = (
            "docs/provenance/original-migration-tools/" + Path(rel).name
            if rel in HISTORICAL_IMPORT_TOOLS else rel
        )
        source_file = source / rel
        target_file = ROOT / destination
        if not source_file.is_file() or source_file.is_symlink():
            raise RuntimeError(f"Invalid source file: {rel}")
        if target_file.exists() or target_file.is_symlink():
            raise RuntimeError(f"Will not overwrite an existing DRIV3R file: {destination}")
        payload = source_file.read_bytes()
        actual_sha = git_blob_sha1(payload)
        if actual_sha != original_sha:
            raise RuntimeError(f"Source blob mismatch for {rel}")
        target_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_file, target_file)
        if mode == "100755":
            target_file.chmod(0o755)
        if git_blob_sha1(target_file.read_bytes()) != original_sha:
            raise RuntimeError(f"Destination blob mismatch for {destination}")
        records.append({
            "source_path": rel,
            "destination_path": destination,
            "git_blob_sha1": original_sha,
            "bytes": len(payload),
        })
    return records


def verify_historical_custody(records: list[dict]) -> dict:
    record_index = {r["source_path"]: r for r in records}
    retained = set()
    advanced = []
    for name in MANIFEST_NAMES:
        f = ROOT / "docs/driv3r-migration" / name
        if not f.is_file():
            raise RuntimeError(f"Missing upstream research custody ledger: {name}")
        manifest = json.loads(f.read_text(encoding="utf-8"))
        for lane in manifest["lanes"]:
            for old in lane["files"]:
                path = old["path"]
                if path not in record_index:
                    raise RuntimeError(f"Historical research file omitted: {path}")
                retained.add(path)
                historical_sha = old.get("sha1") or old.get("git_blob_sha1")
                new_sha = record_index[path]["git_blob_sha1"]
                if historical_sha and historical_sha != new_sha:
                    advanced.append({
                        "path": path,
                        "historical_git_blob_sha1": historical_sha,
                        "import_git_blob_sha1": new_sha
                    })
    if len(retained) < 1200:
        raise RuntimeError(f"Unexpected historical manifest coverage: {len(retained)}")
    return {
        "historically_accounted_files": len(retained),
        "advanced_since_original_cybersand_snapshot": advanced,
    }


def main() -> None:
    if not (ROOT / ".git").is_dir():
        raise RuntimeError("Requires a DRIV3R Git checkout")
    if (ROOT / "experiments/driv3r").exists():
        raise RuntimeError("Refusing re-import into an existing research tree")

    with tempfile.TemporaryDirectory(prefix="driv3r-upstream-snapshot-") as directory:
        source = Path(directory) / "neurounner"
        git(ROOT, "init", "--quiet", str(source))
        git(source, "remote", "add", "origin", SOURCE_REPOSITORY)
        try:
            git(source, "fetch", "--quiet", "--no-tags", "--depth=1",
                "--filter=blob:limit=1048576", "origin", SOURCE_HEAD)
        except RuntimeError:
            git(source, "fetch", "--quiet", "--no-tags", "--depth=1",
                "origin", SOURCE_HEAD)
        git(source, "checkout", "--quiet", "--detach", "FETCH_HEAD")
        actual_head = git(source, "rev-parse", "HEAD").decode().strip()
        if actual_head != SOURCE_HEAD:
            raise RuntimeError(f"Source commit moved: {actual_head} != {SOURCE_HEAD}")
        records = import_files(source)

    historical = verify_historical_custody(records)
    total_research_files = len([r for r in records if r["source_path"].startswith("experiments/")])
    if total_research_files < 1208:
        raise RuntimeError(f"Unexpected research file count: {total_research_files}")
    report = {
        "type": "immutable_cross_repo_git_blob_snapshot",
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": SOURCE_HEAD,
        "source_pr": SOURCE_PR,
        "destination_repository": "https://github.com/techrote/DRIV3R",
        "file_count": len(records),
        "research_file_count": total_research_files,
        "historical_custody": historical,
        "note": (
            "Source files only. GitHub Actions artifacts, external model "
            "checkpoints and original CI results were not copied. This "
            "is not scientific or native-game acceptance."
        ),
        "files": records,
    }
    output = ROOT / "docs/provenance/SOURCE_SNAPSHOT.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(
        f"PASS: copied and SHA-verified {len(records)} tracked files "
        f"from {SOURCE_HEAD}, incl. {historical['historically_accounted_files']} "
        "historically accounted research files",
        flush=True
    )
    print(
        f"Historical source files advanced since first migration: "
        f"{len(historical['advanced_since_original_cybersand_snapshot'])}",
        flush=True
    )


if __name__ == "__main__":
    main()
