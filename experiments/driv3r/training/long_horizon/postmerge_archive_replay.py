"""Portable, CPU-bounded requalification of one merged DRIV3R evidence archive.

This exercises the existing MATERIAL-EPOCH-001 verifier against the actual merged
research source. It deliberately does not download checkpoints, dispatch Actions,
change gameplay, open structural HELDOUT, or accept source-import success as a
substitute for replay. Checkpoints remain external, supplied inside the archive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import stat
import tempfile
import time
import zipfile
from pathlib import Path

from .material_epoch_verify_publication import verify as verify_material_epoch

VERSION = 'postmerge-material-epoch-replay/1'
BASELINE_MAIN = '85df84666a8bad18f0092b930c08557b8010687f'
PATH_ROOT = 'experiments/driv3r/training/long_horizon/'
LEDGER = PATH_ROOT + 'MATERIAL_EPOCH_EVIDENCE_2026-11-18.json'
RECEIPTS = PATH_ROOT + 'MATERIAL_EPOCH_RECEIPTS_2026-11-18.jsonl.gz'
CHECKPOINTS = tuple(f'external_checkpoints/{seed}-conditioned.npz' for seed in (711, 712))
REQUIRED = (LEDGER, RECEIPTS, *CHECKPOINTS)
MAX_ARCHIVE_BYTES = 30 * 1024 * 1024
MAX_MEMBER_BYTES = 12 * 1024 * 1024


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def validate_members(items: list[zipfile.ZipInfo]) -> None:
    """Reject duplicate, unsafe, symlinked and unexpectedly large archive members."""
    paths = [item.filename for item in items]
    if len(paths) != len(set(paths)):
        raise ValueError('Duplicate ZIP member names')
    for item in items:
        name = item.filename
        if (name.startswith('/') or '\\' in name or any(part in ('', '.', '..')
                for part in name.rstrip('/').split('/')) or item.file_size > MAX_MEMBER_BYTES):
            raise ValueError(f'Unsafe ZIP member: {name}')
        mode = (item.external_attr >> 16) & 0o170000
        if mode == stat.S_IFLNK:
            raise ValueError(f'ZIP symlink rejected: {name}')
    if any(path not in paths for path in REQUIRED):
        raise ValueError('Missing required ledger, receipts or checkpoint')


def run(archive: Path, workers: int = 2) -> dict:
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError('CPU workers must be 1..4')
    if not archive.is_file() or archive.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError('Missing or oversized local evidence archive')
    start = time.perf_counter()
    original_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    with zipfile.ZipFile(archive) as z:
        validate_members(z.infolist())
        if z.testzip() is not None:
            raise ValueError('Archive CRC mismatch')
        # Pin actual git-blob identities for the merged source and archive source.
        for name in ('material_epoch_lab.py', 'material_epoch_verify_publication.py'):
            path = PATH_ROOT + name
            data = z.read(path)
            local = Path(__file__).with_name(name).read_bytes()
            if git_blob_sha(data) != git_blob_sha(local):
                raise ValueError(f'Merged source mismatch: {name}')
        with tempfile.TemporaryDirectory(prefix='driv3r-postmerge-') as directory:
            root = Path(directory)
            for name in REQUIRED:
                dst = root / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(z.read(name))
            result = verify_material_epoch(root / LEDGER, root / RECEIPTS,
                                           root / 'external_checkpoints', workers)
    if (result['episodes'], result['action_receipts'], result['probability_vectors']) != (320, 16140, 8542):
        raise ValueError('Historical evidence cardinality changed')
    return {'version': VERSION, 'merged_baseline_commit': BASELINE_MAIN,
            'original_archive_sha256': original_hash, 'verification': result,
            'wall_s': round(time.perf_counter()-start, 3),
            'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'heldout_opened': False, 'native_receipts': 0, 'optimizer_updates': 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(run(args.archive, args.workers), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
