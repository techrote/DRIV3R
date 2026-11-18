"""Losslessly reconstruct the exact C50 experiment source from its Git-stored bundle.

Only the three expected Python source files are accepted; hashes checked before writing.
Usage: python C50-extract.py C50-source-bundle.zlib.b64 /path/to/destination
"""
from __future__ import annotations
import base64, hashlib, io, sys, tarfile, zlib
from pathlib import Path

EXPECTED = {
    "c50_continuous_recovery.py": "97fd4077a6771714515f612fabd2fda365ddb6f4f08cb96079d008331e0d7e2f",
    "test_c50_continuous_recovery.py": "3b099fb043eeee41063ba36924b7375bdbfc64879b550b4593c3757d3651ab9b",
    "c50_gallery.py": "f3f82aa7eca0e202d165d62167c8179073ecd089fa923412f69005647cccb8f3",
}

def extract(bundle: Path, dest: Path):
    data = zlib.decompress(base64.b64decode(bundle.read_text().strip(), validate=True))
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as tar:
        names = {entry.name for entry in tar.getmembers()}
        if names != set(EXPECTED):
            raise ValueError("unexpected bundle members: " + str(names))
        validated = {}
        for name, expected in EXPECTED.items():
            item = tar.getmember(name)
            if not item.isfile():
                raise ValueError("non-file member " + name)
            content = tar.extractfile(item).read()
            digest = hashlib.sha256(content).hexdigest()
            if digest != expected:
                raise ValueError("SHA-256 mismatch " + name)
            validated[name] = content
    dest.mkdir(parents=True, exist_ok=True)
    for name, content in validated.items():
        target = dest / name
        if target.exists() and target.read_bytes() != content:
            raise FileExistsError("refusing to overwrite a different file: " + str(target))
        target.write_bytes(content)
        print(name, EXPECTED[name])
    return len(validated)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python C50-extract.py C50-source-bundle.zlib.b64 SOURCE_DIR")
    extract(Path(sys.argv[1]), Path(sys.argv[2]))
