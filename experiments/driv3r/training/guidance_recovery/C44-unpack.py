"""Restore losslessly compressed C44 research source and evidence (does not train)."""
import hashlib
from pathlib import Path
import zlib

ROOT = Path(__file__).resolve().parent
FILES = {
    "c44_onpolicy_pairwise.py.zlib": (
        "c44_onpolicy_pairwise.py",
        "7eb0a0472c59d9a24561428fb314db083ec733995b32fe20472c44ce7e09c153"
    ),
    "C44-compact-results.json.zlib": (
        "C44-compact-results.json",
        "454bbbc04b399e55591b14b033b75444f3f47c103792097fddaf8e76e03c67d4"
    ),
    "C44-curated-path-receipts.json.zlib": (
        "C44-curated-path-receipts.json",
        "9ebe227b8a097a4d22c3266b96d2371b629f00ac7050155eab2fbf636287149c"
    ),
}

def unpack():
    for packed_name, (target_name, expected_hash) in FILES.items():
        raw = zlib.decompress((ROOT / packed_name).read_bytes())
        digest = hashlib.sha256(raw).hexdigest()
        if digest != expected_hash:
            raise ValueError(f"SHA-256 mismatch: {packed_name}: {digest}")
        path = ROOT / target_name
        if path.exists() and path.read_bytes() != raw:
            raise FileExistsError(f"Refusing to overwrite different existing file: {path}")
        path.write_bytes(raw)
        print(f"Verified {target_name}: {digest}")

if __name__ == "__main__":
    unpack()
