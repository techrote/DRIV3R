"""Restore C53 full executed textual evidence from PR #1 with SHA-256 checks."""
from __future__ import annotations
import base64, hashlib, io, json, zipfile
from pathlib import Path

def extract(root=None):
    root=Path(root) if root else Path(__file__).resolve().parent
    receipt=json.loads((root/'C53-evidence-receipt.json').read_text())
    encoded=''.join((root/'C53-executed-evidence.zip.b64').read_text().split())
    data=base64.b64decode(encoded,validate=True)
    assert hashlib.sha256(data).hexdigest()==receipt['evidence_zip_sha256']
    with zipfile.ZipFile(io.BytesIO(data)) as f:
        assert set(f.namelist())==set(receipt['executed_evidence_files'])
        for name,meta in receipt['executed_evidence_files'].items():
            assert '/' not in name and '\\' not in name and name not in ('.','..')
            value=f.read(name)
            assert len(value)==meta['bytes'] and hashlib.sha256(value).hexdigest()==meta['sha256']
            (root/name).write_bytes(value)
    print('Verified',len(receipt['executed_evidence_files']),'C53 executed evidence files')

if __name__=='__main__':extract()
