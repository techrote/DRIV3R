"""Restore full C52 text evidence from the Git-committed lossless package."""
from pathlib import Path
import argparse,base64,hashlib,json,zlib

def extract(bundle, manifest, destination):
    blob=Path(bundle).read_text().strip()
    data=json.loads(zlib.decompress(base64.b64decode(blob)).decode('utf-8'))
    expected=json.loads(Path(manifest).read_text())['compressed_text_evidence']
    root=Path(destination).resolve();root.mkdir(parents=True,exist_ok=True)
    assert set(data)==set(expected),'manifest entries do not match'
    for relative,payload in data.items():
        p=(root/relative).resolve()
        assert root in p.parents and not p.exists(),f'unsafe or existing output: {p}'
        b=payload.encode('utf-8')
        assert hashlib.sha256(b).hexdigest()==expected[relative]['sha256']
        assert len(b)==expected[relative]['bytes']
        p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
    print('Restored and SHA-256-verified',len(data),'text evidence files')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('bundle');parser.add_argument('manifest');parser.add_argument('destination');a=parser.parse_args()
    extract(a.bundle,a.manifest,a.destination)
