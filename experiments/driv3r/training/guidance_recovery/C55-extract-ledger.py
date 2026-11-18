"""Lossless, integrity-checked recovery of C55 executed evidence JSON documents."""
import argparse,base64,hashlib,json,zlib
from pathlib import Path

def extract(encoded,root):
    wrapper=json.loads(Path(encoded).read_text())
    assert wrapper['schema']=='C55-ledger-zlib-base64/1'
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    raw=zlib.decompress(base64.b64decode(wrapper['archive']))
    assert hashlib.sha256(raw).hexdigest()==wrapper['archive_sha256']
    files=json.loads(raw)
    assert set(files)==set(wrapper['files'])
    for name,base64_content in files.items():
        assert name.endswith('.json') and '/' not in name and '\\' not in name
        payload=base64.b64decode(base64_content)
        assert hashlib.sha256(payload).hexdigest()==wrapper['files'][name]
        (root/name).write_bytes(payload)
    print('Verified',len(files),'executed evidence documents in',root)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('encoded');p.add_argument('output_dir')
    a=p.parse_args();extract(a.encoded,a.output_dir)
