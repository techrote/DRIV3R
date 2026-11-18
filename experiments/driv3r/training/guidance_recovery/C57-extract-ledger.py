"""Integrity-checked extraction of the C57 exact executed synthetic evidence."""
import argparse,base64,hashlib,json,zlib
from pathlib import Path

def extract(src,dest):
    w=json.loads(Path(src).read_text());assert w['schema']=='driv3r-c57-ledger-zlib-base64/1'
    raw=zlib.decompress(base64.b64decode(w['data']))
    assert hashlib.sha256(raw).hexdigest()==w['payload_sha256']
    files=json.loads(raw);assert set(files)==set(w['files'])
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    for name,b64 in files.items():
        assert '/' not in name and '\\' not in name and name.endswith('.json')
        data=base64.b64decode(b64)
        assert hashlib.sha256(data).hexdigest()==w['files'][name]
        (dest/name).write_bytes(data)
    print('Verified',len(files),'C57 executed documents')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('src');ap.add_argument('dest')
    ns=ap.parse_args();extract(ns.src,ns.dest)
