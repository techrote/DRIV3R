"""Verify and extract the C58 lossless executed/scientific evidence bundle."""
import base64,hashlib,json,zlib
from pathlib import Path

def extract(bundle,destination):
    wrapper=json.loads(Path(bundle).read_text())
    assert wrapper['schema']=='DRIV3R-C58-LEDGER-ZLIB-1'
    raw=zlib.decompress(base64.b64decode(wrapper['archive']))
    assert hashlib.sha256(raw).hexdigest()==wrapper['archive_sha256']
    entries=json.loads(raw)
    assert set(entries)==set(wrapper['files'])
    root=Path(destination);root.mkdir(parents=True,exist_ok=True)
    for name,encoded in entries.items():
        assert name.endswith('.json') and '/' not in name and '\\' not in name
        binary=base64.b64decode(encoded)
        assert hashlib.sha256(binary).hexdigest()==wrapper['files'][name]
        (root/name).write_bytes(binary)
    print('Verified',len(entries),'full C58 JSON evidence documents')

if __name__=='__main__':
    import sys
    extract(*sys.argv[1:])
