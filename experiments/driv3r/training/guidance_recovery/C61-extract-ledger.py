"""Extract complete lossless C61 source-executed synthetic-host evidence JSON."""
import base64,hashlib,json,sys,zlib
from pathlib import Path
if __name__=='__main__':
    if len(sys.argv)!=3:raise SystemExit('usage: C61-extract-ledger.py INPUT.zlib.b64 OUTPUT.json')
    source=Path(sys.argv[1]);raw=zlib.decompress(base64.b64decode(source.read_text()))
    data=json.loads(raw)
    assert data['schema']=='driv3r-c61-full-executed-ledger/1'
    assert len(data['results'])*6==data['summary']['suffixes']
    Path(sys.argv[2]).write_bytes(raw+b'\n')
    print('rows',len(data['results']),'suffixes',data['summary']['suffixes'],'sha256',hashlib.sha256(raw).hexdigest())
