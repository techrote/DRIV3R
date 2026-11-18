"""Expand the lossless source-verified C63 executed host evidence ledger."""
import base64,hashlib,json,sys,zlib
from pathlib import Path
if __name__=='__main__':
    source=Path(sys.argv[1]);out=Path(sys.argv[2])
    data=zlib.decompress(base64.b64decode(source.read_text()))
    parsed=json.loads(data)
    assert parsed['schema']=='driv3r-c63-full-executed-ledger/1'
    out.write_text(json.dumps(parsed,indent=2,sort_keys=True)+'\n')
    print('bytes',len(data),'sha256',hashlib.sha256(data).hexdigest(), 'path',out)
