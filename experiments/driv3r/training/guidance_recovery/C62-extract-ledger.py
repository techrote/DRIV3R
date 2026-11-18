"""Losslessly extract the C62 JSON evidence ledger (source/evidence only)."""
import base64,json,sys,zlib
from pathlib import Path
src=Path(sys.argv[1] if len(sys.argv)>1 else 'C62-executed-ledger.zlib.b64')
dst=Path(sys.argv[2] if len(sys.argv)>2 else 'C62-executed-ledger.json')
ledger=json.loads(zlib.decompress(base64.b64decode(src.read_text())))
assert ledger['schema']=='driv3r-c62-executed-ledger/1'
dst.write_text(json.dumps(ledger,indent=2,sort_keys=True)+'\n')
print(dst,'records',len(ledger['records']),'executed suffixes',ledger['summary']['executed_suffixes'])
