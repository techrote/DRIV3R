"""Decode complete text-form C59 synthetic-host evidence ledger."""
from pathlib import Path
import base64,zlib,json,sys,hashlib
p=Path(sys.argv[1]);dest=Path(sys.argv[2])
raw=zlib.decompress(base64.b64decode(''.join(p.read_text().split())))
obj=json.loads(raw)
assert sorted(obj)==['C59-online.json','C59-traces.json']
dest.mkdir(parents=True,exist_ok=True)
for name,value in obj.items():
    (dest/name).write_text(json.dumps(value,sort_keys=True,separators=(',',':'))+'\n')
print('decoded_sha256',hashlib.sha256(raw).hexdigest(),'records',len(obj['C59-online.json']))
