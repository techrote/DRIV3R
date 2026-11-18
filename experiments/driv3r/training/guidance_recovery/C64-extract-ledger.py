"""Extract full JSON from committed compact C64 research ledger. Stdlib only."""
import base64,hashlib,json,sys,zlib
from pathlib import Path
p=Path(sys.argv[1] if len(sys.argv)>1 else Path(__file__).with_name('C64-executed-ledger.zlib.b64'))
raw=zlib.decompress(base64.b64decode(p.read_text()))
data=json.loads(raw)
assert data['schema']=='driv3r-c64-executed-ledger/1'
assert len(data['records'])==data['summary']['matched_condition_rows']
assert len(data['traces'])==data['summary']['replay_traces']
print(json.dumps({'sha256':hashlib.sha256(raw).hexdigest(),'records':len(data['records']),
  'traces':len(data['traces'])},sort_keys=True),file=sys.stderr)
sys.stdout.buffer.write(raw)
