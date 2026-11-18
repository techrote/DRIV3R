"""Extract C60 full executed synthetic-host ledger and verify source SHA."""
import argparse,base64,hashlib,json,zlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('ledger');p.add_argument('output');p.add_argument('--source')
a=p.parse_args();obj=json.loads(zlib.decompress(base64.b64decode(Path(a.ledger).read_text())).decode());assert obj['schema']=='driv3r-c60-full-executed-ledger/1'
if a.source:assert hashlib.sha256(Path(a.source).read_bytes()).hexdigest()==obj['source_sha256']
Path(a.output).write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n');print('verified C60 records',len(obj['records']),'traces',len(obj['traces']))
