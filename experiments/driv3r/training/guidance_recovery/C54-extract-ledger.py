"""Restore C54 JSON evidence from the committed, checksum-enforced text capsule."""
import base64,hashlib,json,zlib,sys
from pathlib import Path
blob=Path(sys.argv[1]).read_text().strip()
items=json.loads(zlib.decompress(base64.b64decode(blob)))
out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
for name,record in sorted(items.items()):
    assert name.startswith("C54-") and "/" not in name
    data=record["text"].encode()
    assert hashlib.sha256(data).hexdigest()==record["sha256"],name
    (out/name).write_bytes(data)
print("verified",len(items),"evidence files in",out)
