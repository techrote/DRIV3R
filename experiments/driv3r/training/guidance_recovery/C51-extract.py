"""Extract SHA-checked C51 research source, compact evidence, and optional full ledger."""
import base64, hashlib, io, pathlib, tarfile, zlib
ROOT=pathlib.Path(__file__).resolve().parent
BUNDLES={
    "C51-publication-bundle.zlib.b64":"5f2c2fa852d929edb3d6cff6f3862733648a5e9767a720ef9ebc03a96cd26c56",
    "C51-full-executed-ledger.zlib.b64":"b880cb0edcb2ffdb9da3dcc78a70ecba41a74ead720e46edce4aa93025e117d8",
}
for filename,expected in BUNDLES.items():
    content=(ROOT/filename).read_bytes()
    assert hashlib.sha256(content).hexdigest()==expected,filename
    data=zlib.decompress(base64.b64decode(content))
    with tarfile.open(fileobj=io.BytesIO(data),mode="r:") as archive:
        for member in archive.getmembers():
            candidate=pathlib.PurePosixPath(member.name)
            assert member.isfile() and not candidate.is_absolute() and ".." not in candidate.parts
            dest=ROOT.joinpath(*candidate.parts)
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes(archive.extractfile(member).read())
print("C51 source, report, gallery, full executed ledger extracted with bundle SHA-256 verification")
