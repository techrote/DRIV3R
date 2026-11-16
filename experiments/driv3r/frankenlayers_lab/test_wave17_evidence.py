"""Minimal source-only regression tests for historic Wave-17 archive validation."""
import hashlib
import io
import unittest
import warnings
import zipfile
from experiments.driv3r.frankenlayers_lab.audit_wave17_evidence import verify_zip, difference


def fixture(payload=b'original',hash_override=None,duplicate=False,traversal=False):
    out=io.BytesIO()
    name='../outside' if traversal else 'input.txt'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning)
        with zipfile.ZipFile(out,'w') as z:
            z.writestr(name,payload)
            if duplicate:z.writestr(name,payload)
            digest=hash_override or hashlib.sha256(payload).hexdigest()
            z.writestr('SHA256SUMS.txt',digest+'  ./'+name+'\n')
    out.seek(0)
    return zipfile.ZipFile(out)


class Wave17AuditTests(unittest.TestCase):
    def test_manifest_valid(self):
        with fixture() as z:self.assertEqual(verify_zip(z),1)
    def test_manifest_tamper_fails(self):
        with fixture(hash_override='0'*64) as z:
            with self.assertRaisesRegex(ValueError,'mismatch'):verify_zip(z)
    def test_manifest_unsafe_fails(self):
        with fixture(traversal=True) as z:
            with self.assertRaisesRegex(ValueError,'unsafe'):verify_zip(z)
    def test_manifest_duplicate_fails(self):
        with fixture(duplicate=True) as z:
            with self.assertRaisesRegex(ValueError,'duplicate'):verify_zip(z)
    def test_paired_divergence(self):
        a=[{'obs':'clear','action':'RIGHT'},{'obs':'wall','action':'RETRACE'}]
        b=[{'obs':'clear','action':'RIGHT'},{'obs':'clear','action':'BRANCH'}]
        self.assertEqual(difference(a,b,lambda x:x['obs']),1)
        self.assertEqual(difference(a,b,lambda x:x['action']),1)
        self.assertIsNone(difference(a,a,lambda x:x['action']))

if __name__=='__main__':unittest.main()
