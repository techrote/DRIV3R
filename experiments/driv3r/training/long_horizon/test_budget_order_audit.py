"""Manual original-archive verification; skips when no original artifact is mounted."""
import unittest
from collections import defaultdict
from pathlib import Path
from .budget_order_audit import load_verified,analyze
ARCHIVE=Path(__file__).parent/'DRIV3R_BUDGET_CALIBRATION_20261118_QUALIFIED_EVIDENCE.zip'
class BudgetOrderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not ARCHIVE.exists():raise unittest.SkipTest('Original verified ZIP required; not in Git')
        cls.rows,cls.samples,_=load_verified(ARCHIVE)
    def test_paired_order(self):
        a=analyze(self.rows,self.samples)
        self.assertEqual(a['source_episodes'],2880)
        self.assertEqual(a['sampled_receipts'],342)
        self.assertEqual([a['census'][k]['order_inversions'] for k in ('calibrated','fixed3','budget_blind')],[0,100,0])
        self.assertEqual(a['census']['fixed3']['inversions_1_to_3'],44)
        self.assertEqual(a['census']['fixed3']['inversions_3_to_5'],56)
    def test_tamper(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as d:
            p=Path(d)/'tampered.zip'
            p.write_bytes(ARCHIVE.read_bytes()+b'modified')
            with self.assertRaises(ValueError):load_verified(p)
if __name__=='__main__':unittest.main()
