"""P21 source integrity and scientific-label guardrail tests."""
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
import p21_real_lfm_bestiary as p


class P21Tests(unittest.TestCase):
    def test_metric_forward_and_backtrack(self):
        seq=[(0,0),(1,0),(2,0),(1,0),(1,1)]
        actions=[{'applied':'STRAIGHT'}]*4
        m=p.metrics(seq,actions)
        self.assertEqual(m['two_step_reversals'],1)
        self.assertEqual(m['longest_straight'],2)
        self.assertGreater(m['repeated_edge_rate'],0)
        self.assertEqual(m['distinct_cells'],4)

    def test_source_sha_gate(self):
        with tempfile.TemporaryDirectory() as td:
            target=Path(td)/'fake.zip';target.write_bytes(b'not a source archive')
            with self.assertRaisesRegex(ValueError,'SHA256 mismatch'):
                p.load('stock',target)

    def test_original_stock(self):
        import os
        source=os.getenv('P21_STOCK_ARCHIVE')
        if not source:self.skipTest('set P21_STOCK_ARCHIVE for full original artifact test')
        episodes=p.load('stock',source)
        self.assertEqual(len(episodes),12)
        self.assertEqual(sum(x['metrics']['ticks'] for x in episodes.values()),1536)
        self.assertTrue(all(x['metrics']['stop_actions']==0 for x in episodes.values()))

    def test_no_fabricated_owner_votes(self):
        import os
        path=os.getenv('P21_GENERATED_OUTPUT')
        if not path:self.skipTest('set P21_GENERATED_OUTPUT for result audit')
        result=json.loads((Path(path)/'P21_SUMMARY.json').read_text())
        self.assertEqual(result['owner_human_ratings'],0)
        self.assertEqual(result['total_model_decisions'],4608)
        import csv
        rows=list(csv.DictReader((Path(path)/'P21_BLANK_OWNER_BALLOT.csv').open()))
        self.assertEqual(len(rows),144)
        self.assertTrue(all(not x['choice_A_B_TIE_SKIP'] for x in rows))

if __name__=='__main__':unittest.main()
