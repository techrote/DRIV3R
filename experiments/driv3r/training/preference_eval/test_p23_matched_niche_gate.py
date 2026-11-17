"""Focused invariant tests for original-source and blind-pair construction."""
import csv
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import p23_matched_niche_gate as m

ROOT=Path(os.environ.get('P23_FIXTURES','__missing_p23_fixtures__'))
SRC=ROOT/'source-p21/original_sources'
HIST=ROOT/'source-p18/P18'
ARGS=(SRC/'gp350012-stock-source.zip',SRC/'gp350012-14244806761.zip',
      SRC/'gp350012-14245172930.zip',HIST/'driv3r_wormpaths_source_manifest.json',
      HIST/'original_historical_artifacts')

class P23Tests(unittest.TestCase):
    def test_repeated_four_cell_loop_not_creative(self):
        p=[(0,0),(1,0),(1,1),(0,1)]*34
        d=m.traits(p)
        self.assertEqual(d['unique_cells'],4)
        self.assertIn('tiny_stationary_loop',d['screening_flags'])
        self.assertIn('high_turn_confinement',d['screening_flags'])

    def test_nonmoving_path_is_not_turning(self):
        d=m.traits([(0,0)]*17)
        self.assertEqual(d['turn90_rate'],0)
        self.assertEqual(d['net_manhattan'],0)
        self.assertEqual(d['unique_cells'],1)

    def test_straight_line(self):
        d=m.traits([(i,0) for i in range(129)])
        self.assertEqual(d['longest_straight'],128)
        self.assertEqual(d['turn90_rate'],0)
        self.assertEqual(d['unique_cells'],129)
        self.assertEqual(d['screening_flags'],[])

    def test_paired_blind_svg_has_no_source_identity(self):
        a={'positions':[(0,0),(1,0),(1,1)],'ink':[True,True]}
        b={'positions':[(0,0),(0,1),(1,1)],'ink':[True,True]}
        result=m.svg([{'blind_id':'X1234567890','arms':(a,b)}])
        self.assertIn('210mm',result)
        self.assertIn('X1234567890',result)
        self.assertNotIn('gp350',result.lower())
        self.assertNotIn('franken',result.lower())

    @unittest.skipUnless(ARGS[0].is_file(), 'requires P23_FIXTURES source ZIPs')
    def test_pinned_corrupted_source_zip_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'bad.zip'; f.write_bytes(ARGS[0].read_bytes()+b'!')
            with self.assertRaisesRegex(ValueError,'SHA256 mismatch'):
                m.load_lfm('stock',f)

    @unittest.skipUnless(ARGS[0].is_file() and ARGS[3].is_file(), 'requires P23_FIXTURES original sources')
    def test_original_audit_and_outputs_are_stable(self):
        with tempfile.TemporaryDirectory() as d:
            o1=Path(d)/'one'; o2=Path(d)/'two'
            a=m.run(*ARGS,o1); b=m.run(*ARGS,o2)
            self.assertEqual(a,b)
            self.assertEqual(a['source_original_trajectories'],50)
            self.assertEqual(a['matched_true_nonidentical_pairs'],14)
            self.assertEqual(a['full_LFM128_identical_excluded'],8)
            self.assertEqual(a['human_owner_labels'],0)
            self.assertEqual(len(list(o1.glob('P23_BLIND_A4_*.svg'))),8)
            self.assertEqual({f.name:f.read_bytes() for f in o1.iterdir()},
                             {f.name:f.read_bytes() for f in o2.iterdir()})
            reveal=json.loads((o1/'P23_REVEAL.json').read_text())
            self.assertEqual(sum(x['identical_visual_catch'] for x in reveal),2)
            self.assertEqual(len({x['blind_id'] for x in reveal}),16)
            with (o1/'P23_BLANK_OWNER_BALLOT.csv').open() as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),64)
            self.assertTrue(all(not x['choice_A_B_TIE_SKIP'] for x in rows))
            self.assertTrue(all(not x['owner_notes'] for x in rows))

if __name__=='__main__': unittest.main()
