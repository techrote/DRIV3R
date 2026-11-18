"""Bounded checks for cross-seed suffix robustness and exact-source custody."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from . import retrace_seed_transfer_lab as lab
from . import retrace_realism_lab as prior
from .world import Spec, World


class SeedTransferTests(unittest.TestCase):
    def test_sealed_partitions_and_original_source(self):
        for name in ('previous_development','fresh_development'):
            self.assertTrue(all(x.split=='development' for x in prior.specimen_specs(name)))
        with self.assertRaises(ValueError): prior.specimen_specs('heldout')
        self.assertIn('policy.py',lab.pins())
        self.assertIn('retrace_seed_transfer_lab.py',lab.pins())

    def test_robustness_not_same_seed_only(self):
        f=Path(os.environ.get('DRIV3R_SEED_TRANSFER_EVIDENCE','/nonexistent'))/'report.json'
        if not f.is_file():self.skipTest('local campaign missing')
        data=json.loads(f.read_text())['summary']
        self.assertEqual(data['overall']['eligible'],87)
        self.assertEqual(data['overall']['original_material'],5)
        self.assertEqual(data['overall']['original_material_survived'],5)
        self.assertEqual(data['overall']['six_material'],5)
        self.assertEqual(data['by_prefix_seed']['619']['six_material'],0)
        self.assertEqual(data['by_prefix_seed']['620']['six_material'],5)
        self.assertEqual(sum(data['executed_action_distribution'].values()),data['forks']['decisions'])
        self.assertNotIn('RELOCATE_START',data['executed_action_distribution'])

    def test_tampered_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'f.txt').write_text('a');(p/'SHA256SUMS.json').write_text('{}')
            with self.assertRaises(ValueError):lab.verify_original(p)

    def test_outcome_thresholds(self):
        h=lambda hit,dist:{'goal_hit_step':hit,'final_distance':dist}
        self.assertTrue(prior.robust_margin(h(4,0),h(8,0)))
        self.assertFalse(prior.robust_margin(h(4,0),h(7,0)))
        self.assertFalse(prior.robust_margin(h(None,1),h(None,8)))
        self.assertTrue(prior.robust_margin(h(None,1),h(None,9)))

    def test_host_menu_untouched(self):
        w=World.create(Spec('development','pickets',3002,'wall'))
        w.apply('MOVE_E',12);w.apply('MOVE_N',12)
        before=w.state_sha();menu=w.menu(12)
        self.assertTrue('BACKTRACK' in menu)
        self.assertNotIn('RELOCATE_START',[a for a in menu if a in prior.MOVE or a=='BACKTRACK'])
        self.assertEqual(w.state_sha(),before)

if __name__=='__main__':unittest.main()
