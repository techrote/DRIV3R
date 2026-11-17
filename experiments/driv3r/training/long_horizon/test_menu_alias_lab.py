"""Bounded source-faithful checks for MENU-ALIAS-001; no native/heldout access."""
from __future__ import annotations
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
from . import menu_alias_lab as lab
from .world import specs, Spec


class TestMenuAlias(unittest.TestCase):
    def test_matched_menu_and_identical_observation(self):
        sp=specs('train',1)[0]
        for turns in lab.rotated.ROTATION:
            cases, feats=lab.matched(sp,703,0,turns)
            self.assertEqual(len({tuple(cases[k]['world'].menu(12)) for k in lab.KINDS}),1)
            self.assertTrue(np.array_equal(feats['far_trap'],feats['open']))
            self.assertFalse(np.array_equal(feats['near_trap'],feats['open']))
            self.assertNotEqual(cases['far_trap']['world'].state_sha(),cases['open']['world'].state_sha())
            self.assertTrue(cases['far_trap']['witness']['forward']['forced_stop'])
            self.assertGreaterEqual(cases['open']['witness']['forward']['moves'],20)
            self.assertEqual(cases['far_trap']['world'].recoveries,12)
            self.assertEqual(cases['open']['world'].recoveries,12)

    def test_split_isolation_and_bad_variants(self):
        heldout=Spec('heldout','archipelago',9000,'wall')
        with self.assertRaises(ValueError):lab.construct(heldout,703,0,0,'open')
        with self.assertRaises(ValueError):lab.construct(specs('train',1)[0],703,0,0,'fabricated')
        with self.assertRaises(ValueError):lab.construct(specs('train',1)[0],703,99,0,'open')

    def test_source_replayed_training_labels(self):
        _,_,_,rows,_=lab.corpus(703,36,16,1)
        self.assertEqual(len(rows),32)
        for row in rows:
            self.assertTrue(lab.verify_train(row))

    def test_optimizer_checkpoint_replay_and_tamper_detection(self):
        with TemporaryDirectory() as d:
            out=Path(d)/'evidence'
            report=lab.run(out,seed=703,base_updates=2,adapt_updates=3,natural_limit=16,quick=True)
            self.assertEqual(report['episodes'],12)
            checked=lab.verify(out)
            self.assertEqual(checked['checkpoints'],4)
            self.assertEqual(checked['aliases'],1)
            self.assertEqual(checked['episodes'],12)
            p=next((out/'traces').glob('*.json'))
            obj=json.loads(p.read_text());obj['rows'][0]['requested']='STOP'
            p.write_text(json.dumps(obj))
            with self.assertRaises(ValueError):lab.verify(out)


if __name__=='__main__':unittest.main()
