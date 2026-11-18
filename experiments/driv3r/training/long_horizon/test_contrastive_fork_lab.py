"""Split, rotation, causal witness, equal-input and replay guards."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from . import contrastive_fork_lab as f
from .world import Spec, specs, MOVE

class ContrastiveForkTests(unittest.TestCase):
    def test_rotation_inverse_and_no_unsafe_cells(self):
        for turns in f.ROTATION:
            for action in MOVE:
                self.assertEqual(f.rotate_action(f.rotate_action(action,turns),(4-turns)%4),action)
            for row in f.pair(specs('train',1)[0],691,0,turns):
                w=row['world']
                self.assertEqual(w.recoveries,12)
                self.assertEqual(w.steps,35)
                self.assertFalse(w.ink & w.protected)
                self.assertFalse(w.ink & set(w.materials))
                self.assertIn(row['forward'],w.menu(12))
                self.assertIn(row['retreat'],w.menu(12))
                self.assertEqual(len(w.decisions),35)

    def test_causal_contrast_four_orientations(self):
        for turns in f.ROTATION:
            trap,opening=f.pair(specs('train',1)[0],691,0,turns)
            self.assertTrue(trap['witness']['forward']['forced_stop'])
            self.assertFalse(trap['witness']['retreat']['forced_stop'])
            self.assertFalse(opening['witness']['forward']['forced_stop'])
            self.assertFalse(opening['witness']['retreat']['forced_stop'])
            self.assertGreaterEqual(opening['witness']['forward']['moves'],20)

    def test_matched_training_inputs_and_labels(self):
        X,M,Y,rows,meta=f.train_corpus(691,36,16,1)
        self.assertEqual(meta['natural_examples'],16)
        self.assertEqual(meta['trap_states'],16)
        self.assertEqual(meta['open_states'],16)
        self.assertEqual(X.shape[0],48)
        self.assertTrue(all(len(y)==48 and M[i,y[i]] for y in Y.values() for i in range(48)))
        self.assertTrue(all(f.verify_train_case(row) for row in rows))
        self.assertTrue(all(Y['universal_retreat'][i]!=Y['contrastive'][i]
                            for i,row in enumerate(rows,start=16) if row['kind']=='open'))
        self.assertTrue(all(Y['goal_imitation'][i]!=Y['contrastive'][i]
                            for i,row in enumerate(rows,start=16) if row['kind']=='trap'))

    def test_rejects_heldout_and_invalid_orientation(self):
        with self.assertRaises(ValueError):f.pair(Spec('heldout','archipelago',9000,'route'),691,0,0)
        with self.assertRaises(ValueError):f.pair(specs('train',1)[0],691,0,5)
        with self.assertRaises(ValueError):f.rotate_xy((48,48),-1)

    def test_quick_checkpoint_traces_and_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'run'
            r=f.run(out,seed=691,base_updates=2,adapt_updates=2,natural_limit=16,quick=True)
            receipt=f.verify(out)
            self.assertEqual(receipt['checkpoints'],4)
            self.assertEqual(receipt['episodes'],8)
            self.assertEqual(receipt['training_states'],96)
            self.assertFalse(r['sealed_heldout_opened'])
            self.assertEqual(r['native_receipts'],0)
            ep=next((out/'traces').glob('*.json'))
            e=json.loads(ep.read_text());e['rows'][0]['requested']='STOP'
            ep.write_text(json.dumps(e))
            with self.assertRaises(ValueError):f.verify(out)

if __name__=='__main__': unittest.main()
