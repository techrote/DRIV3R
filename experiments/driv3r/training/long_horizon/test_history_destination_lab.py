"""Focused, source-exact HISTORY-DESTINATION-001 tests (synthetic only)."""
import unittest
from pathlib import Path
import numpy as np
from . import history_destination_lab as lab
from .world import World, Spec, MOVE

class HistoryDestinationTests(unittest.TestCase):
    def test_split_partition_without_heldout(self):
        tr=list(lab.candidates('train'))
        dv=list(lab.candidates('development'))
        self.assertTrue(tr and dv)
        self.assertTrue(all(s.split=='train' and 1000<=s.seed<2000 for s in tr))
        self.assertTrue(all(s.split=='development' and 3000<=s.seed<4000 for s in dv))
        self.assertFalse(set(s.ident for s in tr) & set(s.ident for s in dv))

    def test_recovery_destination_is_actual_previous_host_position(self):
        sp=next(lab.candidates('train'))
        w=lab.prefix(sp,0,36)
        self.assertIsNotNone(w)
        target=w.path[-2]
        old=abs(w.cursor[0]-target[0])+abs(w.cursor[1]-target[1])
        menu,x=lab.matrix(w,target,36)
        self.assertEqual(menu,w.menu(12))
        self.assertIn('BACKTRACK',menu)
        self.assertEqual(x.shape,(lab.policy.ACTION_MAX,lab.WIDTH))
        i=menu.index('BACKTRACK')
        self.assertAlmostEqual(x[i,-4],old/9.)
        self.assertEqual(x[i,-1],1.)
        before=w.state_sha()
        masked_menu,masked=lab.matrix(w,target,36,True)
        self.assertEqual(masked_menu,menu)
        self.assertTrue(np.all(masked[:,-4:]==0.))
        self.assertEqual(before,w.state_sha())

    def test_training_corpus_balanced_and_host_legal(self):
        X,M,Y,info=lab.corpus(72)
        self.assertEqual(X.shape[1:],(lab.policy.ACTION_MAX,lab.WIDTH))
        self.assertEqual(len(M),len(Y))
        self.assertEqual(info['teacher_actions']['BACKTRACK'],len(Y)//2)
        self.assertTrue(np.all(M[np.arange(len(Y)),Y]))
        self.assertEqual(info['retreat_fraction'],.5)

    def test_all_development_prefix_transitions_are_legal(self):
        for sp in lab.candidates('development'):
            for v in range(4):
                w=lab.prefix(sp,v,128)
                if w is None:continue
                self.assertIn('BACKTRACK',w.menu(12))
                self.assertTrue(any(a in MOVE for a in w.menu(12)))
                self.assertEqual(len(w.decisions),w.steps)
                self.assertTrue(all(r['chosen'] in r['legal'] and not r['painted']
                        or r['chosen'] in r['legal'] for r in w.decisions))

    def test_checkpoint_restore(self):
        p=Path('/mnt/data/lh_run12/history_destination_evidence/checkpoints/619-visible.npz')
        if not p.exists():self.skipTest('local run checkpoint absent')
        m=lab.Model.load(p)
        self.assertEqual(m.updates,144)
        self.assertEqual(sum(x.size for x in m.p.values()),1057)
        self.assertTrue(all(np.isfinite(z).all() for z in m.p.values()))

if __name__=='__main__':unittest.main()
