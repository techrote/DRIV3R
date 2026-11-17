"""C31 host-verified temporal exploration regression tests (offline only)."""
from __future__ import annotations
import hashlib, unittest
import numpy as np
import torch
from collections import deque
from c31_observed_progress import u_world,legal_progress,train
from c28_temporal_option import executed_world

class TestC31(unittest.TestCase):
    def test_oracle_and_withheld_depths(self):
        for i in (0,1,2,3,8,9,10,11):
            for kind in ('decoy','tunnel'):
                for depth in (2,3,4):
                    w=u_world(i,kind,1,9911+i,depth)
                    q=deque([(w.pos,0)]);seen={w.pos};length=None
                    while q:
                        p,d=q.popleft()
                        if p==w.goal:length=d;break
                        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
                            k=(p[0]+dx,p[1]+dy)
                            if k not in seen and w.allowed(k):seen.add(k);q.append((k,d+1))
                    self.assertEqual(length,9 if kind=='tunnel' else 27)
                    self.assertEqual(w.invalid_attempts,0)
                    self.assertEqual(len(w.history),27)
    def test_initial_causal_alias(self):
        for i in range(8,12):
            for depth in (2,3,4):
                a=u_world(i,'decoy',1,1901,depth);b=u_world(i,'tunnel',1,1901,depth)
                for step in range(depth-1):
                    am=a.candidates(.35);bm=b.candidates(.35)
                    self.assertEqual([(c.action,c.target,c.legal) for c in am],
                                     [(c.action,c.target,c.legal) for c in bm])
                    ia=legal_progress(a,am);ib=legal_progress(b,bm)
                    self.assertEqual(ia,ib)
                    self.assertTrue(ia)
                    self.assertTrue(a.apply(am[ia[0]]) and b.apply(bm[ib[0]]))
    def test_visited_goalward_action_is_legal(self):
        w=executed_world(0,'tunnel',1,234,False)
        for k in range(3):
            m=w.candidates(.35);cand=legal_progress(w,m)
            self.assertTrue(cand);self.assertTrue(w.apply(m[cand[0]]))
        menu=w.candidates(.35)
        self.assertTrue(legal_progress(w,menu))
        self.assertFalse(legal_progress(w,menu,False))
    def test_repeatable_optimizer_nontrivial(self):
        a,ma,_=train(310011,3);b,mb,_=train(310011,3)
        self.assertGreater(ma['updates'],0)
        self.assertEqual(ma,mb)
        for k,v in a.state_dict().items():self.assertTrue(torch.equal(v,b.state_dict()[k]))

if __name__=='__main__':unittest.main()
