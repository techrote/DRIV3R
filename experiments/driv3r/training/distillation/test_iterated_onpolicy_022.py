"""D022 split, weighting, legality and paired-tail invariants."""
import unittest
import numpy as np
from .iterated_onpolicy_022 import check_splits, mix, tail, TRAIN, TEST, CONFIRM, STRESS
from .lab import STYLES

class TestD022(unittest.TestCase):
 def test_disjoint_predeclared_sets(self):
  check_splits()
  self.assertFalse(set(TRAIN)&set(TEST));self.assertFalse(set(CONFIRM)&set(STRESS))
 def test_trap_hard_oversamples_only_trap_disagreements(self):
  base={'x':np.ones((2,100),np.float32),'t':np.ones((2,6),np.float32),
        'm':np.ones((2,6),np.uint8),'y':np.array([0,3])}
  x=np.vstack([np.ones(100),np.ones(100)*2,np.ones(100)*3]).astype(np.float32)
  new={k:v for k,v in base.items()};new={**new,'x':x,'t':np.ones((3,6),np.float32),
       'm':np.ones((3,6),np.uint8),'y':np.array([0,1,3]),
       'd':np.array([True,True,False]),'trap':np.array([True,False,True])}
  a=mix(base,base,new,'uniform');b=mix(base,base,new,'trap_hard')
  self.assertEqual(len(a['y']),9);self.assertEqual(len(b['y']),13)
  self.assertEqual(int((b['x'][:,0]==1).sum()-(a['x'][:,0]==1).sum()),3)
 def test_nonexistent_curriculum(self):
  with self.assertRaises(ValueError):mix({}, {}, {},'bad')
 def test_tail_is_paired_by_seed_and_style(self):
  rows=[]
  for seed in (1,2):
   for style in STYLES:
    for m,cells in [('d017',10),('d021',9),('d022',12)]:
     rows.append({'seed':seed,'style':style,'case':'trap','model':m,'cells':cells})
  self.assertEqual(tail(rows,'d022','d017')['mean'],2)
  self.assertEqual(tail(rows,'d022','d021')['seeds_loss_ge_5'],0)
if __name__=='__main__':unittest.main()
