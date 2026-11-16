"""Bounded D021 curriculum and legality regression tests."""
import unittest
import numpy as np
from .onpolicy_repair_021 import check_splits, combine, TRAIN, VALID, TEST, CONFIRM

class TestD021(unittest.TestCase):
 def test_splits_are_disjoint(self):
  check_splits()
  self.assertFalse(set(TRAIN)&set(TEST));self.assertFalse(set(VALID)&set(CONFIRM))
 def test_mix_contains_expected_rows(self):
  x=np.ones((4,100),np.float32);base={'x':x,'t':np.ones((4,6),np.float32),
    'm':np.ones((4,6),np.uint8),'y':np.asarray([0,3,1,5])}
  on={k:v.copy() for k,v in base.items()};on['d']=np.asarray([0,1,1,0],bool)
  mix=combine(base,on,disagreement_weight=2,all_weight=1)
  self.assertEqual(mix['x'].shape,(12,100))
  self.assertEqual(mix['y'].tolist()[-4:],[3,1,3,1])
 def test_limits(self):
  d={'x':np.ones((2,100)),'t':np.ones((2,6)),
     'm':np.ones((2,6)),'y':np.ones(2)}
  on=dict(d,d=np.ones(2,bool))
  with self.assertRaises(ValueError):combine(d,on,10,1)
if __name__=='__main__':unittest.main()
