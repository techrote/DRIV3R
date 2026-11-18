"""FT-022 deterministic synthetic host and checkpoint custody tests."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile
import numpy as np
SPEC=importlib.util.spec_from_file_location('ft022',Path(__file__).with_name('ft022_sequential_transfer.py'))
f=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(f)

class Tests(unittest.TestCase):
 def test_parent_hash_and_legal_world(self):
  p=Path('/mnt/data/ft021_local_original_evidence.zip')
  if not p.exists():self.skipTest('FT021 original archive not present')
  with ZipFile(p) as z:self.assertEqual(set(f.parents(z)),set(f.NICHES))
  for domain in ('train','open','guarded'):
   w=f.World(domain,f.seed(domain,'test'))
   self.assertEqual(w.state(),f.World(domain,f.seed(domain,'test')).state())
   self.assertFalse(w.protected&w.painted)
   self.assertTrue(f.route(w.start,w.goal1,w.protected|w.obstacles))
   for action in w.legal():
    if action!='STOP':self.assertEqual(w.features([action]).shape,(1,20))
 def test_retrace_is_real_parent_and_protected_invariant(self):
  w=f.World('open',f.seed('retrace','test'))
  a=next(a for a in w.legal() if a in f.DELTA)
  w.step(a)
  self.assertIn('RETRACE',w.legal())
  parent=w.trail[-2]
  self.assertEqual(w.after('RETRACE'),parent)
  w.step('RETRACE')
  self.assertEqual(w.cursor,parent)
  self.assertFalse(w.painted&(w.protected|w.obstacles))
 def test_legal_replay_and_goal_switch(self):
  w=f.World('guarded',f.seed('teacher','test'))
  for _ in range(f.MAX_STEPS):
   a=f.teacher(w)
   if a is None:break
   self.assertIn(a,w.legal());w.step(a)
   if w.stopped:break
  self.assertTrue(w.switched)
  self.assertEqual(w.ending,'final_goal')
 def test_split_disjoint_and_optimizer_effect(self):
  train={f.seed(f.ID,n,'train',i) for n in f.NICHES for i in range(f.WORLD_TRAIN)}
  held={f.seed(f.ID,n,d,i) for n in f.NICHES for d in ('open','guarded') for i in range(f.WORLD_TEST)}
  self.assertFalse(train&held)
  p=f.Policy(dict(W1=np.ones((20,24))*.03,b1=np.zeros(24),W2=np.ones(24)*.04,b2=np.zeros(1)))
  before=p.p['W1'].copy()
  examples,_=f.training_set('navigator','unit',2,True)
  self.assertGreater(len(examples),0)
  for X,y in examples[:10]:p.teach(X,y)
  self.assertGreater(float(np.linalg.norm(p.p['W1']-before)),0)
  self.assertEqual(p.updates,min(10,len(examples)))
 def test_rollout_legality_replay(self):
  p=Path('/mnt/data/ft021_local_original_evidence.zip')
  if not p.exists():self.skipTest('FT021 original archive not present')
  with ZipFile(p) as z:model=f.parents(z)['navigator']
  e,w=f.rollout(model,'guarded',f.seed('rollout','test'),trace=True)
  self.assertLessEqual(e['steps'],f.MAX_STEPS)
  self.assertEqual(e['protected_edited'],0)
  self.assertEqual(len(e['receipts']),e['steps'])

if __name__=='__main__':unittest.main()
