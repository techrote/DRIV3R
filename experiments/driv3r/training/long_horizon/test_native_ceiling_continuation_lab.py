"""Focused source-only conformance for NATIVE-CEILING-CONTINUATION-001."""
import tempfile,unittest
from pathlib import Path
from . import native_ceiling_continuation_lab as lab
from .world import World,Spec,replay
from .policy import Ranker

class NativeCeilingTests(unittest.TestCase):
    def test_partition(self):
        specs=lab.dev_specs()
        self.assertEqual(len(specs),12)
        self.assertTrue(all(s.split=='development' and 3000<=s.seed<4000 for s in specs))
    def test_no_host_change(self):
        s=lab.dev_specs()[-1]; w=World.create(s)
        self.assertIn('STOP',w.menu(12))
        for i in range(6):
            self.assertIn('MOVE_E',w.menu(12))
            action,forced=lab.intervention(w,i,'STOP',True)
            self.assertEqual(action,'MOVE_E')
            w.apply(action,12)
        self.assertEqual(w.cursor,(54,48))
        self.assertEqual(replay(s,w.decisions,12).state_sha(),w.state_sha())
    def test_strict_64_censor_does_not_modify_world(self):
        s=lab.dev_specs()[1]
        a,t=lab.one_episode(s,128,'random_persistent',None,False,0)
        self.assertEqual(len(t['events']),a['steps'])
        self.assertLessEqual(a['prefix_64_logical_moves'],64)
        self.assertTrue(all(e['chosen'] in e['legal'] for e in t['events']))
        self.assertEqual(sum(e['chosen']=='RELOCATE_START' for e in t['events']),a['relocations'])
        self.assertEqual(replay(s,t['events']).state_sha(),a['end_sha256'])
    def test_frozen_model_real_probabilities(self):
        with tempfile.TemporaryDirectory() as d:
            m=Ranker(1051);p=Path(d)/'x.npz';m.checkpoint(p)
            out,tr=lab.one_episode(lab.dev_specs()[0],36,'neural',Ranker.restore(p),True,1051)
            self.assertEqual(len(tr['probabilities']),out['steps'])
            self.assertTrue(all(set(x['p'])==set(x['legal']) for x in tr['probabilities']))
            self.assertEqual(replay(lab.dev_specs()[0],tr['events']).state_sha(),out['end_sha256'])

if __name__=='__main__':unittest.main()
