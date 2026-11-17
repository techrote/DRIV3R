"""Small deterministic C46 qualification tests; no native gameplay claim."""
import copy, json, unittest
from pathlib import Path
import numpy as np
import torch
import c46_episode_arbitration as c46

HERE=Path(__file__).resolve().parents[1]

class C46Checks(unittest.TestCase):
    def test_input_host_only_and_reproducible(self):
        w,info,_,_,_,_=c46.worlds(c46.c41.FAMILIES,c46.c41.world,9911,1)[0]
        f=c46.inputs(w,'backtrack',.35,False)
        self.assertEqual(f.shape,(14,))
        self.assertTrue(np.isfinite(f).all())
        self.assertTrue(np.array_equal(f,c46.inputs(copy.deepcopy(w),'backtrack',.35,False)))
        self.assertFalse(np.array_equal(f,c46.inputs(w,'continue',1.,True)))

    def test_goal_count_dominates_speed_and_failure_ties(self):
        a={'goals':1,'steps':78};b={'goals':0,'steps':1}
        self.assertGreater(c46.utility(a),c46.utility(b))
        self.assertEqual(c46.utility({'goals':0,'steps':78}),c46.utility(b))

    def test_checkpoint_repeat_determinism(self):
        rng=np.random.default_rng(412)
        records=[]
        for i in range(24):
            records.append({'observation':rng.normal(size=14).astype('f').tolist(),
              'label':i%3,'graph':str(i),'goals':[int(i%3==j) for j in range(3)],
              'directive':'backtrack','strength':.35})
        one,s1=c46.train(records,442,3);two,s2=c46.train(records,442,3)
        self.assertEqual(s1,s2)
        for a,b in zip(one,two):
            for k in a.state_dict():self.assertTrue(torch.equal(a.state_dict()[k],b.state_dict()[k]))

    def test_recorded_episode_safety_and_no_guidance_equivalence(self):
        f=HERE/'output_c46/C46-online.json'
        if not f.exists():self.skipTest('requires completed local evaluation artifact')
        rows=json.loads(f.read_text())
        self.assertTrue(rows)
        self.assertTrue(all(r['host_rejections']==0 for r in rows))
        g={(r['graph'],r['model'],r['directive'],r['strength']):r for r in rows}
        for r in rows:
            if r['directive']=='none':
                a=g[(r['graph'],'C18-A0','none',0.)]
                self.assertEqual((r['actions_sha'],r['ink_sha']),(a['actions_sha'],a['ink_sha']))
        self.assertEqual(len(set(r['graph'] for r in rows)),48)

if __name__=='__main__':unittest.main()
