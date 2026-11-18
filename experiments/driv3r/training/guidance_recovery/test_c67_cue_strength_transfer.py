"""C67 internal regression tests for synthetic cue-strength selection."""
import unittest, tempfile
from pathlib import Path
import numpy as np
import torch
import c67_cue_strength_transfer as m

class C67(unittest.TestCase):
    def test_seed_and_feature_boundary(self):
        self.assertEqual(m.FEATURES,12)
        self.assertEqual(m.WEAK,('backtrack',.35))
        self.assertEqual(m.STRONG,('backtrack',1.))
        self.assertEqual(m.UPDATES,192)

    def test_identical_training_budget_and_reproducibility(self):
        rows=[]
        for f in ('wilson','kruskal'):
            for d in m.DROPS:
                for i in range(5):
                    x=np.asarray([1.,i/10,d,0,0,0,0,0,0,0,0,0],np.float32)
                    rows.append({'x':x,'adv':float(i-1),'family':f,'drop':d,'graph':str(i)})
        torch.set_num_threads(1)
        a,alog=m.fit(rows,'aligned');a2,_=m.fit(rows,'aligned')
        b,blog=m.fit(rows,'shuffled');b2,_=m.fit(rows,'shuffled')
        self.assertEqual(len(alog),5);self.assertEqual(len(blog),5)
        for u,v in zip(a.parameters(),a2.parameters()):self.assertTrue(torch.equal(u,v))
        for u,v in zip(b.parameters(),b2.parameters()):self.assertTrue(torch.equal(u,v))
        self.assertTrue(any(not torch.equal(u,v) for u,v in zip(a.parameters(),b.parameters())))

    def test_non_oracle_deterministic_action(self):
        model=torch.nn.Linear(m.FEATURES,1,bias=True)
        with torch.no_grad():model.weight.zero_();model.bias.fill_(1.)
        self.assertEqual(m.choice(model,np.zeros(m.FEATURES,np.float32)),m.WEAK[1])
        with torch.no_grad():model.bias.fill_(-1.)
        self.assertEqual(m.choice(model,np.zeros(m.FEATURES,np.float32)),m.STRONG[1])

    def test_summary_aggregate_synthetic(self):
        r={'graph':'g','family':'wilson','drop':0.,'reset_weak':0,'base_weak':0,'frontier_weak':0,
           'policies':{},'choices':{}}
        for p in m.ALLOWED:
            r['policies'][p]={'goal':1,'new_sites':7,'morphology':{'turn_rate':.5,'unique_ratio':.25}}
            r['choices'][p]=m.WEAK[1]
        v=m.summarize([r,dict(r,drop=.5)])['0.0']
        self.assertEqual(v['policies']['ALIGNED']['goals'],1)
        self.assertEqual(v['policies']['STRONG']['weak_selected'],1)
        self.assertEqual(v['conditions'],1)

if __name__=='__main__':unittest.main()
