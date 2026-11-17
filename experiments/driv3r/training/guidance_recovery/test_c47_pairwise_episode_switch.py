"""C47 regression: legal matched execution, group shuffle and abstention semantics."""
import hashlib, json, unittest
from pathlib import Path
import numpy as np
import torch
import c47_pairwise_episode_switch as c

class DifferentialGoalTests(unittest.TestCase):
    def test_shape_and_exact_source_fixture(self):
        root=Path(__file__).parents[1]/'predecessor_C42'
        base,option,prior=c.c44.prerequisites(root)
        w,info=c.c41.world(471731,'tree','detour',1)
        for cue,strength in c.ARMS:
            x=c.context(w,prior,cue,strength,True)
            self.assertEqual(x.shape,(19,));self.assertTrue(np.isfinite(x).all())
        # A snapshot may not modify the protected mask while computing features.
        self.assertEqual(info['graph_sha'],c.sha(sorted({(x,y) for x in range(15) for y in range(15)}-w.protected)))

    def test_goal_dominance_and_model_abstention(self):
        m=torch.zeros(19).numpy();row={'x':m.tolist()}
        class Constant(torch.nn.Module):
            def __init__(self,v):super().__init__();self.v=v
            def forward(self,x):return torch.tensor(self.v)
        nets=[Constant(.149),Constant(.1495),Constant(.151)]
        names=c.MODELS
        self.assertEqual(c.select(row,nets,np.zeros(19),np.ones(19),names[3]),'C34-RESET')
        self.assertEqual(c.select(row,nets,np.zeros(19),np.ones(19),names[4]),'C34-RESET')
        self.assertEqual(c.select(row,nets,np.zeros(19),np.ones(19),names[5]),'C40-PAIR')
        self.assertEqual(c.select(row,nets,np.zeros(19),np.ones(19),'C47-RAW-RULE'),'C34-RESET')

    def test_group_shuffling_preserves_stratum_and_arm(self):
        rows=[]
        for family in ('tree','braid'):
            for graph in range(4):
                for cue in ('.35','1.0','continue'):
                    d=(((graph*2+len(cue))%3)-1)
                    rows.append({'family':family,'case':'detour','personality':1,'switch':True,
                        'graph':f'{family}-{graph}','delta':d,'directive':cue})
        y=c.shuffled_deltas(rows,c.SEED)
        for fam in ('tree','braid'):
            a=sorted(r['delta'] for r in rows if r['family']==fam)
            b=sorted(y[i] for i,r in enumerate(rows) if r['family']==fam)
            self.assertEqual(a,b)
        self.assertEqual(len(y),len(rows))

    def test_trained_heads_use_real_optimizer(self):
        rows=[]
        for graph in range(8):
            for cue in range(3):
                x=np.zeros(19,np.float32);x[0]=graph/4;x[1]=cue
                rows.append({'x':x.tolist(),'delta':(-1,0,1)[(graph+cue)%3],
                    'family':'tree','case':'detour','personality':1,'switch':False,
                    'graph':f'tree-{graph}'})
        nets,stats,mean,scale=c.train(rows,epochs=3)
        self.assertGreater(stats['aligned']['updates'],0)
        self.assertFalse(all(torch.equal(nets[0].state_dict()[k],nets[2].state_dict()[k]) for k in nets[0].state_dict()))
        self.assertTrue((scale>=.05).all())

    def test_local_outputs_exact_unguided_parity(self):
        f=Path(__file__).with_name('C47-summary.json')
        if not f.exists():self.skipTest('full cohort results not yet generated')
        data=json.loads(f.read_text());n=data['graph_counts']['online']
        self.assertEqual(data['guided_episodes_per_policy'],n*3)
        for name,z in data['models'].items():
            self.assertEqual(z['unguided_hash_equal'],n,name)
        self.assertEqual(set(data['graph_sets_sha256']),{'train','blind','online'})

if __name__=='__main__':unittest.main()
