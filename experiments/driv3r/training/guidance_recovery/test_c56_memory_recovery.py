"""C56: source-matched memory erasure, legality, refit and replay checks."""
from __future__ import annotations
import collections, copy, json, unittest
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
import c56_memory_recovery as c56
import c55_observed_waypoint_distillation as c55

HERE=Path(__file__).resolve().parent
EVIDENCE=HERE.parent/'output'
PRIOR=HERE.parent/'prior'

class C56Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=json.loads((EVIDENCE/'C56-online.json').read_text())
        cls.summary=json.loads((EVIDENCE/'C56-summary.json').read_text())
        cls.traces=json.loads((EVIDENCE/'C56-traces.json').read_text())
        cls.base,cls.option,_=c55.c54.c52.c50.c47.c44.prerequisites(PRIOR/'predecessor_C42')
        cls.heads={}
        for name in ('C55-ALIGNED','C55-SHUFFLED','C56-AUGMENTED','C56-SHUFFLED'):
            src=PRIOR/'checkpoints'/(name+'.safetensors') if name.startswith('C55') else EVIDENCE/(name+'.safetensors')
            head=c55.Head();head.load_state_dict(load_file(str(src)));head.eval();cls.heads[name]=head
    def test_provenance_and_disjoint_graphs(self):
        import hashlib
        self.assertEqual(hashlib.sha256((HERE/'c56_memory_recovery.py').read_bytes()).hexdigest(),self.summary['source_sha256'])
        tr={x['graph'] for x in self.summary['cohorts']['train']}
        on={x['graph'] for x in self.summary['cohorts']['online']}
        self.assertFalse(tr & on)
        self.assertEqual(len(on),self.summary['online_graphs'])
        prev=set()
        for fn in ('C55-training-records.json','C55-blind-records.json','C55-online.json','C55-external-online.json'):
            prev.update(x['graph'] for x in json.loads((PRIOR/'previous_C55'/fn).read_text()))
        self.assertFalse(on & prev)
    def test_mask_is_monotone_deterministic_and_no_added_edges(self):
        from c54_delayed_branch_option import ObservedLedger
        x=ObservedLedger()
        x.neighbors={(0,0):((1,0),), (1,0):((0,0),(2,0)), (2,0):((1,0),)}
        x.actions=[((1,0),(2,0),'MOVE')]
        a=c56.erase(x,'testgraph',9,.5)
        b=c56.erase(x,'testgraph',9,.5)
        self.assertEqual(a.neighbors,b.neighbors)
        self.assertTrue(set(a.neighbors).issubset(x.neighbors))
        self.assertEqual(a.neighbors[(2,0)],x.neighbors[(2,0)])
        self.assertEqual(x.neighbors,c56.erase(x,'testgraph',9,0.).neighbors)
    def test_independent_unguided_legality(self):
        self.assertEqual(len(self.rows),self.summary['evaluation_rows'])
        for row in self.rows:
            self.assertTrue(all(v['host_rejections']==0 for v in row['outcomes'].values()))
            self.assertEqual(1,len({v['protected_sha'] for v in row['outcomes'].values()}))
            if row['arm'][0]=='none':
                hashes={(v['action_sha'],v['ink_sha']) for v in row['outcomes'].values()}
                self.assertEqual(len(hashes),1)
    def test_remote_not_relabelled_observed(self):
        self.assertTrue(any(r['goal_type']=='remote' for r in self.rows))
        self.assertTrue(any(r['goal_type']=='observed' for r in self.rows))
        for r in self.rows:
            if r['goal_type']=='remote':
                self.assertEqual(r['route_retained'],0)
    def test_deterministic_checkpoint_refit(self):
        import hashlib
        torch.set_num_threads(1)
        train=json.loads((EVIDENCE/'C56-training.json').read_text())
        heads,report=c55.train(train,epochs=9)
        for new,old in (('C56-AUGMENTED','C55-ALIGNED'),('C56-SHUFFLED','C55-SHUFFLED')):
            prior=load_file(str(EVIDENCE/(new+'.safetensors')))
            for k,t in prior.items():
                self.assertTrue(torch.equal(t,heads[old].state_dict()[k]),(new,k))
        self.assertEqual(report['X_sha256'],self.summary['training']['X_sha256'])
    def test_reexecute_executed_trajectory_gallery(self):
        from experiment import sha
        lookup={r['graph']:r for r in self.summary['cohorts']['online']}
        self.assertEqual(len(self.traces),24)
        for item in self.traces:
            meta=lookup[item['graph']]
            fn=(c55.c54.c52.c50.c47.c42.world if item['family'] in c55.c54.c52.ONLINE_FAMILIES
                else c55.c54.c52.c50.c47.c41.world)
            w,info=fn(item['seed'],item['family'],item['case'],meta['personality'])
            self.assertEqual(info['graph_sha'],item['graph'])
            got=c55.c54.prefix(self.base,w,item['seed'],item['tick'])
            self.assertIsNotNone(got)
            snap,rng,ledger,receipt=got
            self.assertEqual(receipt['positions'],[tuple(p) for p in item['positions']])
            result=c56.rollout(self.base,self.option,snap,rng,ledger,item['tick'],
                item['target'],('backtrack',1.),item['policy'],self.heads,item['graph'],item['drop'],True)
            self.assertEqual(result,item['outcome'])

if __name__=='__main__':unittest.main()
