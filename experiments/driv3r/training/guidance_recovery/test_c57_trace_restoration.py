"""C57 source/evidence, no-leakage, held-out and executable replay tests."""
from __future__ import annotations
import collections, copy, hashlib, json, unittest
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
import c57_trace_restoration as c57
import c55_observed_waypoint_distillation as c55
from c54_delayed_branch_option import ObservedLedger

ROOT=Path(__file__).resolve().parent.parent

class C57Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out=ROOT/'out';cls.prior=ROOT/'prior'
        cls.summary=json.loads((cls.out/'C57-summary.json').read_text())
        cls.rows=json.loads((cls.out/'C57-online.json').read_text())
        cls.traces=json.loads((cls.out/'C57-traces.json').read_text())
        cls.base,cls.option,_=c55.c54.c52.c50.c47.c44.prerequisites(cls.prior/'predecessor_C42')
        cls.heads={}
        for name in ('C57-ALIGNED','C57-SHUFFLED'):
            m=c55.Head();m.load_state_dict(load_file(str(cls.out/(name+'.safetensors'))));m.eval()
            cls.heads[name]=m
    def test_source_hash_and_cohort_exclusion(self):
        self.assertEqual(hashlib.sha256((ROOT/'lab/c57_trace_restoration.py').read_bytes()).hexdigest(),self.summary['source_sha256'])
        a={r['graph'] for r in self.summary['cohorts']['train']}; b={r['graph'] for r in self.summary['cohorts']['online']}
        self.assertFalse(a&b);self.assertEqual(len(b),self.summary['online_graphs'])
        earlier=json.loads((self.prior/'C56-summary.json').read_text())
        old={x['graph'] for group in ('train','online') for x in earlier['cohorts'][group]}
        self.assertFalse(b&old)
    def test_trace_restoration_only_from_executed_directed_edges(self):
        l=ObservedLedger();l.neighbors={(0,0):((1,0),)}
        l.actions=[((0,0),(1,0),'INK'),((1,0),(2,0),'LIFT'),((2,0),(2,0),'LIFT')]
        original=copy.deepcopy(l.neighbors);r=c57.restore(l)
        self.assertEqual(l.neighbors,original)
        self.assertEqual(r.neighbors[(1,0)],((2,0),))
        self.assertNotIn((2,0),r.neighbors)
        self.assertNotIn((0,0),r.neighbors[(1,0)])
        self.assertFalse(any((2,0) in ns for pos,ns in r.neighbors.items() if pos==(2,0)))
    def test_legality_no_guidance_and_protected_invariance(self):
        self.assertEqual(len(self.rows),self.summary['online_rows'])
        no_guidance=0
        for row in self.rows:
            vals=row['outcomes']
            self.assertTrue(all(x['host_rejections']==0 for x in vals.values()))
            self.assertEqual(len({x['protected_sha'] for x in vals.values()}),1)
            if row['arm'][0]=='none':
                no_guidance+=1
                self.assertEqual(len({(x['action_sha'],x['ink_sha']) for x in vals.values()}),1)
        self.assertEqual(no_guidance,self.summary['unguided_parity'])
    def test_real_cpu_optimizer_refit(self):
        data=json.loads((self.out/'C57-train.json').read_text())
        heads,stats=c55.train(data,epochs=8)
        self.assertEqual(stats['X_sha256'],self.summary['training']['X_sha256'])
        self.assertGreater(stats['shuffled_changes'],0)
        for new,old in (('C57-ALIGNED','C55-ALIGNED'),('C57-SHUFFLED','C55-SHUFFLED')):
            ref=load_file(str(self.out/(new+'.safetensors')))
            self.assertTrue(all(torch.equal(ref[k],head) for k,head in heads[old].state_dict().items()))
    def test_evidence_replay(self):
        look={r['graph']:r for r in self.summary['cohorts']['online']}
        self.assertEqual(len(self.traces),18)
        for item in self.traces[:12]:
            meta=look[item['graph']]
            fn=(c55.c54.c52.c50.c47.c42.world if item['family'] in c55.c54.c52.ONLINE_FAMILIES
                else c55.c54.c52.c50.c47.c41.world)
            w,info=fn(item['seed'],item['family'],item['case'],meta['personality'])
            self.assertEqual(info['graph_sha'],item['graph'])
            got=c55.c54.prefix(self.base,w,item['seed'],item['tick'])
            self.assertIsNotNone(got)
            snap,rng,ledger,receipt=got
            self.assertEqual(json.loads(json.dumps(receipt)),item['prefix'])
            result=c57.rollout(self.base,self.option,snap,rng,ledger,item['tick'],
               item['target'],('backtrack',1.),item['policy'],self.heads,item['graph'],item['drop'],True)
            self.assertEqual(json.loads(json.dumps(result)),item['outcome'])

if __name__=='__main__':unittest.main()
