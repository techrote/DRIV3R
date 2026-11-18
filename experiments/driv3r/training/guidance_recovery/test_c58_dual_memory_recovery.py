"""C58 independent source, legality, erasure, optimizer and replay tests."""
import copy,hashlib,json,unittest
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
import c58_dual_memory_recovery as c58
from c54_delayed_branch_option import ObservedLedger

ROOT=Path(__file__).resolve().parent.parent

class DualMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary=json.loads((ROOT/'out/C58-summary.json').read_text())
        cls.rows=json.loads((ROOT/'out/C58-online.json').read_text())
        cls.traces=json.loads((ROOT/'out/C58-traces.json').read_text())
        cls.base,cls.option,_=c58.c55.c54.c52.c50.c47.c44.prerequisites(ROOT/'prior/predecessor_C42')
        cls.heads={}
        for name in ('C58-ALIGNED','C58-SHUFFLED'):
            m=c58.c55.Head();m.load_state_dict(load_file(str(ROOT/'out'/(name+'.safetensors'))));m.eval()
            cls.heads[name]=m
    def test_source_and_graph_disjointness(self):
        self.assertEqual(hashlib.sha256((ROOT/'lab/c58_dual_memory_recovery.py').read_bytes()).hexdigest(),self.summary['source_sha256'])
        train={r['graph'] for r in self.summary['cohorts']['train']}
        online={r['graph'] for r in self.summary['cohorts']['online']}
        self.assertFalse(train&online)
        for name in ('C56','C57'):
            old=json.loads((ROOT/'prior'/(name+'-summary.json')).read_text())
            o={r['graph'] for k in ('train','online') for r in old['cohorts'][k]}
            self.assertFalse(o&online)
    def test_dual_erasure_provenance_no_fabrication(self):
        l=ObservedLedger()
        l.neighbors={(0,0):((1,0),), (1,0):((0,0),(2,0)),(2,0):((1,0),)}
        l.actions=[((0,0),(1,0),'INK'),((1,0),(2,0),'LIFT')]
        l.visits[(0,0)]=2
        orig=copy.deepcopy(l)
        zero=c58.damaged(l,'fixture',9,0.)
        half=c58.damaged(l,'fixture',9,.5)
        full=c58.damaged(l,'fixture',9,1.)
        self.assertEqual(orig.neighbors,l.neighbors)
        self.assertEqual(orig.actions,l.actions)
        self.assertEqual(zero.actions,orig.actions)
        self.assertEqual(half.neighbors,c58.damaged(l,'fixture',9,.5).neighbors)
        self.assertEqual(half.actions,c58.damaged(l,'fixture',9,.5).actions)
        self.assertFalse(full.actions)
        self.assertTrue(set(full.neighbors).issubset({(2,0)}))
        demonstrated={(a,b) for a,b,k in orig.actions}
        edges={(a,b) for a,ns in half.neighbors.items() for b in ns}
        original_menus={(a,b) for a,ns in orig.neighbors.items() for b in ns}
        self.assertTrue(edges.issubset(demonstrated|original_menus))
    def test_legality_no_guidance_and_personality_counts(self):
        self.assertEqual(len(self.rows),self.summary['online_rows'])
        parity=0
        for row in self.rows:
            vals=row['outcomes']
            self.assertTrue(all(v['host_rejections']==0 for v in vals.values()))
            self.assertEqual(len({v['protected_sha'] for v in vals.values()}),1)
            if row['arm'][0]=='none':
                parity+=1
                self.assertEqual(len({(v['action_sha'],v['ink_sha']) for v in vals.values()}),1)
            for v in vals.values():
                counts=v['counts']
                self.assertLessEqual(counts.get('personality_preferred',0)+counts.get('personality_opposed',0),v['steps'])
        self.assertEqual(parity,self.summary['unguided_parity'])
    def test_real_adamw_refit(self):
        rows=json.loads((ROOT/'out/C58-training.json').read_text())
        trained,fit=c58.c55.train(rows,epochs=8)
        self.assertEqual(fit['X_sha256'],self.summary['training']['X_sha256'])
        self.assertEqual(fit['optimizer_updates'],self.summary['training']['optimizer_updates'])
        for k,orig in [('C58-ALIGNED','C55-ALIGNED'),('C58-SHUFFLED','C55-SHUFFLED')]:
            ref=load_file(str(ROOT/'out'/(k+'.safetensors')))
            self.assertTrue(all(torch.equal(t,trained[orig].state_dict()[n]) for n,t in ref.items()))
    def test_independent_gallery_replay(self):
        self.assertGreaterEqual(len(self.traces),9)
        bygraph={x['graph']:x for x in self.summary['cohorts']['online']}
        for item in self.traces[:9]:
            meta=bygraph[item['graph']]
            fn=(c58.c55.c54.c52.c50.c47.c42.world if item['family'] in c58.c55.c54.c52.ONLINE_FAMILIES else c58.c55.c54.c52.c50.c47.c41.world)
            w,info=fn(item['seed'],item['family'],item['case'],meta['personality'])
            self.assertEqual(info['graph_sha'],item['graph'])
            got=c58.c55.c54.prefix(self.base,w,item['seed'],item['tick'])
            self.assertIsNotNone(got)
            snap,rng,ledger,receipt=got
            self.assertEqual(json.loads(json.dumps(receipt)),item['prefix'])
            z=c58.execute(self.base,self.option,snap,rng,ledger,item['tick'],item['target'],('backtrack',1.),item['policy'],self.heads,item['graph'],item['drop'],True)
            self.assertEqual(json.loads(json.dumps(z)),item['outcome'])

if __name__=='__main__':unittest.main()
