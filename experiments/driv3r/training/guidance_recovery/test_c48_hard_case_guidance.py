"""C48 tests: predeclared interventions, actual-optimizer controls, data custody."""
import collections,hashlib,json,unittest
from pathlib import Path
import numpy as np
import torch
import c48_hard_case_guidance as c

ROOT=Path(__file__).resolve().parent
OUTPUT=(ROOT.parent/'output') if (ROOT.parent/'output'/'C48-summary.json').exists() else ROOT

def require_evidence(testcase,*names):
    missing=[name for name in names if not (OUTPUT/name).exists()]
    if missing:testcase.skipTest('full C48 archive required: '+', '.join(missing))

class C48Tests(unittest.TestCase):
    def test_backtrack_only_and_no_graph_oracle(self):
        row={'x':[0.]*19,'graph':'abc','directive':'continue','strength':1.}
        net=c.BinaryHead();old=(net,np.zeros(19,dtype=np.float32),np.ones(19,dtype=np.float32))
        for mode in c.MODELS[3:]:
            self.assertEqual(c.choose(row,(net,net),old[1],old[2],old,mode),'C34-RESET')
        row['directive']='none'
        for mode in c.MODELS[3:]:
            self.assertEqual(c.choose(row,(net,net),old[1],old[2],old,mode),'C34-RESET')

    def test_model_is_trained_with_nonidentity_shuffled_control(self):
        require_evidence(self,'C48-training.json','C48-training-ledger.json','C48-HARD.safetensors','C48-SHUFFLED.safetensors')
        tr=json.loads((OUTPUT/'C48-training.json').read_text())
        labels=np.asarray([r['delta'] for r in tr],np.float32)
        shifted=c.c47.shuffled_deltas(tr,c.SEED)
        self.assertFalse(np.array_equal(labels,shifted))
        self.assertEqual(collections.Counter(labels),collections.Counter(shifted))
        ledger=json.loads((OUTPUT/'C48-training-ledger.json').read_text())
        self.assertEqual(len({r['graph'] for r in tr}),128)
        self.assertEqual(ledger['optimizer']['train_rows'],len(tr))
        for mode in ('C48-HARD','C48-SHUFFLED'):
            p=OUTPUT/(mode+'.safetensors')
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),ledger['checks'][mode])
            self.assertGreater(ledger['optimizer']['hard' if mode.endswith('HARD') else 'shuffled']['adamw_updates'],0)

    def test_bounded_local_optimizer_repeat(self):
        # The miniature run must exercise actual AdamW steps and independent replicas.
        rows=[]
        for graph in range(4):
            for cue in ('backtrack-weak','backtrack-strong'):
                x=np.zeros(19,np.float32);x[0]=graph/3;x[1]=float('strong' in cue)
                rows.append({'x':x.tolist(),'graph':f'graph-{graph}', 'family':'braid',
                             'case':'detour','personality':1,'switch':False,
                             'directive':cue,'delta':(graph%3)-1})
        h,m,s,stats=c.train(rows,epochs=2)
        h2,m2,s2,stats2=c.train(rows,epochs=2)
        self.assertEqual(stats,stats2)
        self.assertTrue(np.array_equal(m,m2) and np.array_equal(s,s2))
        self.assertTrue(all(all(torch.equal(a.state_dict()[k],b.state_dict()[k])
                                for k in a.state_dict()) for a,b in zip(h,h2)))

    def test_full_cohorts_legality_parity_and_disjoint(self):
        require_evidence(self,'C48-summary.json','C48-training.json','C48-blind.json','C48-online.json')
        j=json.loads((OUTPUT/'C48-summary.json').read_text())
        self.assertEqual(j['graph_counts'],{'train':128,'blind':16,'online':96})
        tr=json.loads((OUTPUT/'C48-training.json').read_text())
        bl=json.loads((OUTPUT/'C48-blind.json').read_text())
        on=json.loads((OUTPUT/'C48-online.json').read_text())
        self.assertTrue({r['graph'] for r in tr}.isdisjoint({r['graph'] for r in bl}))
        self.assertTrue({r['graph'] for r in tr}.isdisjoint({r['graph'] for r in on}))
        self.assertTrue({r['graph'] for r in bl}.isdisjoint({r['graph'] for r in on}))
        self.assertEqual(len(on),96*4*len(c.MODELS))
        base={(r['graph'],r['directive'],r['strength']):r for r in on if r['model']=='C18-A0'}
        for r in on:
            self.assertEqual(r['host_rejections'],0)
            if r['directive']=='none':
                a=base[(r['graph'],r['directive'],r['strength'])]
                self.assertEqual((r['actions_sha'],r['ink_sha']),(a['actions_sha'],a['ink_sha']))
        self.assertEqual(j['source_sha256'],hashlib.sha256(Path(c.__file__).read_bytes()).hexdigest())

    def test_replay_retained_hashes_from_online(self):
        require_evidence(self,'C48-online.json','C48-curated-paths.json')
        online=json.loads((OUTPUT/'C48-online.json').read_text())
        selected=json.loads((OUTPUT/'C48-curated-paths.json').read_text())
        self.assertEqual(len(selected),40)
        ref={(r['graph'],r['model'],r['directive'],r['strength']):r for r in online}
        for p in selected:
            self.assertGreater(len(p['trajectory']),0)
            match=ref[(p['graph'],p['model'],'backtrack',1.)]
            self.assertEqual((p['actions_sha'],p['ink_sha']), (match['actions_sha'],match['ink_sha']))
            self.assertEqual(p['goal_events'],match['goals'])

if __name__=='__main__':unittest.main()
