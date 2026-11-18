"""C49 independent source/evidence qualification against a synthetic host."""
import collections, hashlib, json, unittest
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
import c49_midtrajectory_correction as c

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'output'

class C49Qualification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary=json.loads((OUT/'C49-summary.json').read_text())
        cls.train=json.loads((OUT/'C49-training.json').read_text())
        cls.online=json.loads((OUT/'C49-online.json').read_text())
        cls.blind=json.loads((OUT/'C49-blind.json').read_text())
        cls.paths=json.loads((OUT/'C49-curated-paths.json').read_text())

    def test_cue_gating_and_abstention(self):
        head=c.Head();mean=np.zeros(19,np.float32);scale=np.ones(19,np.float32)
        row={'x':[0.]*19,'graph':'g','prefix_t':19,'directive':'continue','strength':1.}
        for model in c.MODELS[3:]:
            self.assertEqual(c.select(row,[head,head],mean,scale,model),'C18-A0')
        row['directive']='none'
        for model in c.MODELS[3:]:
            self.assertEqual(c.select(row,[head,head],mean,scale,model),'C18-A0')

    def test_data_split_and_nontrivial_counterfactual_labels(self):
        tr={r['graph'] for r in self.train};bl={r['graph'] for r in self.blind};on={r['graph'] for r in self.online}
        self.assertTrue(tr.isdisjoint(bl) and tr.isdisjoint(on) and bl.isdisjoint(on))
        self.assertGreaterEqual(len(tr),30)
        self.assertGreater(sum(r['delta']!=0 for r in self.train),15)
        self.assertEqual(sum(r['delta']!=0 for r in self.train),int(self.summary['training']['deltas'].get('-1',0))+int(self.summary['training']['deltas'].get('1',0)))
        self.assertGreater(sum(r['prefix_steps']>=9 for r in self.online),0)
        self.assertTrue(set(r['directive'] for r in self.online)=={'none','backtrack','continue'})

    def test_training_checkpoint_hashes_and_source(self):
        self.assertEqual(self.summary['source_sha256'],hashlib.sha256(Path(c.__file__).read_bytes()).hexdigest())
        ledger=json.loads((OUT/'C49-training-ledger.json').read_text())
        self.assertEqual(ledger['optimizer'],self.summary['training'])
        for m in ('C49-ALIGNED','C49-SHUFFLED'):
            f=OUT/(m+'.safetensors')
            self.assertEqual(hashlib.sha256(f.read_bytes()).hexdigest(),self.summary['checkpoints_sha256'][m])
            head=c.Head();head.load_state_dict(load_file(str(f)))
            self.assertTrue(all(torch.isfinite(t).all() for t in head.state_dict().values()))
        shuffled=c.permuted_labels(self.train,c.SEED)
        truth=np.asarray([r['delta'] for r in self.train],dtype=np.float32)
        self.assertFalse(np.array_equal(truth,shuffled))
        self.assertEqual(collections.Counter(truth),collections.Counter(shuffled))
        self.assertEqual(self.summary['training']['targets_sha256'],hashlib.sha256(truth.tobytes()).hexdigest())

    def test_exact_unguided_parity_and_host_legality(self):
        assert self.online
        ref={(r['graph'],r['prefix_t'],r['directive'],r['strength']):r for r in self.online if r['model']=='C18-A0'}
        for r in self.online:
            self.assertEqual(r['host_rejections'],0)
            self.assertEqual(r['raw_illegal_top1'],0)
            self.assertEqual(r['unguided_exact'] if 'unguided_exact' in r else 1,1)
            if r['directive']=='none':
                a=ref[r['graph'],r['prefix_t'],r['directive'],r['strength']]
                self.assertEqual((r['actions_sha'],r['ink_sha']),(a['actions_sha'],a['ink_sha']))
        for m,d in self.summary['models'].items():
            self.assertEqual(d['unguided_exact'],len({(r['graph'],r['prefix_t']) for r in self.online}))

    def test_replay_curated_actions_from_fresh_generated_graphs(self):
        self.assertGreater(len(self.paths),12)
        torch.set_num_threads(1)
        base,option,prior=c.c47.c44.prerequisites(ROOT/'prior/previous/predecessor_C42')
        worlds=c.c47.world_rows(c.c47.c42.FAMILIES,c.c47.c42.world,497117,1)
        byid={info['graph_sha']:(w,info,seed) for w,info,seed,_,_,_ in worlds}
        ref={(r['graph'],r['prefix_t'],r['model']):r for r in self.online if r['directive']=='backtrack' and r['strength']==1.}
        replays=0
        for p in self.paths:
            w,info,seed=byid[p['graph']]
            snap,rec=c.prefix(base,option,w,seed,p['prefix_t'])
            self.assertEqual(rec['actions_sha'],p['prefix_actions_sha'])
            self.assertEqual(rec['prefix_positions'],[tuple(x) for x in p['prefix_positions']])
            dest=tuple(info['second_goal'])
            if dest in snap.history or dest==snap.pos: dest=tuple(snap.goal)
            self.assertNotIn(dest,snap.history)
            snap.goal=dest;snap.goal_change=True;snap.goals_seen+=1;snap.stuck=2
            got=c.c47.execute(base,option,prior,snap,info,seed,'backtrack',1.,False,p['model'],True)
            self.assertEqual((got['goals'],got['actions_sha'],got['ink_sha']),
                             (p['goal_events'],p['actions_sha'],p['ink_sha']))
            self.assertEqual(json.loads(json.dumps(got['trajectory'])),p['trajectory'])
            self.assertEqual((got['actions_sha'],got['ink_sha']),
                             (ref[p['graph'],p['prefix_t'],p['model']]['actions_sha'],ref[p['graph'],p['prefix_t'],p['model']]['ink_sha']))
            self.assertEqual(got['host_rejections'],0)
            replays+=1
        self.assertEqual(replays,len(self.paths))

if __name__=='__main__':unittest.main(verbosity=2)
