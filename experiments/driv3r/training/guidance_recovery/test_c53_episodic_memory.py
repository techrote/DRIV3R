"""Source-exact deterministic, host-legal and replay qualification for C53."""
from pathlib import Path
import collections, copy, hashlib, json, tempfile, unittest, inspect
import numpy as np
from safetensors.torch import save_file, load_file
import c53_episodic_memory as c53
import c52_shortest_step_cue as c52

ROOT=Path(__file__).resolve().parent
SUMMARY=json.loads((ROOT/'output/C53-summary.json').read_text())

class C53Tests(unittest.TestCase):
    def test_frozen_source_and_checkpoints(self):
        receipts={'c52_shortest_step_cue.py':'20b367bea58e2148da8903275e2614f1fa8eaf53bb7189a365d76546f9921129',
            'C52-ALIGNED.safetensors':'7a717ca28afa025dbb069dcfe8846e4577da3f3b91538ac96614d8bd3ca3d076'}
        for fn,s in receipts.items():
            p=ROOT/('source_and_evidence' if fn.endswith('.py') else 'results')/fn
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),s)
        for name,digest in SUMMARY['checkpoints_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT/'output'/(name+'.safetensors')).read_bytes()).hexdigest(),digest)
        self.assertEqual(hashlib.sha256((ROOT/'c53_episodic_memory.py').read_bytes()).hexdigest(),SUMMARY['source_sha256'])

    def test_memory_observation_and_private_teacher_boundary(self):
        from experiment import World
        from signed_feedback import HostAdvice
        w=World(931,'islands',1);w.history=[(1,1),(2,1)];w.pos=(3,1);w.goal=(13,13)
        w.events=[('INK',(1,1),(2,1),w.goal),('INK',(2,1),(3,1),w.goal)]
        a=HostAdvice(w);a.issue_backtrack(w,strength=1.,ttl=9)
        before=c53.features(w,a)
        w.protected.add((10,10))
        np.testing.assert_array_equal(before,c53.features(w,a))
        self.assertEqual(len(before),24)
        self.assertTrue(np.isfinite(before).all())
        s=inspect.getsource(c53.suffix)+inspect.getsource(c53.observed_memory)
        self.assertNotIn('distance_oracle(',s)
        self.assertNotIn('label_state(',s)

    def test_cohort_partition_and_training(self):
        self.assertEqual(SUMMARY['cohort_graphs'],{'train':32,'blind':16,'online':64})
        self.assertEqual(SUMMARY['train_records'],1118)
        self.assertEqual(SUMMARY['blind_records'],576)
        self.assertGreater(SUMMARY['train']['shuffled_label_changes'],0)
        self.assertEqual(set(SUMMARY['train']['updates'].values()),{288})
        ids=[{r['graph'] for r in SUMMARY['world_generation'][s]} for s in ('train','blind','online')]
        self.assertFalse(ids[0]&ids[1] or ids[0]&ids[2] or ids[1]&ids[2])

    def test_all_host_legal_and_unguided_personality(self):
        rows=json.loads((ROOT/'output/C53-online.json').read_text())
        self.assertEqual(len(rows),SUMMARY['online_rows'])
        idle=0
        for r in rows:
            self.assertEqual(set(r['outcomes']),set(c53.POLICIES))
            for z in r['outcomes'].values():
                self.assertEqual(z['host_rejections'],0)
                self.assertEqual(z['protected_sha'],r['protected_sha'])
            if r['arm']=='none':
                idle+=1
                self.assertEqual(len({(o['action_sha'],o['ink_sha']) for o in r['outcomes'].values()}),1)
        for m,z in SUMMARY['models'].items():
            self.assertEqual(z['zero_guidance_hash_parity'],idle)
        self.assertGreater(idle,0)

    def test_optimizer_determinism_repeated_cpu(self):
        train=json.loads((ROOT/'output/C53-training.json').read_text())
        for run in range(2):
            heads,mean,scale,meta=c53.fit(train)
            for m,head in heads.items():
                with tempfile.TemporaryDirectory() as td:
                    p=Path(td)/(m+'.safetensors');save_file(head.state_dict(),str(p))
                    self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),SUMMARY['checkpoints_sha256'][m])
            self.assertEqual(meta['features_sha256'],SUMMARY['train']['features_sha256'])

    def test_replay_selected_paths_and_painting(self):
        root=ROOT
        base,opt,frozen=c52.c50.c47.c44.prerequisites(root/'c47/previous/predecessor_C42')
        norm=json.loads((root/'output/C53-normalization.json').read_text())
        mean=np.asarray(norm['mean'],np.float32);scale=np.asarray(norm['scale'],np.float32)
        heads={name:c53.Head() for name in c53.VARIANTS}
        for m,h in heads.items():h.load_state_dict(load_file(str(root/'output'/(m+'.safetensors'))));h.eval()
        old=c52.Head();old.load_state_dict(load_file(str(root/'results/C52-ALIGNED.safetensors')));old.eval()
        old_norm=json.loads((root/'results/C52-normalization.json').read_text())
        models={**heads,'C52-PREV':old}
        means={**{m:mean for m in heads},'C52-PREV':np.asarray(old_norm['mean'],np.float32)}
        scales={**{m:scale for m in heads},'C52-PREV':np.asarray(old_norm['scale'],np.float32)}
        traces=json.loads((root/'output/C53-selected-traces.json').read_text())
        self.assertEqual(len(traces),18)
        for t in traces:
            w,info=c52.c50.c47.c42.world(t['seed'],t['family'],t['case'],t['personality'])
            self.assertEqual(info['graph_sha'],t['graph'])
            tick=len(t['prefix_positions'])-1
            got=c52.c50.prefix(base,w,t['seed'],tick)
            self.assertIsNotNone(got)
            snap,rng_state,rec=got
            self.assertEqual(list(rec['initial']),t['start'])
            z=c53.suffix(base,opt,frozen,snap,rng_state,tick,tuple(t['target']),('backtrack',1.),t['model'],models,means,scales,True)
            self.assertEqual((z['action_sha'],z['ink_sha']),
                             (t['outcome']['action_sha'],t['outcome']['ink_sha']))
            self.assertEqual(json.dumps(z['path'],sort_keys=True),json.dumps(t['outcome']['path'],sort_keys=True))

    def test_negative_findings_not_promoted(self):
        self.assertFalse(SUMMARY['promotion_eligible'])
        self.assertFalse(SUMMARY['gates']['heldout_goals_above_RESET'])
        self.assertLess(SUMMARY['models']['C53-MEM']['guided_goals'],SUMMARY['models']['C34-RESET']['guided_goals'])
        self.assertEqual(SUMMARY['paired']['C53-MEM__C53-LOCAL']['delta'],-1)

if __name__=='__main__':unittest.main()
