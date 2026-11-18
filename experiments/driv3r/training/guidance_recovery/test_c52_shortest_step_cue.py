"""C52 independence, host legality, frozen-source and optimizer regression checks."""
from pathlib import Path
import copy, hashlib, json, tempfile, unittest
import numpy as np
import torch
from safetensors.torch import save_file
import c52_shortest_step_cue as c52
import c50_continuous_recovery as c50
from experiment import World, Candidate
from signed_feedback import HostAdvice

class C52Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(__file__).resolve().parents[1]
        cls.out=cls.root/'output'
        cls.summary=json.loads((cls.out/'C52-summary.json').read_text())

    def test_matched_and_disjoint(self):
        s=self.summary
        gs=[{r['graph'] for r in s['world_generation'][k]} for k in ('train','blind','online')]
        self.assertEqual([len(v) for v in gs],[s[k+'_graphs'] for k in ('train','blind','online')])
        self.assertFalse(gs[0]&gs[1] or gs[0]&gs[2] or gs[1]&gs[2])
        self.assertEqual(s['policy_outcome_records'],s['online_rows']*len(c52.POLICIES))

    def test_training_oracle_not_in_features(self):
        w=World(931,'islands',1);w.history=[(1,1),(2,1)];w.pos=(3,1);w.goal=(13,13)
        w.stuck=2;w.goal_change=True;a=HostAdvice(w);a.issue_backtrack(w,1,9)
        before=c52.features(w,a)
        # A hidden cell far from the observed local menu must not affect inference.
        w.protected.add((10,10))
        after=c52.features(w,a)
        np.testing.assert_array_equal(before,after)
        self.assertEqual(c52.NFEATURES,len(before))
        self.assertTrue(set(np.unique([c52.label_state(w)])).issubset({0,1,None}))

    def test_unsafe_action_is_host_rejected(self):
        w=World(337,'islands',-1);w.pos=(4,4)
        w.protected.add((5,4));start=(w.pos,sorted(w.inked),sorted(w.protected))
        forged=Candidate(('INK',1,0),(5,4),True,[0.]*25)
        self.assertFalse(w.apply(forged));self.assertEqual(w.invalid_attempts,1)
        self.assertEqual((w.pos,sorted(w.inked),sorted(w.protected)),start)

    def test_full_online_host_safety_and_unguided(self):
        rows=json.loads((self.out/'C52-online.json').read_text())
        idle=0
        for row in rows:
            self.assertEqual(len(row['outcomes']),len(c52.POLICIES))
            for out in row['outcomes'].values():
                self.assertEqual(out['host_rejections'],0)
                self.assertEqual(out['protected_sha'],row['protected_sha'])
            if row['arm']=='none':
                idle+=1
                self.assertEqual(len({(z['action_sha'],z['ink_sha']) for z in row['outcomes'].values()}),1)
        self.assertGreater(idle,0)
        self.assertTrue(self.summary['promotion_gates']['exact_all_unguided_hashes'])

    def test_real_replay_and_cue_only_parity(self):
        prior=self.root/'prereq/prereq50/previous/prev47'
        base,opt,head=c50.c47.c44.prerequisites(prior/'previous/predecessor_C42')
        seen=0
        rows=json.loads((self.out/'C52-online.json').read_text())
        for row in rows:
            if row['arm'] not in ('none','continue'):continue
            w,info=c50.c47.c42.world(row['seed'],row['family'],row['case'],row['personality'])
            got=c50.prefix(base,w,row['seed'],row['cue_tick'])
            self.assertIsNotNone(got)
            snap,rngstate,receipt=got
            self.assertEqual(receipt['action_sha'],row['prefix_sha'])
            for name in ('C34-RESET','C40-PAIR'):
                z=c50.suffix(base,opt,head,snap,rngstate,row['seed'],name,
                  (row['arm'],row['strength']),row['cue_tick'],tuple(row['target']))
                self.assertEqual((z['action_sha'],z['ink_sha']),
                  (row['outcomes'][name]['action_sha'],row['outcomes'][name]['ink_sha']))
            seen+=1
            if seen>=4:break
        self.assertEqual(seen,4)

    def test_optimizer_determinism_and_receipts(self):
        rows=json.loads((self.out/'C52-training.json').read_text())
        checks=self.summary['checkpoints_sha256']
        with tempfile.TemporaryDirectory() as td:
            for run in (0,1):
                models,mean,scale,stats=c52.fit(rows)
                for name,model in models.items():
                    dest=Path(td)/(f'{name}-{run}.safetensors')
                    save_file(model.state_dict(),str(dest))
                    self.assertEqual(hashlib.sha256(dest.read_bytes()).hexdigest(),checks[name])
                self.assertEqual(stats['labels_sha256'],self.summary['training']['labels_sha256'])

    def test_blind_oracle_and_no_postselection(self):
        blind=json.loads((self.out/'C52-blind.json').read_text())
        online=json.loads((self.out/'C52-online.json').read_text())
        self.assertEqual(len(blind),self.summary['blind_size'])
        self.assertFalse({r['graph'] for r in blind}&{r['graph'] for r in online})
        self.assertTrue(all(r['y'] in (0,1) for r in blind))
        for row in blind:
            for p in row['predictions'].values():self.assertTrue(0<=p<=1)

if __name__=='__main__': unittest.main()
