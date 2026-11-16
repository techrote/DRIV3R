"""P17 offline synthetic-world guardrails, no game or network access."""
import tempfile,unittest
from pathlib import Path
import numpy as np
from population_lab import NICHES,FEATURES,stable_seed
from population_p17 import (MissionWorld,SmallPolicy,ALL_FEATURES,ARMS,expert_action,
    observations,train,rollout,run,sha)

class P17Tests(unittest.TestCase):
    def test_source_derived_synthetic_host_legality_and_dynamic_debris(self):
        for seed in (672,673,674):
            w=MissionWorld(seed,holdout=True)
            while not w.stopped and w.steps<w.max_steps:
                choices=w.legal();a=expert_action(w)
                self.assertIn(a,choices)
                w.step(a)
                self.assertFalse(w.painted&(w.obstacles|w.protected))
            self.assertEqual(w.stage,3)
            self.assertEqual(w.legal(),['STOP'])
            self.assertEqual(w.disturbance_count,1)
            self.assertTrue(w.switched)

    def test_observation_goal_ablation_preserves_action_vectors_and_materials(self):
        w=MissionWorld(670)
        acts,x=observations(w,'goal_aware_curriculum')
        _,y=observations(w,'goal_blind_curriculum')
        for feature in ('goal_progress','guidance','goal_switched','relative_goal_distance','phase','goal_dx','goal_dy'):
            self.assertTrue(np.allclose(y[:,ALL_FEATURES.index(feature)],0))
        for feature in ('direction_x','direction_y','material_sand','material_metal','material_clay'):
            self.assertTrue(np.array_equal(x[:,ALL_FEATURES.index(feature)],y[:,ALL_FEATURES.index(feature)]))
        self.assertEqual(len(ALL_FEATURES),len(FEATURES)+10)

    def test_real_optimizer_changes_weights_and_exact_update_budget(self):
        p=SmallPolicy(1001,hidden=8)
        initial=p.p['W1'].copy()
        log=train(p,'recoverer',[101,102,103,104],'goal_aware_curriculum',42)
        self.assertEqual(log['optimizer_updates'],42)
        self.assertFalse(np.array_equal(initial,p.p['W1']))
        self.assertGreater(log['train_episodes'],0)

    def test_niche_and_treatments_are_preserved(self):
        self.assertEqual(len(NICHES),6)
        self.assertEqual(len(ARMS),3)
        self.assertIn('material_cautious',NICHES)
        self.assertIn('goal_blind_curriculum',ARMS)

    def test_original_closed_loop_trace_is_host_legal_and_heldout(self):
        p=SmallPolicy(1002)
        o=rollout(p,'navigator',stable_seed('heldout',9),'goal_aware_curriculum',full_trace=True)
        self.assertTrue(o['synthetic'])
        self.assertFalse(o['native'])
        self.assertEqual(o['legal_actions'],o['steps'])
        for t in o['history']:self.assertIn(t['action'],t['legal_actions'])
        self.assertFalse({tuple(x) for x in o['painted']} &
                         ({tuple(x) for x in o['protected']}|{tuple(x) for x in o['obstacles']}))

    def test_boundaries_and_local_campaign_original_manifest(self):
        with self.assertRaises(ValueError):run('/tmp/never',updates=50000)
        with self.assertRaises(ValueError):run('/tmp/never',offspring=10)
        with self.assertRaises(ValueError):run('/tmp/never',seeds=[1,1])
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'p17'
            r=run(out,seeds=(778,),train_worlds=4,selection_worlds=3,heldout_worlds=4,
                  updates=40,offspring=1,gallery_limit=1)
            self.assertEqual(r['candidates'],18)
            self.assertEqual(r['total_adam_updates'],720)
            self.assertEqual(len(list((out/'checkpoints').glob('*.json'))),18)
            self.assertTrue((out/'SHA256SUMS').exists())
            self.assertEqual(len(r['champions']),18)
            self.assertEqual(len(r['niches']),6)
            self.assertGreater(r['diversity']['goal_aware_curriculum']['compared_pairs'],0)

if __name__=='__main__':unittest.main()
