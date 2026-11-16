"""Source-local CPU guardrails; no native code and no network access."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from population_lab import NICHES, Policy
from population_p18_tournament import ARMS, TRAIN_CARDS, HOLDOUT_CARDS, scenario, trajectory, run, lineage_ranking


class P18Tests(unittest.TestCase):
    def test_traps_materials_and_legal_menu(self):
        for c in range(6):
            w=scenario(123+c,c,48)
            self.assertFalse(w.painted & (w.obstacles|w.protected))
            self.assertNotIn(w.cursor,w.obstacles|w.protected)
            self.assertIn('STOP',w.legal())
            self.assertTrue(any(x in w.legal() for x in ('N','E','S','W')))

    def test_train_heldout_scenarios_are_disjoint(self):
        self.assertFalse(set(TRAIN_CARDS)&set(HOLDOUT_CARDS))
        self.assertEqual(len(NICHES),6)
        self.assertEqual(len(ARMS),4)

    def test_long_game_has_goal_switch_and_paint_tree(self):
        out,w=trajectory('persistent',NICHES[0],321,0,48)
        self.assertEqual(out['illegal_paint'],0)
        self.assertEqual(out['legal_rate'],1)
        self.assertTrue(all(v in w.painted for v in w.parent))
        self.assertTrue(all(v in w.painted for v in w.parent.values()))
        self.assertTrue(out['steps']<=48)

    def test_forced_host_legal_retrace_crosses_goal_change(self):
        w=scenario(744,5,48)
        for _ in range(24):
            legal=w.legal()
            action=next((a for a in legal if a in ('E','S','N','W')),None)
            if action is None:
                action='RETRACE' if 'RETRACE' in legal else 'STOP'
            w.step(action)
        self.assertEqual(w.steps,24)
        self.assertTrue(w.switched)
        self.assertEqual(w.goal,w.second_goal)
        self.assertFalse(w.painted & (w.obstacles | w.protected))

    def test_all_niches_update_actual_weights(self):
        for niche in NICHES:
            p=Policy(789)
            from population_p18_tournament import teach_offspring
            initial=p.serialized()['parameters']['W1']
            info=teach_offspring(p,niche,[31,32],24,'reward_pbt')
            self.assertGreater(info['updates'],0)
            self.assertFalse(np.allclose(initial,p.serialized()['parameters']['W1']))

    def test_bounded_run_retains_candidates_and_original_traces(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'p18_tournament'
            r=run(path,cohorts=(719,),initial=2,generations=1,offspring=1,
                  training_worlds=2,selection_worlds=2,heldout_worlds=2,
                  horizon=24,arms=('reward_pbt','novelty_pbt'))
            self.assertEqual(r['candidates'],2*6*3)
            self.assertEqual(len(list((path/'checkpoints').glob('*.json'))),r['candidates'])
            self.assertEqual(len(list((path/'gallery').glob('*.svg'))),2*6)
            self.assertTrue((path/'SHA256SUMS').exists())
            self.assertTrue(all(x['heldout']['legal_rate']==1 for x in r['selection']))
            self.assertEqual(sum(r['split_manifest'][0]['splits'][k]['seeds']!=[] for k in ('train','selection','holdout')),3)

    def test_bounds_reject_invalid_config(self):
        with self.assertRaises(ValueError):run('/tmp/p18-should-not-exist',cohorts=(1,1))
        with self.assertRaises(ValueError):run('/tmp/p18-should-not-exist',horizon=120)
        with self.assertRaises(ValueError):scenario(1,7,48)

if __name__=='__main__': unittest.main()
