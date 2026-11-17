"""Focused opt-in smoke tests for POST-SWITCH-REGRET-001 (no training)."""
import copy
import unittest
from . import post_switch_regret_lab as m, goal_switch_curriculum_lab as prior
from .world import World, Spec

class PostSwitchRegretTests(unittest.TestCase):
    def test_development_only_and_no_oracle_heldout(self):
        self.assertEqual({x.split for x in prior.dev_specs()}, {'development'})
        sp=Spec('heldout','archipelago',9000,'wall')
        from . import goal_recovery_lab as base
        model=base.Ranker(1)
        with self.assertRaises(ValueError):m.prefix_states(sp,619,0,model)

    def test_counterfactual_menu_and_replay_determinism(self):
        from . import goal_recovery_lab as base
        sp=prior.dev_specs()[0];w=World.create(sp);goal=prior.goals_for(w,619)[0]
        model=base.Ranker(7)
        menu=w.menu(12)
        self.assertEqual(m.vote(model,w,goal)['legal'],menu)
        before=w.state_sha()
        for a in menu:
            x=m.suffix(copy.deepcopy(w),a,goal,model)
            y=m.suffix(copy.deepcopy(w),a,goal,model)
            self.assertEqual(x,y)
            self.assertEqual(x['first'],a)
            self.assertLessEqual(len(x['rows']),m.LOOKAHEAD)
            self.assertTrue(all(z['executed'] in z['menu'] for z in x['rows']))
            for z in x['rows']:
                self.assertAlmostEqual(sum(z['probs']),1.,places=7)
                self.assertEqual(z['receipt']['native_receipt'],None)
        self.assertEqual(w.state_sha(),before)

    def test_predeclared_oracle_rule_prefers_arrival(self):
        v={'MOVE_E':{'hit_step':None,'final_distance':0,'added_recoveries':0,'stopped':False},
           'BACKTRACK':{'hit_step':9,'final_distance':3,'added_recoveries':1,'stopped':False},
           'RELOCATE_START':{'hit_step':3,'final_distance':3,'added_recoveries':1,'stopped':False}}
        self.assertEqual(m.best_action(v),['RELOCATE_START'])
        self.assertEqual(m.best_action({k:v[k] for k in ('MOVE_E','BACKTRACK')}),['BACKTRACK'])

    def test_regret_nonnegative_and_cluster_interval(self):
        fake=[]
        for i in range(2):
            fake.append({'seed':619,'spec':{'family':'pickets','challenge':'normal'},'variant':i,
                         'switch_step':32,'goal_already_ink':False,
                         'votes':{'frozen':{'chosen':'MOVE_E'},'sham':{'chosen':'MOVE_E'},'switch':{'chosen':'MOVE_E'}},
                         'best':{a:['MOVE_E'] for a in m.ARMS},
                         'suffix':{a:{'MOVE_E':{'final_distance':5,'hit_step':None,'added_recoveries':0,'stopped':False}}
                                   for a in m.ARMS}})
        a=m.cluster_interval(fake,lambda r:2.)
        self.assertEqual(a,[2.,2.])
        self.assertEqual(m.aggregate(fake)['arms']['frozen']['mean_cross_suffix_distance_regret'],0.)

    def test_synthetic_relocation_contract_caveat(self):
        sp=prior.dev_specs()[0];w=World.create(sp)
        for _ in range(8):
            moves=[x for x in w.menu(12) if x.startswith('MOVE_')]
            if not moves:break
            w.apply(moves[0],12)
        if len(w.path)>5:
            self.assertIn('RELOCATE_START',w.menu(12))
            before=len(w.path)-1
            after=w.apply('RELOCATE_START',12)
            self.assertEqual(after['logical_moves'],2*before)
            self.assertEqual(w.cursor,(48,48))
            self.assertEqual(w.recoveries,1)

if __name__=='__main__':unittest.main()
