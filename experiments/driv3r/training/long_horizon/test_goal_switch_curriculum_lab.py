"""Focused tests for bounded on-policy retarget curriculum (synthetic only)."""
import copy, json, tempfile, unittest
from pathlib import Path
from . import goal_switch_curriculum_lab as lab, goal_recovery_lab as base
from .world import World

class TestSwitchCurriculum(unittest.TestCase):
    def test_split_exclusion(self):
        a=lab.train_specs();b=lab.dev_specs()
        self.assertTrue(all(x.split=='train' and x.family not in ('archipelago','staggered_vault') for x in a))
        self.assertTrue(all(x.split=='development' and x.seed%100==2 for x in b))
        self.assertFalse({x.ident for x in a}&{x.ident for x in b})
    def test_target_schedule_fixed_and_phase_changes(self):
        w=World.create(lab.dev_specs()[0]);a=lab.goals_for(w,619)
        self.assertEqual(a,lab.goals_for(World.create(w.spec),619))
        self.assertEqual(len(set(a)),4)
        self.assertEqual([lab.phase(i,128,'switch') for i in (0,32,67,103)],[0,1,2,3])
        self.assertTrue(all(lab.phase(i,128,'sham')==0 for i in (0,32,67,103)))
    def test_student_generated_teacher_labels(self):
        m=base.Ranker(619)
        x,mask,y,meta=lab.collect(m,619,36,'switch')
        self.assertEqual((len(x),len(mask),len(y)),(meta['samples'],)*3)
        self.assertTrue(mask[range(len(y)),y].all())
        self.assertTrue(all(v in range(4) for v in map(int,meta['phase_samples'])))
        self.assertLessEqual(meta['samples'],len(lab.train_specs())*2*36)
    def test_replay_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'campaign'
            # A short real train and full horizon trace; no sealed heldout.
            models,_=lab.train(619,1,1)
            sp=lab.dev_specs()[0]
            e=lab.episode(sp,619,models['switch'],'switch',128)
            self.assertEqual(lab.replay(e,models),e['decisions'])
            self.assertTrue(all(r['event']['chosen'] in r['menu'] for r in e['rows']))
            corrupted=copy.deepcopy(e);corrupted['rows'][0]['p'][0]+=0.25
            with self.assertRaises(ValueError):lab.replay(corrupted,models)
            cp=out.with_suffix('.npz');models['switch'].save(cp)
            self.assertEqual(base.Ranker.load(cp).updates,models['switch'].updates)
    def test_observed_goal_is_only_instruction(self):
        w=World.create(lab.dev_specs()[0]);targets=lab.goals_for(w,619)
        menu0,x0=base.matrix(w,targets[0],128,False)
        menu1,x1=base.matrix(w,targets[1],128,False)
        self.assertEqual(menu0,menu1)
        self.assertFalse((x0==x1).all())
        self.assertEqual(w.steps,0)

if __name__=='__main__':unittest.main()
