"""Offline controls and safety gates for the real Pythia-70M adapter runner.

These tests DO NOT claim that pretrained weights were loaded or trained.
"""
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0,str(Path(__file__).resolve().parent))
import gal017_pythia_adapter as x
from gal016_volatile_memory import PulseWorld,BoundedLedger
from lab import HostRefusal

class TestAdapterContract(unittest.TestCase):
    def test_sources_and_seed_custody(self):
        self.assertEqual(len(set(x.TRAIN_SEEDS+x.DEV_SEEDS+x.FINAL_SEEDS)),
                         len(x.TRAIN_SEEDS+x.DEV_SEEDS+x.FINAL_SEEDS))
        self.assertEqual(len(x.MODEL_REVISION),40)
        self.assertEqual(x.MODEL_ID,'EleutherAI/pythia-70m-deduped')

    def test_menu_is_offered_and_cell_has_only_one_step(self):
        w,led=x.make_world(3001001,'tree',48,81,'static','test')
        led.observe(w)
        orig,lease=w.offer()
        a,t,p=x.decision_state(w,led,'cell')
        b,u,q=x.decision_state(w,led,'swept')
        self.assertEqual(t,lease);self.assertEqual(u,lease)
        self.assertTrue(set(a)<=set(b)<=set(orig))
        self.assertTrue(all(len(v)<=1 for v in a.values()))
        self.assertLessEqual(len(b),len(x.ALPHABET))
        self.assertIn('local_9x9=',p)
        self.assertIn('legal:',q)
        with self.assertRaises(ValueError):x.action_menu(orig,'secret_full_map')

    def test_menu_and_prompt_cannot_add_illegal_action(self):
        w,led=x.make_world(3001002,'path',48,81,'static','test')
        led.observe(w)
        menu,token,_=x.decision_state(w,led,'swept')
        self.assertNotIn('TELEPORT',menu)
        with self.assertRaises(HostRefusal):w.apply('TELEPORT',token)
        with self.assertRaises(HostRefusal):w.apply('STOP','stale-lease')

    def test_hidden_distant_material_does_not_change_model_prompt(self):
        w,led=x.make_world(3001001,'tree',48,81,'static','test')
        led.observe(w)
        a,_,p1=x.decision_state(w,led,'swept')
        far=(40,40)
        self.assertGreater(abs(far[0]-w.cursor[0])+abs(far[1]-w.cursor[1]),4)
        w.material[w.idx(far)] = 0 if w.material[w.idx(far)]==3 else 3
        b,_,p2=x.decision_state(w,led,'swept')
        self.assertEqual(a,b)
        self.assertEqual(p1,p2)

    def test_transient_obstructions_remain_host_owned(self):
        w,led=x.make_world(3001001,'tree',48,81,'pulse','test')
        for _ in range(8):
            led.observe(w)
            menu,token,_=x.decision_state(w,led,'cell')
            choice=x.oracle_choice(w,led,menu)
            before={pos:w.material[w.idx(pos)] for pos in w.protected}
            receipt=w.apply(choice,token,original_response='GAL017:test')
            led.accept(choice,menu[choice],receipt)
            self.assertEqual(before,{pos:w.material[w.idx(pos)] for pos in w.protected})
            if w.stopped:break
        self.assertTrue(all(t['action_id'] for t in w.trace))
        self.assertTrue(all(q not in w.protected for q in w.own))

    def test_teacher_heldout_separates_and_consumes_only_legal_menu(self):
        rows,results=x.collect(x.DEV_SEEDS,'cell',max_decisions=12)
        self.assertEqual(len(results),2*len(x.KINDS)*len(x.DYNAMICS))
        self.assertTrue(rows)
        self.assertTrue(all(r['menu'][r['label']] in r['prompt'] for r in rows))
        self.assertTrue(set(r['seed'] for r in rows).isdisjoint(x.TRAIN_SEEDS))

    def test_fail_closed_without_pretrained_runtime(self):
        # This local runtime has no Transformers/PEFT/weights. Do not claim SLM training.
        try:
            import peft, transformers
        except ImportError:
            with self.assertRaisesRegex(RuntimeError,'BLOCKED: pinned pretrained'):
                x.load_slm()
        else:
            self.skipTest('Pretrained deps available; actual model requires pinned HF access')

if __name__=='__main__':unittest.main()
