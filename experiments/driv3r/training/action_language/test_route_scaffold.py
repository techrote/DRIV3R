"""GAL-015 independent external-host authority and model-side regression tests."""
import json
from pathlib import Path
import tempfile
import unittest
from lab import START, HostRefusal, sha
from frontier_memory import Ledger, Scorer
from route_scaffold import (
    HorizonWorld, PaintScorer, get_paint_candidates, route_actions, load_frozen,
    episode, train, run, _features, EVAL_BASE,
)

# Unit tests do not require a research checkpoint in Git history.
FROZEN=Path(__file__).parent/'_external_checkpoint_not_committed.json'

class RouteScaffoldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base=Scorer()
        cls.adapter=PaintScorer()

    def test_accepted_actions_only(self):
        w=HorizonWorld(EVAL_BASE,'tree','lift','gal015:test:lease',32)
        opts,lease=w.offer()
        with self.assertRaises(HostRefusal):w.apply('MALICIOUS_PAINT',lease,original_response='bad')
        with self.assertRaises(HostRefusal):w.apply('STOP','stale',original_response='bad')
        self.assertEqual(w.decisions,0)

    def test_start_is_not_paint_candidate(self):
        w=HorizonWorld(EVAL_BASE,'tree','lift','gal015:test:start',32)
        l=Ledger();l.observe(w)
        for name,path in get_paint_candidates(w,l,w.legal()):
            self.assertNotIn(START,path)
            self.assertTrue(name.startswith(('INK_','JOG_')))

    def test_host_and_ledger_are_distinct(self):
        w=HorizonWorld(EVAL_BASE,'tree','lift','gal015:test:observation',32)
        l=Ledger();view=l.observe(w)
        self.assertLess(len(view),len(w.material))
        self.assertEqual(l.known,view)
        self.assertFalse(set(w.corridor).issubset(l.known.keys()))

    def test_no_lift_to_fresh_target(self):
        w=HorizonWorld(EVAL_BASE,'path','lift','gal015:test:guard',32)
        led=Ledger();led.observe(w)
        # A host-legal 1-step LIFT to a fresh air cell must never qualify as
        # completion-preserving pen-up travel (cannot ink under the cursor).
        strict=route_actions(w,led,w.legal())
        unsafe=route_actions(w,led,w.legal(),allow_endpoint=True)
        self.assertTrue(any(path[-1] in led.targets() for _,path in unsafe))
        for _,path in strict:
            self.assertNotIn(path[-1],led.targets())

    def test_tiny_train_reproducible(self):
        a,sa=train(self.base,worlds=8)
        b,sb=train(self.base,worlds=8)
        self.assertEqual(sa,sb)
        self.assertEqual(a.weights,b.weights)
        self.assertGreater(sa['updates'],0)
        self.assertTrue(any(abs(x)>1e-8 for x in a.weights))

    def test_known_branching_survives(self):
        _,guard=episode(EVAL_BASE,'deep_tree',64,'route_untrained',self.base,self.adapter)
        _,unsafe=episode(EVAL_BASE,'deep_tree',64,'route_unguarded',self.base,self.adapter)
        self.assertTrue(guard['complete'])
        self.assertGreaterEqual(guard['correct'],unsafe['correct'])

    def test_deterministic_episode(self):
        _,a=episode(EVAL_BASE,'comb',48,'route_trained',self.base,self.adapter)
        _,b=episode(EVAL_BASE,'comb',48,'route_trained',self.base,self.adapter)
        self.assertEqual(a,b)

    def test_independent_host_receipt_replay(self):
        seed=EVAL_BASE;kind='deep_tree';mode='route_untrained';target=64
        world,row=episode(seed,kind,target,mode,self.base,self.adapter,save_trace=True)
        copy=HorizonWorld(seed,kind,'lift',f'gal015:{mode}:{target}:{kind}:{seed}',target)
        for receipt in world.trace:
            actions,token=copy.offer()
            self.assertIn(receipt['action_id'],actions)
            self.assertEqual(token,receipt['offered_sha256'])
            self.assertEqual(actions[receipt['action_id']],receipt['path'])
            r=copy.apply(receipt['action_id'],token,original_response=receipt['original_response'])
            self.assertEqual(r,receipt)
        self.assertEqual(sha(copy.trace),row['receipts_sha256'])
        self.assertEqual(sha(copy.material),row['material_sha256'])
        self.assertFalse(copy.own&copy.protected)

    def test_bounded_evaluation_and_training(self):
        with self.assertRaises(ValueError):train(self.base,worlds=1)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):run(tmp,FROZEN,eval_per_panel=1000)

    def test_trained_checkpoint_feature_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'bad.json';p.write_text(json.dumps({'schema':'wrong','features':[],'mode':'trained'}))
            with self.assertRaises(ValueError):load_frozen(p)
        with self.assertRaises(ValueError):PaintScorer([1.])

if __name__=='__main__':unittest.main()
