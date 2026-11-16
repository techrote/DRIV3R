"""GAL-016 external host mutation, material integrity and finite-state tests."""
import unittest
from frontier_memory import Scorer
from lab import START, HostRefusal, sha
from route_scaffold import PaintScorer
from gal016_volatile_memory import BoundedLedger, PulseWorld, episode, audit_trace, EVAL_BASE

class VolatileMemoryTests(unittest.TestCase):
    def test_capacity_limited_to_recent_material(self):
        w=PulseWorld(EVAL_BASE,'tree','lift','gal016:test:cap',64,'static')
        led=BoundedLedger(41)
        led.observe(w)
        self.assertLessEqual(len(led.known),41)
        self.assertFalse(set(w.corridor).issubset(led.known))

    def test_observation_invalidates_stale_recorded_material(self):
        w=PulseWorld(EVAL_BASE,'tree','lift','gal016:test:stale',64,'static')
        led=BoundedLedger(81);led.observe(w)
        q=next(p for p in led.known if led.known[p]==0 and p!=START)
        led.recorded_paints.add(q)
        led.observe(w)
        self.assertNotIn(q,led.recorded_paints)

    def test_host_lease_binds_transient_material(self):
        w=PulseWorld(EVAL_BASE,'tree','lift','gal016:test:lease',64,'pulse')
        options,stale=w.offer()
        q=next(p for p in w.goal if w.material[w.idx(p)]==0 and p!=w.cursor)
        w._change(q,3,'test obstruction')
        self.assertNotEqual(w.offer()[1],stale)
        self.assertNotIn(q,w.protected)
        with self.assertRaises(HostRefusal):
            w.apply('STOP',stale,original_response='stale')
        w._change(q,0,'test release')
        self.assertEqual(w.material[w.idx(q)],0)

    def test_no_protected_mutation_and_exact_host_replay(self):
        w,led,row=episode(EVAL_BASE,'deep_tree',64,None,'pulse',Scorer(),PaintScorer(),trace=True)
        self.assertTrue(audit_trace(w,row))
        self.assertFalse(w.own&w.protected)
        self.assertEqual(row['receipts_sha256'],sha(w.trace))
        self.assertGreaterEqual(row['events'],1)

    def test_eval_determinism_and_capacity(self):
        for cap in (41,81,None):
            _,_,a=episode(EVAL_BASE,'comb',48,cap,'pulse',Scorer(),PaintScorer())
            _,_,b=episode(EVAL_BASE,'comb',48,cap,'pulse',Scorer(),PaintScorer())
            self.assertEqual(a,b)
            if cap:self.assertLessEqual(a['known_cells'],cap)

if __name__=='__main__':unittest.main()
