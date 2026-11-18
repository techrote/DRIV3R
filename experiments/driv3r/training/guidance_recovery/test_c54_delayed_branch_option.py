"""C54 targeted synthetic-host safety and statistical invariants."""
from __future__ import annotations
import unittest, numpy as np
from experiment import World
import c54_delayed_branch_option as c54

class Tests(unittest.TestCase):
    def test_ledger_uses_current_menu_and_reaches_known_frontier(self):
        w=World(11,'islands',1); log=c54.ObservedLedger()
        log.see(w)
        p=log.choose(w)
        self.assertIsNotNone(p)
        route=log.route(w.pos,p)
        self.assertEqual(len(route),2)
        self.assertIn(p,log.neighbors[w.pos])
    def test_observed_branch_does_not_use_remote_protected(self):
        w=World(13,'islands',1);log=c54.ObservedLedger()
        log.see(w);before=log.neighbors.copy()
        w.protected.add((13,11)) # remote geometry never queried
        self.assertEqual(log.neighbors,before)
        self.assertTrue(np.isfinite(c54.features(w,log,('backtrack',1.))).all())
    def test_bounded_option_stochastic(self):
        w=World(19,'islands',-1);log=c54.ObservedLedger();menu=log.see(w)
        from signed_feedback import HostAdvice
        advice=HostAdvice(w);advice.issue_backtrack(w,strength=1.,ttl=9)
        o=c54.BranchOption(log,True,max_ticks=2)
        bias=o.prepare(w,advice,menu)
        self.assertIsNotNone(bias)
        self.assertTrue(all(menu[i].legal for i in bias))
        self.assertIsNotNone(o.prepare(w,advice,menu))
        self.assertIsNone(o.prepare(w,advice,menu))
    def test_no_retrospective_oracle(self):
        # A visited node is not automatically granted knowledge of its
        # unexplored neighbours until see() actually observes the live menu.
        w=World(20,'islands',1);log=c54.ObservedLedger()
        self.assertIsNone(log.choose(w))
        self.assertEqual(log.neighbors,{})
    def test_feature_shape(self):
        w=World(27,'islands',1); log=c54.ObservedLedger();log.see(w)
        x=c54.features(w,log,('backtrack',.35))
        self.assertEqual(x.shape,(c54.FCOUNT,))
        self.assertTrue(np.isfinite(x).all())

if __name__=='__main__':unittest.main()
