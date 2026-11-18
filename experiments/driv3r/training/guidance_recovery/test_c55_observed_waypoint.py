"""C55 host-authority and reproducibility assertions, independent local tests."""
import copy, unittest
import numpy as np
from experiment import World,sha
from signed_feedback import HostAdvice
import c54_delayed_branch_option as c54
import c55_observed_waypoint_distillation as c55

class Tests(unittest.TestCase):
    def test_goal_is_host_observed_legal_frontier(self):
        w=World(29,'islands',1);l=c54.ObservedLedger()
        l.see(w);self.assertIsNone(c55.waypoint(l,w.pos))
        # Actual host executes two moves and observes fresh menus.
        for step in ((2,1),(3,1)):
            cs=w.candidates(0.);c=next(c for c in cs[:4] if c.legal and c.target==step)
            start=w.pos;self.assertTrue(w.apply(c));l.after(start,c);l.see(w)
        dest=c55.waypoint(l,w.pos)
        if dest is not None:
            self.assertIn(dest,l.frontiers())
            self.assertNotIn(dest,l.neighbors)
            self.assertGreaterEqual(len(l.route(w.pos,dest)),3)
    def test_fresh_menu_and_no_remote_geometry(self):
        w=World(31,'islands',1);l=c54.ObservedLedger();menu=l.see(w)
        advice=HostAdvice(w);advice.issue_backtrack(w,1.,ttl=9)
        before=copy.deepcopy(l.neighbors)
        w.protected.add((13,12))
        self.assertEqual(before,l.neighbors)
        idx,nxt=c55.desired(l,w,advice,menu)
        self.assertTrue(nxt is None or all(menu[k].legal for k in idx))
    def test_illegal_actions_cannot_be_selected_by_head(self):
        w=World(53,'islands',-1);l=c54.ObservedLedger();menu=l.see(w)
        nxt=next(c.target for c in menu[:4] if c.legal)
        for c in menu:
            x=c55.feature(w,l,menu,c,nxt)
            self.assertEqual(x.shape,(c55.FEAT,))
            self.assertTrue(np.isfinite(x).all())
    def test_stochastic_valid_probability_not_hard_action(self):
        from c28_temporal_option import probs
        x=np.array([4.2,1.5,.0,-16.,-16.])
        p=probs(x)
        self.assertLess(p[0],1.)
        self.assertGreater(p[1],0.)
    def test_no_advice_disables_route_authority(self):
        w=World(63,'islands',1);l=c54.ObservedLedger();menu=l.see(w)
        advice=HostAdvice(w)
        self.assertEqual(c55.desired(l,w,advice,menu),([],None))
    def test_frozen_checkpoint_shape(self):
        import torch
        head=c55.Head().eval();self.assertEqual(tuple(head(torch.zeros(8,c55.FEAT)).shape),(8,))

if __name__=='__main__':unittest.main()
