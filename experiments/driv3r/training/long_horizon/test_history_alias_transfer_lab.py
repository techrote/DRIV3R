"""HISTORY-ALIAS-TRANSFER-001 host-legal/metamorphic regression tests."""
import unittest
from collections import Counter
from . import history_alias_transfer_lab as h

class TestHistoryAliasTransfer(unittest.TestCase):
    def test_rotate_inverse_and_legal_directions(self):
        for k in range(4):
            for q in ((0,1),(3,1),(-4,-1),(1,-2)):
                self.assertEqual(h.rotate(h.rotate(q,k),4-k),q)
    def test_all_new_forks_are_genuinely_replayed(self):
        seen=Counter()
        for c in h.cases():
            w=c['world'];self.assertLessEqual(w.logical_moves,64)
            self.assertEqual(12-w.recoveries,c['remaining'])
            self.assertIn(c['label'],w.menu(12))
            self.assertGreaterEqual(w.steps,25)
            seen[c['rotation'],c['remaining']]+=1
        self.assertEqual(len(seen),12)
        self.assertEqual(set(seen.values()),{2*h.CASES})
    def test_observation_is_rotated_history_alias(self):
        from itertools import islice
        for a,b in zip(islice(h.cases(),0,24,2),islice(h.cases(),1,24,2)):
            self.assertEqual((a['id'],a['rotation'],a['remaining']),
                             (b['id'],b['rotation'],b['remaining']))
            self.assertEqual(a['world'].ink,b['world'].ink)
            self.assertEqual(a['world'].menu(),b['world'].menu())
            self.assertEqual(a['world'].last_action,b['world'].last_action)
            self.assertTrue((h.blind_matrix(a['world'],a['goal'])[1]==
                             h.blind_matrix(b['world'],b['goal'])[1]).all())
    def test_unreachable_label_cannot_recover_with_budget(self):
        for c in list(h.cases())[:60]:
            if c['label']=='BACKTRACK':
                w=c['world']
                for _ in range(c['depth']):
                    self.assertIn('BACKTRACK',w.menu(12));w.apply('BACKTRACK',12)
                self.assertEqual(w.cursor,c['goal'])
            else:
                self.assertGreater(c['depth'],c['remaining'])
    def test_bounded_no_structural_heldout(self):
        self.assertEqual(h.HORIZON,12)
        self.assertEqual(h.BUDGETS,(2,3,4))
        self.assertTrue(all(c['spec'].split=='development' for c in h.cases()))

if __name__=='__main__':unittest.main()
