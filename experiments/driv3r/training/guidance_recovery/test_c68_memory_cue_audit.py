import unittest
from pathlib import Path
import c68_memory_cue_audit as m

class C68AuditTests(unittest.TestCase):
    def setUp(self):
        self.root=Path(__file__).parent
    def test_schema_and_cohort_identity(self):
        a=m.audit(self.root)
        self.assertEqual(a['unique_graphs'],72)
        self.assertEqual([a['cohorts'][c]['graphs'] for c in ('C65','C66','C67')],[12,24,36])
    def test_memory_intervention_not_overclaimed(self):
        a=m.audit(self.root)
        self.assertIn('does not prove injection was skipped',a['interpretation_boundary'][0])
        self.assertEqual(a['cohorts']['C66']['memory_corruption_hash_parity']['C60-FROZEN@0.35']['same_action_and_ink_hashes'],23)
    def test_cue_transfer_not_monotonic(self):
        a=m.audit(self.root)
        self.assertEqual([a['cohorts'][c]['cue_contrast']['0.0']['paired_goal_delta'] for c in ('C65','C66','C67')],[2,-1,4])
    def test_repeat(self):
        self.assertEqual(m.audit(self.root),m.audit(self.root))
    def test_duplicate_rows_rejected(self):
        rows=[{'graph':'x','drop':0.,'arm':['backtrack',.35]},{'graph':'x','drop':0.,'arm':['backtrack',.35]}]
        with self.assertRaises(AssertionError):m.records_for('C66',{'records':rows})
    def test_graph_cluster_interval_includes_zero_for_c66(self):
        x=m.audit(self.root)['cohorts']['C66']['cue_contrast']['0.0']['graph_cluster_bootstrap95_descriptive']
        self.assertTrue(x[0]<=0<=x[1])

if __name__=='__main__':unittest.main()
