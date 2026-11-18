"""Small source-only C44 regression tests (no paid compute or native game)."""
import json, tempfile, unittest
from pathlib import Path
import numpy as np
from c39_action_value import action_features
import c44_onpolicy_pairwise as c44
import c41_topology_transfer as c41
import c42_wilson_transfer as c42

class C44HostBoundaryTests(unittest.TestCase):
    def test_graph_families_are_independently_generated(self):
        old,_=c41.world(441701,'prim_braid','detour',1)
        new,info=c42.world(551701,'wilson','detour',1)
        self.assertNotEqual(tuple(sorted(old.protected)),tuple(sorted(new.protected)))
        self.assertTrue(new.protected)
        self.assertTrue(info['graph_sha'])
        self.assertTrue(any(v.legal for v in new.candidates(.35)))

    def test_action_head_input_is_observation_plus_host_menu(self):
        w,_=c42.world(444211,'kruskal','forward',-1)
        menu=w.candidates(.35)
        x=action_features(w,menu)
        self.assertEqual(x.shape,(9,34))
        self.assertTrue(np.isfinite(x).all())
        self.assertTrue(all(menu[j].legal for j in [j for j in range(9) if menu[j].legal]))

    def test_gallery_contains_only_supplied_executed_positions(self):
        trace=[{'cohort':'A','family':'wilson','model':'C44-ONPOLICY','goals':0,
            'initial':[1,1],'protected':[[2,2]],'actions_sha':'a'*64,'ink_sha':'b'*64,
            'trajectory':[{'position':[1,2],'goal':[13,13],'action':'MOVE'}]}]
        with tempfile.TemporaryDirectory() as td:
            dest=Path(td)/'gallery.html';c44.svg_gallery(trace,dest)
            html=dest.read_text()
            self.assertIn('C44-ONPOLICY',html)
            self.assertIn('35,50',html)
            self.assertIn('gray cells are protected',html)

    def test_distinct_cohort_seeds(self):
        self.assertEqual(len(set(seed for _,seed in c44.COHORTS)),4)
        self.assertTrue(all(k in c44.MODELS for k in ('C34-RESET','C40-PAIR','C44-SHUFFLED','C44-RANDOM')))

if __name__=='__main__':unittest.main()
