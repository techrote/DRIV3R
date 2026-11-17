import importlib.util
import json
from pathlib import Path
import unittest
import tempfile
import zipfile

HERE=Path(__file__).parent

def load(name):
    spec=importlib.util.spec_from_file_location(name,HERE/(name+'.py'))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

p=load('p17_niche_evidence');c=load('p17_cross_model')

class AuditTests(unittest.TestCase):
    def test_vectors_and_trivial_reversal(self):
        self.assertEqual(p.unit((1,1),(2,1)),(1,0))
        self.assertIsNone(p.unit((1,1),(3,1)))
        g=c.geometry([[0,0],[1,0],[0,0],[1,0]])
        self.assertEqual(g['reversal2'],2)
        self.assertGreater(g['repeated_edge_rate'],.6)
    def test_curves_and_orientation(self):
        g=c.geometry([[0,0],[1,0],[1,1],[0,1],[0,0]])
        self.assertEqual(g['cycles'],1)
        self.assertEqual(g['longest_straight'],1)
        self.assertEqual(g['turn90_rate'],1)
    def test_unchanged_history(self):
        g=c.geometry([[0,0],[1,0],[2,0],[3,0]])
        self.assertEqual(g['longest_straight'],3)
        self.assertEqual(g['turn90_rate'],0)
    def test_host_legality_rejection(self):
        doc=dict(events=[dict(position_after=[1,0], observation=dict(x=0,y=0,legal=['LEFT'],tick=0), applied='RIGHT',raw_legal=False)],trajectory=[[0,0],[1,0]],size=96,steps_executed=1,step_budget=128)
        with self.assertRaisesRegex(AssertionError,'host illegal'):p.analyze(doc)
    def test_position_mismatch_rejection(self):
        doc=dict(events=[dict(position_after=[0,0],observation=dict(x=0,y=0,legal=['STRAIGHT'],tick=0),applied='STRAIGHT',raw_legal=True)],trajectory=[[0,0],[1,0]],size=96,steps_executed=1,step_budget=128)
        with self.assertRaisesRegex(AssertionError,'position receipt mismatch'):p.analyze(doc)
    def test_real_cross_model_manifest(self):
        manifest=HERE/'driv3r_wormpaths_source_manifest.json'
        if not manifest.exists():self.skipTest('source manifest not supplied')
        with tempfile.TemporaryDirectory() as temp:
            r=c.run(manifest,temp)
            self.assertEqual(len(r['entries']),14)
            self.assertEqual(len(r['matched_pairs_without_preference_labels']),10)
            self.assertTrue(all(x['owner_preference'] is None for x in r['matched_pairs_without_preference_labels']))
            self.assertTrue(all(x['model_preference'] is None for x in r['matched_pairs_without_preference_labels']))
            self.assertEqual(len(list(Path(temp).glob('*blind-A4.svg'))),4)
    def test_real_original_trajectory_inputs(self):
        inputs=[Path('/mnt/data/driv3r_preference_P10_real_W16_STOP_clock_model_inference_qualified.zip'),Path('/mnt/data/DRIV3R_P16_FRESH_SEEDS_SOURCE_MATCHED_LOCAL_V2.zip')]
        if not all(x.exists() for x in inputs):self.skipTest('exact two source zips not provided')
        with tempfile.TemporaryDirectory() as temp:
            r=p.run(inputs,temp)
            self.assertEqual((r['episodes'],r['original_execution_receipts'],r['duplicate_original_receipts']),(63,135,72))
            self.assertEqual(r['eligible'],47)
            self.assertTrue(all(x['metrics']['host_illegal_actions']==0 for x in r['entries']))
            self.assertEqual(len(list(Path(temp).glob('P17-*-blind-A4.svg'))),4)

if __name__=='__main__':unittest.main()
