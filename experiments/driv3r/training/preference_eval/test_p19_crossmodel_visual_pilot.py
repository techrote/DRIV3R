"""Source integrity, anonymity, leakage, and deterministic CPU-ranker tests for P19."""
import csv, hashlib, json, os, tempfile, unittest
from pathlib import Path
import numpy as np
import p19_crossmodel_visual_pilot as p

ROOT=Path(__file__).resolve().parent
P18=Path(os.environ.get('P19_P18_ARCHIVE','/mnt/data/DRIV3R_P18_ORIGINAL_PRODUCER_REVERIFIED_BLIND_BALLOTS_LOCAL.zip'))
P10=Path(os.environ.get('P19_P10_ARCHIVE','/mnt/data/driv3r_preference_P10_real_W16_STOP_clock_model_inference_qualified.zip'))
VOTES=Path(os.environ.get('P19_VOTES',str(ROOT/'P19_MODEL_DERIVED_VISUAL_PILOT.csv')))

class OriginalSourceTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  if not (P18.exists() and P10.exists()): raise unittest.SkipTest('Original producer archives absent; provide P19_P18_ARCHIVE and P19_P10_ARCHIVE')
  cls.pairs,cls.lfm=p.sources(P18,P10)

 def test_01_fixed_source_digests(self):
  self.assertEqual(p.sha(P18.read_bytes()),p.P18_SHA)
  self.assertEqual(p.sha(P10.read_bytes()),p.P10_SHA)

 def test_02_real_sources_and_no_duplicate_path(self):
  self.assertEqual(len(self.pairs),28)
  self.assertEqual(sum(x['cohort']=='P10_W16' for x in self.pairs),18)
  self.assertEqual(sum(x['cohort']=='P18_HISTORICAL' for x in self.pairs),10)
  self.assertEqual(len({x['id'] for x in self.pairs}),28)
  for x in self.pairs:self.assertNotEqual(x['a']['positions'],x['b']['positions'])

 def test_03_genuine_pretrained_LFM_TIES_are_not_preferences(self):
  self.assertEqual(self.lfm['paired_worlds'],9)
  self.assertEqual(self.lfm['same_path'],9)
  self.assertEqual(self.lfm['different_logit_ticks'],216)
  self.assertTrue(all(x['cohort']!='GP350006_TIE' for x in self.pairs))

 def test_04_blind_ids_reveal_neither_cohort_nor_model(self):
  self.assertTrue(all(x['id'].startswith('B19-') for x in self.pairs))
  pages=p.pair_svg(self.pairs)
  self.assertEqual(len(pages),14)
  for svg in pages:
   self.assertNotIn('P18-',svg)
   self.assertNotIn('P19-',svg)
   self.assertNotIn('W16',svg)
   self.assertNotIn('Franken',svg)
   self.assertNotIn('LFM350',svg)
   self.assertNotIn('model-derived',svg)

 def test_05_legal_original_p10_traces_are_individually_replayed(self):
  # sources() checks all 27 original and all 27 counterfactual traces, event-by-event.
  self.assertTrue(all(len(x['a']['positions'])==x['a']['decisions']+1 for x in self.pairs))
  self.assertTrue(all(len(x['b']['positions'])==x['b']['decisions']+1 for x in self.pairs))
  self.assertTrue(all(x['a']['source_json_sha256']!=x['b']['source_json_sha256'] for x in self.pairs))

 def test_06_antisymmetry_of_pairwise_features(self):
  for pair in self.pairs:
   x=p.feat(pair['a'])-p.feat(pair['b'])
   y=p.feat(pair['b'])-p.feat(pair['a'])
   np.testing.assert_array_equal(x,-y)
   self.assertEqual(x.shape,(12,))
   self.assertTrue(np.isfinite(x).all())

 def test_07_votes_model_only_separate_four_niches(self):
  with open(VOTES,newline='') as f: rows=list(csv.DictReader(f))
  self.assertEqual(len(rows),112)
  self.assertEqual({x['rating_source'] for x in rows},{'GPT6_ASSISTANT_VISUAL_PILOT_NOT_HUMAN'})
  self.assertEqual({x['niche'] for x in rows},set(p.NICHES))
  self.assertEqual(len({(x['pair_id'],x['niche']) for x in rows}),len(rows))
  self.assertEqual({x['pair_id'] for x in rows},{x['id'] for x in self.pairs})
  self.assertEqual(sum(x['vote_A_B_TIE_SKIP']=='TIE' for x in rows),18)

 def test_08_train_test_source_disjoint_and_real_optimizer(self):
  data=[(np.array([i/20 for i in range(12)]),1),(np.array([-i/22 for i in range(12)]),0)]
  w1,s1=p.fit_logit(data,epochs=30)
  w2,s2=p.fit_logit(data,epochs=30)
  np.testing.assert_array_equal(w1,w2)
  np.testing.assert_array_equal(s1,s2)
  self.assertFalse(np.all(w1==0))
  for niche in p.NICHES:
   self.assertTrue({x['cohort'] for x in self.pairs}=={'P10_W16','P18_HISTORICAL'})

 def test_09_tampered_archive_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   dest=Path(d)/'tampered.zip'
   b=bytearray(P18.read_bytes());b[len(b)//2]^=1;dest.write_bytes(b)
   with self.assertRaisesRegex(ValueError,'SOURCE SHA-256 MISMATCH'):p.checked_zip(dest,p.P18_SHA)

 def test_10_invalid_author_injection_rejected(self):
  with open(VOTES,newline='') as f: rows=list(csv.DictReader(f))
  rows[0]['rating_source']='OWNER_HUMAN'
  with tempfile.TemporaryDirectory() as d:
   bogus=Path(d)/'bad.csv'
   with bogus.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
   with self.assertRaises(AssertionError):p.fit_report(self.pairs,bogus,Path(d)/'out.json')

if __name__=='__main__':unittest.main(verbosity=2)