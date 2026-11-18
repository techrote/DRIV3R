"""P18 independently executable guardrails; source archives optional for local tests."""
import copy, csv, importlib.util, json, tempfile, unittest
from pathlib import Path
P=Path(__file__).parent
spec=importlib.util.spec_from_file_location('p18',P/'p18_reverify_ballots.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class TestP18(unittest.TestCase):
    def test_reference_hash_and_artifacts(self):
        rows=m.verify(P/'driv3r_wormpaths_source_manifest.json', '/mnt/data')
        self.assertEqual(len(rows),14)
        self.assertEqual(sum(r['decisions'] for r in rows),1096) # 72 native + 512 LFM + 448 mutagen + 64 Franken
        self.assertTrue(all(r['menu_violations']==0 for r in rows))
    def test_exact_original_geometry_refuses_tamper(self):
        obj=json.loads((P/'driv3r_wormpaths_source_manifest.json').read_text());obj['specimens'][0]['positions'][1][0]+=1
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'manifest.json';path.write_text(json.dumps(obj))
            with self.assertRaisesRegex(ValueError,'historical manifest hash mismatch'):m.verify(path,'/mnt/data')
    def test_balanced_pairing_and_provenance(self):
        r=m.verify(P/'driv3r_wormpaths_source_manifest.json','/mnt/data');pairs=m.paired(r)
        self.assertEqual(len(pairs),10)
        self.assertEqual(len({p['id'] for p in pairs}),10)
        self.assertEqual(len({tuple(p['world_block']) for p in pairs}),3)
        self.assertTrue(all(p['first']['world']==p['second']['world'] and p['first']['decisions']==p['second']['decisions'] for p in pairs))
    def test_blind_ballot_has_no_identity_or_fabricated_score(self):
        with tempfile.TemporaryDirectory() as t:
            d=m.execute(P/'driv3r_wormpaths_source_manifest.json','/mnt/data',t)
            with (Path(t)/'P18_BLIND_BALLOT.csv').open() as f: rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),40)
            self.assertTrue(all(not x['choice_A_B_tie_skip'] and not x['owner_confidence_1_to_5'] for x in rows))
            svgs=''.join(x.read_text() for x in Path(t).glob('*.svg'))
            self.assertEqual(svgs.count('<svg'),5)
            for forbidden in ('Greyhound','Stock d1','MLP scoop','LFM','Sprocket','Attention gouge','model_preference'):
                self.assertNotIn(forbidden,svgs)
            self.assertEqual(d['human_ratings'],0)
    def test_real_pretrained_lfm350_weight_change_without_path_change(self):
        d,pairs=m.verify_lfm_006('/mnt/data/gp350006-step96.zip')
        self.assertEqual((d['paired_worlds'],d['same_path'],d['same_actions'],d['model_forward_decisions'],d['different_logit_ticks']),(9,9,9,432,216))
        self.assertEqual(d['adapter_step96_sha256'],'436159bd6508063221d31d46f057b5fba7ee5b6f05b27edf003c54e76863898c')
    def test_catch_trials_are_not_pretended_independent_labels(self):
        with tempfile.TemporaryDirectory() as t:
            d=m.execute(P/'driv3r_wormpaths_source_manifest.json','/mnt/data',t,'/mnt/data/gp350006-step96.zip')
            self.assertEqual(d['matched_pairs'],13)
            self.assertEqual(d['lfm350_original_applied_identical_catch_trials'],3)
            self.assertEqual(d['historical_distinct_trajectory_pairs'],10)
            self.assertEqual(len(list(Path(t).glob('P18-blind-A4-*.svg'))),7)
            self.assertNotIn('CATCH', (Path(t)/'P18_BLIND_BALLOT.csv').read_text())
            self.assertNotIn('CATCH', ''.join(x.read_text() for x in Path(t).glob('*.svg')))
            pairs=json.loads((Path(t)/'P18_REVEAL_AFTER_RATING.json').read_text())
            self.assertEqual(sum(p['identical_result_catch_trial'] for p in pairs),3)
            self.assertTrue(all(p['owner_vote'] is None for p in pairs))
    def test_invalid_archive_bites(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(FileNotFoundError):m.verify(P/'driv3r_wormpaths_source_manifest.json',t)

if __name__=='__main__':unittest.main()
