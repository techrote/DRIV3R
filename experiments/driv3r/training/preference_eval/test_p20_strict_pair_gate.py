"""P20 gate tests; provide P20_P19_ZIP when outside original session."""
import csv,os,tempfile,unittest
from pathlib import Path
import p20_strict_pair_gate as p
BUNDLE=Path(os.environ.get('P20_P19_ZIP','/mnt/data/DRIV3R_P19_FOUR_NICHE_ORIGINAL_MODEL_VISUAL_PILOT_NEGATIVE_LOCAL.zip'))
class GateTests(unittest.TestCase):
 def test_verified_confound_and_no_fabricated_preferences(self):
    with tempfile.TemporaryDirectory() as d:
        r=p.audit(BUNDLE,d)
        self.assertEqual((r['original_pairs'],r['W16_shared_movement_until_STOP'],r['historical_independent_world_blocks']),(28,18,3))
        self.assertEqual((r['W16_assistant_preferred_longer'],r['W16_non_tie_assistant_labels']),(53,57))
        self.assertEqual((r['original_pretrained_LFM350_trained_vs_stock_identical_paths'],r['original_LFM350_changed_logit_ticks']),(9,216))
        self.assertEqual((r['human_owner_labels'],r['new_root_SLM_forwards']), (0,0))
        rows=list(csv.DictReader((Path(d)/'P20_UNRATED_OWNER_BALLOT.csv').read_text().splitlines()))
        self.assertEqual(len(rows),40);self.assertTrue(all(not q['owner_A_B_TIE_SKIP'] for q in rows))
        pages=list(Path(d).glob('P20_HISTORICAL_BLIND_*.svg'));self.assertEqual(len(pages),5)
        self.assertNotIn('Greyhound',''.join(x.read_text() for x in pages))
 def test_tampered_original_rejected(self):
    with tempfile.TemporaryDirectory() as d:
        b=bytearray(BUNDLE.read_bytes());b[31]^=1;f=Path(d)/'mutant.zip';f.write_bytes(b)
        with self.assertRaisesRegex(ValueError,'checksum mismatch'):p.audit(f,d)
if __name__=='__main__':unittest.main()
