"""Source/evidence regression guards for the C64 experimental cohort."""
import base64,hashlib,json,os,sys,unittest,zlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import c64_delayed_frontier_credit as c64

class C64Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output=Path(os.environ.get('DRIV3R_C64_EVIDENCE_DIR',str(HERE)))
        cls.summary=json.loads((cls.output/'C64-summary.json').read_text())
        cls.ledger=json.loads(zlib.decompress(base64.b64decode((cls.output/'C64-executed-ledger.zlib.b64').read_text())))
    def test_source_identity(self):
        self.assertEqual(self.summary['sources']['c64_delayed_frontier_credit.py'],hashlib.sha256((HERE/'c64_delayed_frontier_credit.py').read_bytes()).hexdigest())
        self.assertEqual(self.summary['sources']['c63_counterfactual_gate.py'],hashlib.sha256((HERE/'c63_counterfactual_gate.py').read_bytes()).hexdigest())
    def test_train_test_independence(self):
        r=self.summary['graph_receipts']
        train={x['graph'] for group in r[:2] for x in group}
        test={x['graph'] for group in r[2:] for x in group}
        self.assertFalse(train&test)
        self.assertEqual(len(train),sum(map(len,r[:2])))
        self.assertEqual(len(test),sum(map(len,r[2:])))
    def test_matched_legal_outcomes_and_no_guidance(self):
        rows=self.ledger['records'];self.assertEqual(len(rows),self.summary['matched_condition_rows'])
        n=0
        for r in rows:
            outs=r['outcomes'];self.assertEqual(set(outs),set(c64.POLICIES))
            self.assertTrue(all(o['host_rejections']==0 for o in outs.values()))
            self.assertEqual(len({o['protected_sha'] for o in outs.values()}),1)
            if r['arm'][0]=='none':
                n+=1;self.assertEqual(len({(o['action_sha'],o['ink_sha']) for o in outs.values()}),1)
        self.assertEqual(n,self.summary['unguided_parity_conditions'])
    def test_traces_match_executed_records(self):
        rec={(r['graph'],tuple(r['arm']),r['drop']):r for r in self.ledger['records']}
        for t in self.ledger['traces']:
            r=rec[(t['graph'],('backtrack',1.),0.)]
            z=r['outcomes'][t['policy']]
            self.assertEqual((z['action_sha'],z['ink_sha']),(t['action_sha'],t['ink_sha']))
            self.assertEqual(z['goal'],t['goal'])
        self.assertEqual(len(self.ledger['traces']),self.summary['replay_traces'])
    def test_optimizer_equal_steps_and_retrain_hashes(self):
        a=self.summary['optimizer']['aligned'];b=self.summary['optimizer']['shuffled']
        self.assertEqual((a['performed_updates'],b['performed_updates']),(c64.UPDATES,c64.UPDATES))
        for pol,name in [('C64-ALIGNED','aligned'),('C64-SHUFFLED','shuffled')]:
            p=self.output/(pol+'.safetensors')
            self.assertEqual(len(self.summary['checkpoints'][pol]),64)
            if p.exists():self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),self.summary['checkpoints'][pol])
    def test_wrong_credit_and_caveats(self):
        self.assertGreater(len(self.summary['limitations']),4)
        self.assertEqual(self.ledger['schema'],'driv3r-c64-executed-ledger/1')
        self.assertTrue(c64.training_reward([{'reward':0.5},{'reward':-0.1}],{'goal':1},'aligned',__import__('numpy').random.default_rng()).shape==(2,))
if __name__=='__main__':unittest.main()
