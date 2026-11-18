"""C65 source-matched synthetic-host and optimizer exposure regression checks."""
import base64,hashlib,json,os,sys,unittest,zlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import c65_paired_delayed_credit as c65

class C65Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out=Path(os.environ.get('DRIV3R_C65_EVIDENCE_DIR',str(HERE)))
        cls.s=json.loads((cls.out/'C65-summary.json').read_text())
        cls.ledger=json.loads(zlib.decompress(base64.b64decode((cls.out/'C65-executed-ledger.zlib.b64').read_text())))
    def test_provenance(self):
        src=self.s['source_sha256']
        self.assertEqual(src['c65_paired_delayed_credit.py'],hashlib.sha256((HERE/'c65_paired_delayed_credit.py').read_bytes()).hexdigest())
        self.assertEqual(src['c64_delayed_frontier_credit.py'],hashlib.sha256((HERE/'c64_delayed_frontier_credit.py').read_bytes()).hexdigest())
    def test_graph_separation(self):
        r=self.s['graph_receipts'];tr={x['graph'] for g in r[:2] for x in g};te={x['graph'] for g in r[2:] for x in g}
        self.assertTrue(tr and te);self.assertFalse(tr&te)
        self.assertEqual(sum(len(g) for g in r),len(tr)+len(te))
    def test_equal_training_exposure(self):
        a=self.s['optimizer']['ALIGNED'];b=self.s['optimizer']['SHIFT']
        self.assertEqual(a['shared_training_actions'],b['shared_training_actions'])
        self.assertEqual(a['shared_training_actions'],self.s['training_actions_each'])
        self.assertEqual((a['updates'],b['updates']),(c65.UPDATES,c65.UPDATES))
        self.assertEqual(a['matched_minibatches'],b['matched_minibatches'])
        self.assertEqual(sum(x['steps'] for x in self.s['training_receipts']),self.s['training_actions_each'])
    def test_legal_outcomes(self):
        rr=self.ledger['records'];self.assertEqual(len(rr),self.s['matched_condition_rows'])
        no=0
        for row in rr:
            results=row['outcomes'];self.assertEqual(set(results),set(c65.POLICIES))
            self.assertTrue(all(z['host_rejections']==0 for z in results.values()))
            self.assertEqual(len({z['protected_sha'] for z in results.values()}),1)
            if row['arm'][0]=='none':
                no+=1;self.assertEqual(len({(z['action_sha'],z['ink_sha']) for z in results.values()}),1)
        self.assertEqual(no,self.s['unguided_parity_conditions'])
    def test_replayed_traces(self):
        for t in self.ledger['traces']:
            row=next(r for r in self.ledger['records'] if r['graph']==t['graph'] and r['drop']==0. and r['arm']==['backtrack',1.])
            z=row['outcomes'][t['policy']]
            self.assertEqual((z['action_sha'],z['ink_sha'],z['goal']),(t['action_sha'],t['ink_sha'],t['goal']))
        self.assertEqual(len(self.ledger['traces']),self.s['replayed_traces'])
    def test_checkpoints_and_control(self):
        for pol,h in self.s['checkpoint_sha256'].items():
            f=self.out/(pol+'.safetensors')
            if f.exists():self.assertEqual(hashlib.sha256(f.read_bytes()).hexdigest(),h)
        self.assertTrue(any('Offline' in l for l in self.s['limitations']))
        self.assertEqual(self.ledger['schema'],'driv3r-c65-executed-ledger/1')

if __name__=='__main__':unittest.main()
