"""Small bounded correctness tests. Run explicitly; no CI training on push."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np

MOD_PATH = Path(__file__).with_name('ft020_resident_inference.py')
spec = importlib.util.spec_from_file_location('ft020_resident_inference', MOD_PATH)
ft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ft)

class FT020Tests(unittest.TestCase):
    def test_synthetic_menus_are_deterministic_and_bounded(self):
        a = list(ft.menus(444, 6))
        b = list(ft.menus(444, 6))
        self.assertEqual(len(a),6)
        for (names,x),(names2,y) in zip(a,b):
            self.assertEqual(names,names2)
            self.assertEqual(names.count('STOP'),1)
            self.assertEqual(x.shape[1],20)
            self.assertEqual(x.shape[0],len(names))
            self.assertTrue(np.array_equal(x,y))
            self.assertTrue(np.isfinite(x).all())
            self.assertEqual(float(x[names.index('STOP'),3]),1.)

    def test_optimizer_updates_change_real_model_weights(self):
        p = ft.Policy(718)
        before=p.p['W1'].copy()
        names,X=next(ft.menus(113,1))
        target=ft.softmax(ft.score('navigator',X))
        losses=[p.teach(X,target) for _ in range(24)]
        self.assertEqual(p.updates,24)
        self.assertTrue(np.isfinite(losses).all())
        self.assertGreater(float(np.linalg.norm(p.p['W1']-before)),.001)
        self.assertLess(losses[-1],losses[0])

    def test_four_policy_residency_keeps_identical_decisions(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'ft020'
            out=ft.run(root,16)
            self.assertEqual(out['status'],'completed')
            self.assertEqual([m['resident_models_per_process'] for m in out['modes']],[1,2,4])
            self.assertEqual([m['process_count'] for m in out['modes']],[4,2,1])
            self.assertTrue(out['parity_verified'])
            self.assertEqual(len({v['sha256'] for v in out['models'].values()}),4)
            for v in out['models'].values():
                self.assertEqual(v['adam_updates'],180)
                self.assertLess(v['heldout']['cross_entropy'],v['initial']['cross_entropy'])
            self.assertTrue((root/'SHA256SUMS').is_file())

if __name__=='__main__':
    unittest.main()
