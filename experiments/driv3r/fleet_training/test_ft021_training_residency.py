"""FT-021 focused correctness and failure-custody tests (explicit opt-in)."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import json
import tempfile
import unittest
import numpy as np

SRC = Path(__file__).with_name('ft021_training_residency.py')
spec = importlib.util.spec_from_file_location('ft021_training_residency', SRC)
ft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ft)


class FT021Tests(unittest.TestCase):
    def test_deterministic_separated_legal_synthetic_menus(self):
        a=ft.menus('navigator','train',24)
        b=ft.menus('navigator','train',24)
        hold=ft.menus('navigator','heldout',24)
        for (names,X),(bnames,B) in zip(a,b):
            self.assertEqual(names,bnames)
            np.testing.assert_array_equal(X,B)
            self.assertEqual(names.count('STOP'),1)
            self.assertEqual(X.shape,(len(names),20))
            self.assertTrue(np.isfinite(X).all())
        self.assertFalse(np.array_equal(a[0][1],hold[0][1]))

    def test_foreign_lease_and_budget_rejected(self):
        with self.assertRaises(ValueError):ft.menus('foreign','train',2)
        with self.assertRaises(ValueError):ft.menus('navigator','train',513)
        with self.assertRaises(ValueError):ft.run('/missing','/notrun',1201)
        with tempfile.TemporaryDirectory() as td:
            sp=Path(td)/'spec.json'
            sp.write_text(json.dumps(dict(schema=ft.SCHEMA,mode='two_resident',
                niches=['navigator','navigator'],updates=2,lease='invalid',checkpoint_dir=td)))
            with self.assertRaises(ValueError):ft.child(sp,Path(td)/'result.json')

    def test_adam_updates_and_nonzero_weight_change(self):
        p=ft.Policy({'W1':np.ones((20,24))*.03,'b1':np.zeros(24),
                    'W2':np.ones(24)*.04,'b2':np.zeros(1)})
        before=p.p['W1'].copy()
        _,X=ft.menus('branch_artist','train',1)[0]
        y=ft.softmax(ft.shifted_teacher('branch_artist',X)/.42)
        losses=[p.teach(X,y) for _ in range(18)]
        self.assertEqual(p.updates,18)
        self.assertTrue(np.isfinite(losses).all())
        self.assertLess(losses[-1],losses[0])
        self.assertGreater(float(np.linalg.norm(p.p['W1']-before)),1e-3)

    def test_matched_four_experiment_dispatch_and_parent_custody(self):
        # External model archive, not committed to source. Skip absent fixtures.
        fixture=Path('/mnt/data/ft021_inputs/ft020_long/checkpoints')
        if not fixture.is_dir():self.skipTest('Requires original FT020 local checkpoint fixture')
        with tempfile.TemporaryDirectory() as td:
            r=ft.run(fixture,Path(td)/'result',12)
            self.assertEqual(r['status'],'completed')
            self.assertTrue(r['weight_and_metric_parity'])
            self.assertEqual(r['updates_total'],96)
            self.assertEqual(sorted(x['process_count'] for x in r['modes']),[2,4])
            self.assertEqual(len({x['model_sha256'] for m in r['modes'] for x in m['models']}),4)
            self.assertTrue((Path(td)/'result'/'SHA256SUMS').is_file())
            self.assertEqual(len({x['niche'] for m in r['modes'] for x in m['models']}),4)
        with tempfile.TemporaryDirectory() as td:
            for niche in ft.NICHES:
                (Path(td)/(niche+'.json')).write_bytes((fixture/(niche+'.json')).read_bytes())
            (Path(td)/'navigator.json').write_text('{}')
            with self.assertRaises(ValueError):ft.validate_parent(td,'navigator')


if __name__=='__main__':unittest.main()
