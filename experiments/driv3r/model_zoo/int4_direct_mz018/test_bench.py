"""MZ-018 deterministic source/ABI/kernel tests; no pretrained artifact required."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from bench import compile_kernel, Kernels, pack_int4, unpack_reference, run

class NativePackedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.library = Path(cls.tmp.name) / "int4.so"
        compile_kernel(cls.library)
        cls.kernels = Kernels(cls.library)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_even_odd_and_sign(self):
        for cols in (1, 2, 3, 7, 31, 96, 155, 224, 379):
            w = np.random.default_rng(cols).normal(size=(13, cols)).astype(np.float32)
            packed, scales = pack_int4(w)
            x = np.random.default_rng(20 + cols).normal(size=cols).astype(np.float32)
            y = np.empty(13, dtype=np.float32)
            self.kernels.direct_int4(packed, scales, x, y)
            ref = unpack_reference(packed, scales, cols) @ x
            np.testing.assert_allclose(y, ref, rtol=3e-5, atol=2e-5)

    def test_negative_eight_decodes_and_padding_ignored(self):
        p = np.array([[0x87, 0xf8]], dtype=np.uint8)
        s = np.array([.5], dtype=np.float32)
        x = np.array([1., 2., 3.], dtype=np.float32)
        y = np.empty(1, dtype=np.float32)
        self.kernels.direct_int4(p,s,x,y)
        self.assertAlmostEqual(float(y[0]), (7-16-24)*.5)

    def test_no_mutation_of_packed_or_input(self):
        w = np.arange(15,dtype=np.float32).reshape(3,5)-7
        p,s=pack_int4(w)
        x=np.arange(5,dtype=np.float32)
        pp,ss,xx=p.copy(),s.copy(),x.copy()
        self.kernels.direct_int4(p,s,x,np.empty(3,dtype=np.float32))
        np.testing.assert_array_equal(p,pp)
        np.testing.assert_array_equal(s,ss)
        np.testing.assert_array_equal(x,xx)

    def test_wrong_checkpoint_denied(self):
        with tempfile.TemporaryDirectory() as td:
            fake=Path(td)/"fake.pt";fake.write_bytes(b"fake checkpoint")
            with self.assertRaisesRegex(ValueError, "sha256 mismatch"):
                run(fake,batch=1,repeats=3)

    def test_bad_shapes_rejected(self):
        with self.assertRaises(ValueError):
            pack_int4(np.ones(5,dtype=np.float32))
        with self.assertRaises(ValueError):
            pack_int4(np.ones((2,3),dtype=np.float64))

if __name__ == '__main__':
    unittest.main()
