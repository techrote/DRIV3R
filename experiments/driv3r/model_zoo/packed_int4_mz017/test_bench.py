"""Deterministic tests for genuine nibble packing and source custody."""
import unittest
import torch
from bench import pack_int4, unpack_int4, run


class PackedInt4Tests(unittest.TestCase):
    def test_signed_nibble_roundtrip_even_odd(self):
        torch.manual_seed(4)
        for width in (1, 2, 7, 16, 155, 224):
            w = torch.randn(13, width)
            packed, scales, cols = pack_int4(w)
            out = unpack_int4(packed, scales, cols)
            q = (w / scales[:, None]).round().clamp(-7, 7)
            self.assertTrue(torch.equal(out, q * scales[:, None]))
            self.assertEqual(packed.dtype, torch.uint8)
            self.assertEqual(packed.shape[1], (width + 1) // 2)

    def test_no_quantization_input_mutation(self):
        x = torch.randn(8, 15)
        ref = x.clone()
        pack_int4(x)
        self.assertTrue(torch.equal(x, ref))

    def test_type_rejections(self):
        with self.assertRaises(ValueError):
            pack_int4(torch.ones(5))
        with self.assertRaises(ValueError):
            pack_int4(torch.ones(3, 5, dtype=torch.float16))

    def test_source_mismatch_hard_fails(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "wrong.pt"
            p.write_bytes(b"not an authorised model")
            with self.assertRaisesRegex(ValueError, "checkpoint SHA256"):
                run(p, decisions=1, repeats=3)


if __name__ == "__main__":
    unittest.main()
