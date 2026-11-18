"""P20 strict source/mechanical tests. These do NOT load pretrained weights."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SOURCE=Path(__file__).with_name("population_p20_real_slm.py")
spec=importlib.util.spec_from_file_location("population_p20_real_slm",SOURCE)
p20=importlib.util.module_from_spec(spec)
spec.loader.exec_module(p20)

class P20PreflightTests(unittest.TestCase):
    def test_envelope_rejects_excessive_or_unknown_work(self):
        self.assertTrue(p20.envelope(("navigator","recoverer"),(690,),4,32))
        for arguments in [
            (("no_such_niche",),(690,),4,32),
            (("navigator","navigator"),(690,),4,32),
            (("navigator",),(690,690),4,32),
            (("navigator",),(690,),13,32),
            (("navigator",),(690,),4,96),
            ((),(690,),4,32),
        ]:
            with self.assertRaises(ValueError):
                p20.envelope(*arguments)

    def test_missing_pretrained_is_explicit_blocker(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=p20.root_preflight(Path(tmp)/"absent","0"*64)
            self.assertEqual(result["status"],"BLOCKED")
            self.assertEqual(result["actual_optimizer_updates"],0)
            self.assertIn("pinned pretrained model directory missing",result["failures"])

    def test_sha256_is_compulsory_and_not_guessed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/"config.json").write_text("{}")
            (root/"tokenizer.json").write_text("{}")
            data=b"test fixture, NOT an actual pretrained model"
            (root/"model.safetensors").write_bytes(data)
            real=hashlib.sha256(data).hexdigest()
            ok=p20.root_preflight(root,real,require_packages=False)
            self.assertEqual(ok["status"],"READY")  # identity check only
            bad=p20.root_preflight(root,"f"*64,require_packages=False)
            self.assertIn("pinned model checksum mismatch",bad["failures"])
            self.assertEqual(p20.root_preflight(root,"BAD",require_packages=False)["status"],"BLOCKED")

    def test_legal_menu_and_raw_illegality_separate(self):
        scores=[0.1,9.0,0.0,-2.0,-1.0,-3.0]
        chosen,raw=p20.choose(scores,["N","RETRACE","STOP"])
        self.assertEqual(chosen,"N")
        self.assertEqual(raw,1)
        with self.assertRaises(ValueError):
            p20.choose(scores,["PAINT_ILL"])

    def test_pinned_roots_and_source_research_only(self):
        self.assertEqual(p20.REVISION,"12fd25f77366fa6b3b4b768ec3050bf629380bac")
        self.assertEqual(len(p20.NICHES),6)
        self.assertEqual(p20.HOLDOUT_CARDS,(4,5))
        self.assertFalse(set(p20.TRAIN_CARDS)&set(p20.HOLDOUT_CARDS))

    def test_real_lora_injection_initial_identity_and_gradients(self):
        import torch
        from torch import nn
        class Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn=nn.Module()
                self.self_attn.q_proj=nn.Linear(8,8)
                self.self_attn.v_proj=nn.Linear(8,8)
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers=nn.ModuleList([Layer() for _ in range(4)])
        torch.manual_seed(700)
        root=Tiny().eval().requires_grad_(False)
        x=torch.randn(2,3,8)
        ref=root.layers[3].self_attn.q_proj(x).detach().clone()
        names=p20.inject_lora(root,torch)
        self.assertEqual(len(names),4)
        self.assertEqual(len(p20.adapters(root)),8)
        torch.testing.assert_close(root.layers[3].self_attn.q_proj(x),ref)
        root.layers[3].self_attn.q_proj(x).square().sum().backward()
        self.assertIsNone(root.layers[3].self_attn.q_proj.base.weight.grad)
        self.assertGreater(float(root.layers[3].self_attn.q_proj.B.grad.abs().sum()),0)

if __name__=="__main__":
    unittest.main()
