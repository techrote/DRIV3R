"""CPU-only structural preflight: NOT a pretrained-model qualification."""
import importlib.util, pathlib, sys, types, unittest
import torch
from torch import nn

pkg='experiments.driv3r.training.tiny_policy'
for i in range(1,len(pkg.split('.'))+1):
    part='.'.join(pkg.split('.')[:i])
    if part not in sys.modules:
        module=types.ModuleType(part);module.__path__=[];sys.modules[part]=module
sys.modules[pkg+'.m15_pretrained_goal']=types.ModuleType(pkg+'.m15_pretrained_goal')
spec=importlib.util.spec_from_file_location(pkg+'.m19_adapter',pathlib.Path(__file__).with_name('m19_adapter.py'))
m19=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m19;spec.loader.exec_module(m19)

class TestM19(unittest.TestCase):
    def test_initial_equivalence_and_gradient(self):
        linear=nn.Linear(8,8);lora=m19.LowRank(linear)
        x=torch.randn(2,8)
        torch.testing.assert_close(lora(x),linear(x))
        lora(x).square().sum().backward()
        self.assertIsNone(linear.weight.grad)
        self.assertGreater(float(lora.B.grad.detach().abs().sum()),0)
    def test_inject_last_two_only(self):
        class Layer(nn.Module):
            def __init__(self):
                super().__init__();self.self_attn=nn.Module()
                self.self_attn.q_proj=nn.Linear(8,8);self.self_attn.v_proj=nn.Linear(8,8)
        class Model(nn.Module):
            def __init__(self):
                super().__init__();self.layers=nn.ModuleList([Layer() for _ in range(4)])
        model=Model().requires_grad_(False)
        self.assertEqual(len(m19.inject(model)),4)
        self.assertEqual(len(m19.adapter_parameters(model)),8)
        self.assertIsInstance(model.layers[0].self_attn.q_proj,nn.Linear)
        self.assertIsInstance(model.layers[3].self_attn.q_proj,m19.LowRank)
    def test_train_frozen_and_lora(self):
        class Fake(nn.Module):
            def __init__(self):
                super().__init__();self.proj=m19.LowRank(nn.Linear(8,8).requires_grad_(False))
            def forward(self,**kwargs):
                return types.SimpleNamespace(last_hidden_state=self.proj(kwargs['input_ids'].float()))
        tok=lambda texts,**k:{'input_ids':torch.randn(len(texts),2,8),'attention_mask':torch.ones(len(texts),2,dtype=torch.long)}
        net=Fake();rows=[{'text':'x','label':0,'legal':[0,1,4]}];head=nn.Linear(8,5)
        _,f=m19.train(net,tok,head,rows,[[0]],False)
        _,a=m19.train(net,tok,head,rows,[[0]],True)
        self.assertTrue(torch.isfinite(torch.tensor(f+a)).all())
        self.assertGreater(sum(float(p.detach().abs().sum()) for _,p in m19.adapter_parameters(net)),0)

if __name__=='__main__':unittest.main()
