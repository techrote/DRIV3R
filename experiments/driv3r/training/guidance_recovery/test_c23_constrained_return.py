"""Offline C23 synthetic host and scientific-gate regressions; no checkpoints needed."""
from __future__ import annotations
import copy,unittest
import numpy as np
import torch
from torch import nn
from c23_constrained_return import suffix,acceptance
from c22_branch_personality import world,StyleSidecar,fav_opposition
from attested_stack_cue import observe32
from signed_feedback import HostAdvice
from experiment import ARMS,sha

class Dummy(nn.Module):
    def forward(self,x):
        return (.25*x[...,7]+.1*x[...,21]+.15*x[...,0]).masked_fill(x[...,4]<.5,-1e7)

class ResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_real_host_pair_repeat_and_no_mutation(self):
        w=world(2311,'hairpin',1,'reverse',2,False,1,7)
        before=sha({'pos':w.pos,'history':w.history,'protected':sorted(w.protected),
                    'ink':sorted(w.inked)})
        a=HostAdvice(w);a.issue_backtrack(w,strength=.35,ttl=9)
        menu,x=observe32(w,.35,a);fav,opp=fav_opposition(x)
        self.assertTrue(fav.any() and opp.any())
        j=int(np.flatnonzero(fav)[0]);k=int(np.flatnonzero(opp)[0])
        r1=suffix(Dummy(),w,a,.35,j,771,18)
        r2=suffix(Dummy(),w,a,.35,j,771,18)
        r3=suffix(Dummy(),w,a,.35,k,771,18)
        self.assertEqual(r1,r2)
        self.assertTrue(all(np.isfinite(z) for z in r1))
        self.assertTrue(all(np.isfinite(z) for z in r3))
        self.assertTrue(menu[j].legal and menu[k].legal)
        self.assertEqual(before,sha({'pos':w.pos,'history':w.history,
                 'protected':sorted(w.protected),'ink':sorted(w.inked)}))

    def test_host_legality_is_unaffected_by_hint_strength(self):
        w=world(2312,'staircase',-1,'ahead',0,False,2,6)
        expected=[(c.action,c.target,c.legal) for c in w.candidates(0)]
        for arm in ARMS:
            self.assertEqual([c.legal for c in w.candidates(arm)],
                            [p[-1] for p in expected])

    def test_adapter_only_active_and_legal_preferred_ink(self):
        base=Dummy();mod=StyleSidecar(base,scale=1.2)
        with torch.no_grad():mod.head[-1].bias.fill_(.75)
        for directive in ('none','backtrack','continue'):
            w=world(2313,'hook',-1,'reverse',0,False,2,11)
            a=HostAdvice(w)
            if directive!='none':getattr(a,'issue_'+directive)(w,strength=1.,ttl=9)
            menu,x=observe32(w,.35,a);tensor=torch.from_numpy(x)
            with torch.no_grad():old=base(tensor[:,:29]);new=mod(tensor)
            fav,op=fav_opposition(x)
            self.assertTrue(torch.equal(old[8:],new[8:]))
            for j in range(8):
                if directive=='none' or not fav[j]:
                    self.assertTrue(torch.equal(old[j],new[j]))
            if directive=='none':self.assertTrue(torch.equal(old,new))
            self.assertEqual([c.legal for c in menu],x[:,4].astype(bool).tolist())

    def test_scientific_gate_rejects_single_group_regression(self):
        # Synthetic metrics-shaped rows, used only to verify a strict gate.
        rows=[]
        for name,n in [('C18-A0',4),('C23-GUARD',3),('C23-RETURN',4)]:
            for i in range(4):
                rows.append({'outcome':'reverse','directive':'backtrack','strength':.35,
                    'model':name,'reached':i<n,'distance':0 if i<n else 4,'steps':8,
                    'personality':1,'preferred':{'active':[1,1,0],'post':[0,0,0]}})
        audit={'models':{'C18-A0':{'fav_ink_mass':.25},'C23-GUARD':{'fav_ink_mass':.26},
                         'C23-RETURN':{'fav_ink_mass':.24}}}
        q=acceptance(rows,audit)
        self.assertFalse(q['C23-GUARD']['pass'])
        self.assertFalse(q['C23-RETURN']['pass'])
        self.assertEqual(len(q['C23-GUARD']['regressions']),1)
        self.assertFalse(q['C23-RETURN']['retention_nonregression'])

if __name__=='__main__': unittest.main()
