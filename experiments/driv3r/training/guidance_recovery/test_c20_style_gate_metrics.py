"""Synthetic unit tests for C20 personality hard-gate accounting."""
import unittest
from c20_style_gate_metrics import metric, gate


def row(model,person,directive,seed=1,turns=20,reached=True):
    return {'model':model,'seed':seed,'outcome':'reverse','directive':directive,
      'personality':person,'arm':0.,'left':turns if person==1 else 0,
      'right':0 if person==1 else turns,'reached':reached,'host_rejected':0,
      'bad':0,'trace':[{'action':('INK',1,0)}],'ink_sha256':'same'}

class MetricsTest(unittest.TestCase):
    def test_signed_gap(self):
        m=metric([row('m',p,'backtrack') for p in (-1,1)])
        self.assertEqual(m['gap'],2.)
        self.assertEqual(m['goals'],2)

    def test_missing_turns_are_unknown(self):
        m=metric([row('m',-1,'backtrack',turns=0),row('m',1,'backtrack')])
        self.assertIsNone(m['gap'])

    def test_mismatched_cohort_rejected(self):
        rows=[row(model,p,d) for model in ('C12','C18-A0','C20-STYLE')
              for p in (-1,1) for d in ('none','backtrack','continue')]
        rows=[r for r in rows if not (r['model']=='C20-STYLE' and r['directive']=='backtrack' and r['personality']==-1)]
        with self.assertRaises(ValueError):gate(rows)

    def test_illegal_action_rejected(self):
        rows=[row(model,p,d) for model in ('C12','C18-A0','C20-STYLE')
              for p in (-1,1) for d in ('none','backtrack','continue')]
        rows[-1]['host_rejected']=1
        with self.assertRaises(ValueError):gate(rows)

if __name__=='__main__':unittest.main()
