"""Source-only checks. NOT a real pretrained-model qualification."""
import unittest
from .run import MODEL,PIN,legal_choice,pct,validate_budget
class Tests(unittest.TestCase):
    def test_source_pin(self):
        self.assertEqual(MODEL,"tiiuae/Falcon3-1B-Instruct-1.58bit")
        self.assertEqual(PIN,"72fd3f95fcd82639c902304919629edda8c6f2b4")
    def test_legal_host_mask(self):
        self.assertEqual(legal_choice([40,2,1],("LEFT","RIGHT","STOP"),
                                      ("RIGHT","STOP")),"RIGHT")
        with self.assertRaises(ValueError):
            legal_choice([1,2],("LEFT","STOP"),())
    def test_bounded(self):
        from types import SimpleNamespace as S
        a=S(train_seeds=14,eval_seeds=4,steps=9,updates=64,horizon=16,batch=12)
        validate_budget(a)
        a.updates=161
        with self.assertRaises(ValueError):validate_budget(a)
    def test_percentile(self):
        self.assertEqual(pct([1,2,3,4,5],.95),4)
if __name__=="__main__":unittest.main()
