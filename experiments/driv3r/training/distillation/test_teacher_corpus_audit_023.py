"""Pure-stdlib regression checks for the D023 provenance gate."""
import unittest
try:
    from .teacher_corpus_audit_023 import ACTIONS, validate_receipt
except ImportError:
    from teacher_corpus_audit_023 import ACTIONS, validate_receipt

class TestCorpusValidation(unittest.TestCase):
    def setUp(self):
        self.obs = {'legal': ['STRAIGHT', 'LEFT']}
        self.p = {a: 0.0 for a in ACTIONS}
        self.p['STRAIGHT'] = 0.6
        self.p['LEFT'] = 0.4

    def test_valid(self):
        validate_receipt('LEFT', self.obs, self.p)

    def test_reject_illegal_chosen(self):
        with self.assertRaises(ValueError):
            validate_receipt('BRANCH', self.obs, self.p)

    def test_reject_illegal_probability(self):
        self.p['RIGHT'] = 0.2
        self.p['STRAIGHT'] = 0.4
        with self.assertRaises(ValueError):
            validate_receipt('LEFT', self.obs, self.p)

    def test_reject_bad_normalization(self):
        self.p['LEFT'] = 0.1
        with self.assertRaises(ValueError):
            validate_receipt('LEFT', self.obs, self.p)

    def test_reject_duplicate_menu(self):
        self.obs['legal'].append('LEFT')
        with self.assertRaises(ValueError):
            validate_receipt('LEFT', self.obs, self.p)

if __name__ == '__main__':
    unittest.main()
