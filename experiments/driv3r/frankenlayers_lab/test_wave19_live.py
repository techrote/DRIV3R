"""Source-only Wave 19 qualification: no model files or external runner required."""
import json
import inspect
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from experiments.driv3r.frankenlayers_lab import counterfactual as cf
from experiments.driv3r.frankenlayers_lab import wave19_live as w


class Wave19Tests(unittest.TestCase):
    def test_imported_parent_run_arm_signature_is_exactly_five_arguments(self):
        self.assertEqual(list(inspect.signature(w.paired.run_arm).parameters),
                         ['binary', 'model', 'w0', 'mode', 'out'])

    def test_evidence_sha_requires_matching_clean_git_checkout(self):
        def mock_run(args, **kwargs):
            if args[1:3] == ['rev-parse', '--verify']:
                return SimpleNamespace(stdout='a'*40 + '\\n')
            return SimpleNamespace(stdout='')
        with patch.object(w.subprocess, 'run', side_effect=mock_run):
            self.assertEqual(w.validate_source_checkout('a'*40), 'a'*40)
            with self.assertRaises(ValueError):
                w.validate_source_checkout('b'*40)
        def dirty_run(args, **kwargs):
            return SimpleNamespace(stdout='a'*40 if args[1] == 'rev-parse' else ' M experiments/driv3r/frankenlayers_lab/wave19_live.py')
        with patch.object(w.subprocess, 'run', side_effect=dirty_run):
            with self.assertRaises(ValueError):
                w.validate_source_checkout('a'*40)

    def test_fold_context_binds_actual_parent_contract_and_restores_on_error(self):
        before = (w.paired.SEED, w.paired.STEPS, w.paired.FAMILIES,
                  cf.SEED, cf.STEPS, cf.FAMILY, cf.ALL_VARIANTS, cf.PAIRS, cf.variant_obstacles)
        with self.assertRaisesRegex(RuntimeError, 'deliberate'):
            with w.fold_context(*w.FOLDS[0]):
                self.assertEqual(w.paired.SEED, 6089)
                self.assertEqual(w.paired.FAMILIES, ('pipework',))
                self.assertEqual(cf.FAMILY, 'pipework')
                self.assertEqual(cf.variant_obstacles('east_blocked'), {(12, 48)})
                self.assertEqual(cf.variant_obstacles('north_blocked'), {(10, 46), (11, 46)})
                with self.assertRaises(ValueError):
                    cf.variant_obstacles('east_blocked', 12)
                raise RuntimeError('deliberate')
        self.assertEqual(before, (w.paired.SEED, w.paired.STEPS, w.paired.FAMILIES,
                  cf.SEED, cf.STEPS, cf.FAMILY, cf.ALL_VARIANTS, cf.PAIRS, cf.variant_obstacles))
        with self.assertRaises(ValueError):
            with w.fold_context(4979, 'greebles', 2):
                pass

    def test_parent_five_argument_api_with_four_variant_host_episodes(self):
        invocations = []

        def fake_run_arm(binary, model, host, mode, out):
            invocations.append((w.paired.SEED, w.paired.STEPS, w.paired.FAMILIES, host.variant))
            self.assertEqual(len(invocations), w.VARIANTS.index(host.variant) + 1)
            self.assertEqual(mode, w.MODE)
            obs = {'tick': 0, 'legal': ['RIGHT', 'BRANCH'], 'x': 7, 'y': 48}
            event = {'status': 'ok', 'choice': 'RIGHT', 'observation': obs}
            out.mkdir(parents=True, exist_ok=True)
            (out / f'{mode}-{w.FOLDS[1][1]}-choices.json').write_text(json.dumps([event]))
            return {'model_decisions': 1, 'p50_seconds': 0.4,
                    'experiments': [{'raw_choices': ['RIGHT'], 'metrics': {
                        'new_steps': 1, 'total_revisit_contacts': 0,
                        'terminated_by_stop': False, 'invalid_raw_choices': 0},
                        'decisions': 1}]}

        with TemporaryDirectory() as tmp, patch.object(w.paired, 'run_arm', fake_run_arm):
            result = w.run_fold(Path('/dummy'), Path('/model'), object(), w.MODE,
                                Path(tmp), w.FOLDS[1])
        self.assertEqual(len(invocations), 4)
        self.assertTrue(all(a[0] == 7127 and a[1] == w.STEPS and a[2] == ('greebles',)
                            for a in invocations))
        self.assertEqual(result['summary']['decisions'], 4)
        self.assertEqual(result['summary']['budget'], 4*w.STEPS)
        self.assertEqual(result['summary']['invalid_actions'], 0)
        self.assertEqual(result['summary']['exposed_pairs'], 0)

    def test_loop_escape_follows_observed_past_not_family_or_hidden_map(self):
        class Base:
            def __init__(self, *args, **kwargs):
                pass
            def choose(self, obs, records):
                records.append({'parsed': {'probabilities': {'RIGHT': .6, 'BRANCH': .4}}})
                return 'RIGHT'
        def obs(tick, x, heading, wall, history):
            return {'tick': tick, 'x': x, 'y': 48, 'heading': heading,
                    'legal': ['RIGHT', 'BRANCH', 'RETRACE'],
                    'local_5x5': ['.....', '.....', '..@.#' if wall else '..@..', '.....', '.....'],
                    'last_actions': history}
        with patch.object(w.paired, 'Server', Base), patch.object(w, 'predict', side_effect=lambda o, r, p, c: ('BRANCH' if '#' in ''.join(o['local_5x5']) else 'RETRACE', {})):
            with w.model_policy({'mock': True}, 'loop'):
                server = w.paired.Server()
                calls = [obs(0, 8, 1, True, []),
                         obs(1, 7, 2, False, ['RIGHT']),
                         obs(2, 8, 1, True, ['RIGHT','RETRACE']),
                         obs(3, 7, 2, False, ['RIGHT','RETRACE','RIGHT']),
                         obs(4, 8, 1, True, ['RIGHT','RETRACE','RIGHT','RETRACE'])]
                choices = [server.choose(o, []) for o in calls]
                self.assertEqual(choices, ['RIGHT', 'RETRACE', 'RIGHT', 'RETRACE', 'BRANCH'])
                second = w.paired.Server()
                self.assertEqual(second.choose(calls[-1], []), 'RIGHT')
                self.assertEqual(server.choose(obs(0, 8, 1, True, []), []), 'RIGHT')
            self.assertIs(w.paired.Server, Base)

    def test_pareto_reports_independent_axes_not_one_rank(self):
        def fold():
            return {name: {'summary': {'decisions': 12, 'budget': 96,
                'new_cells': 7, 'revisits': 3, 'early_stop_episodes': 1,
                'invalid_actions': 0, 'exposed_pairs': 1, 'responded_pairs': 1,
                'unexplained_divergences': 0, 'median_decision_seconds': .5,
                'choice_counts': {'STRAIGHT': 4, 'LEFT': 0, 'RIGHT': 0,
                                  'BRANCH': 8, 'RETRACE': 0, 'STOP': 0}}}
                for name in w.ARMS}
        summary = w.compare([fold(), fold()])
        p = summary['q4_mutant_loop']
        self.assertEqual((p['decisions'], p['budget'], p['novel_cells']), (24, 192, 14))
        self.assertEqual((p['exposed_pairs'], p['responded_exposed_pairs']), (2, 2))
        self.assertEqual(p['premature_stop_episodes'], 2)
        self.assertEqual(p['legal_fraction'], 1.0)
        self.assertGreater(p['action_entropy_bits'], .9)


if __name__ == '__main__':
    unittest.main()
