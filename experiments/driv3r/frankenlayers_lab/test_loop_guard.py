import unittest
try:
    from .loop_guard import LoopAwareGate
except ImportError:
    from loop_guard import LoopAwareGate


def obs(tick, x=8, y=48, heading=1, history=None, wall=True):
    return {'tick': tick, 'x': x, 'y': y, 'heading': heading,
            'local_5x5': ['.....', '.....', '..@.#' if wall else '..@..', '.....', '.....'],
            'last_actions': history or [], 'legal': ['RIGHT', 'BRANCH', 'RETRACE', 'STOP']}


class LoopGuardTests(unittest.TestCase):
    def test_escapes_only_after_two_identical_pose_revisits_with_oscillation(self):
        gate = LoopAwareGate()
        self.assertEqual(gate.choose(obs(11), 'RIGHT', 'BRANCH')[0], 'RIGHT')
        self.assertEqual(gate.choose(obs(12, x=7, heading=2, wall=False), 'RIGHT', 'RETRACE')[0], 'RETRACE')
        self.assertEqual(gate.choose(obs(13), 'RIGHT', 'BRANCH')[0], 'RIGHT')
        self.assertEqual(gate.choose(obs(14, x=7, heading=2, wall=False), 'RIGHT', 'RETRACE')[0], 'RETRACE')
        choice, note = gate.choose(obs(15, history=['RIGHT','RETRACE','RIGHT','RETRACE']), 'RIGHT', 'BRANCH')
        self.assertEqual(choice, 'BRANCH')
        self.assertTrue(note['escape'])
        self.assertEqual(note['pose_repeats'], 2)

    def test_does_not_escape_only_from_history_or_only_from_repeated_pose(self):
        gate = LoopAwareGate()
        for t in range(2):
            gate.choose(obs(t), 'RIGHT', 'BRANCH')
        self.assertEqual(gate.choose(obs(2, history=['RIGHT','RIGHT','RIGHT','RIGHT']), 'RIGHT', 'BRANCH')[0], 'RIGHT')
        gate2 = LoopAwareGate()
        self.assertEqual(gate2.choose(obs(5, history=['RIGHT','RETRACE','RIGHT','RETRACE']), 'RIGHT', 'BRANCH')[0], 'RIGHT')

    def test_open_area_nonwall_and_stop_remain_unchanged(self):
        gate = LoopAwareGate()
        self.assertEqual(gate.choose(obs(0,wall=False),'RIGHT','BRANCH')[0],'BRANCH')
        self.assertEqual(gate.choose(obs(1),'STOP','BRANCH')[0],'BRANCH')
        self.assertEqual(gate.choose(obs(2),'RIGHT','RIGHT')[0],'RIGHT')

    def test_rejects_illegal_adapted_action(self):
        with self.assertRaises(ValueError):
            LoopAwareGate().choose(obs(0), 'RIGHT', 'FABRICATED')
        with self.assertRaises(ValueError):
            LoopAwareGate().choose(obs(0, heading=8), 'RIGHT', 'BRANCH')


if __name__ == '__main__':
    unittest.main()
