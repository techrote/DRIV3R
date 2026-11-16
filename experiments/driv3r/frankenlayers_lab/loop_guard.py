"""Wave-18 observation-only loop-aware alternative to the Wave-17 wall shield.

The controller accepts only host observations, the frozen model's legal raw
choice and the trained sidecar's legal choice. It never uses hidden terrain,
scenario names, future observations, native game state, or an oracle.

A loop alert demands two *previous* sightings of the identical pose within
an 8-decision window AND the alternating movement/RETRACE suffix. A flagged
loop allows the trained sidecar to override an otherwise shielded raw action.
This is a testable hypothesis, not demonstrated post-divergence gameplay.
"""
from collections import deque

ACTIONS = frozenset(('STRAIGHT', 'LEFT', 'RIGHT', 'BRANCH', 'RETRACE', 'STOP'))


class LoopAwareGate:
    def __init__(self, window=8):
        if type(window) is not int or window < 6 or window > 32:
            raise ValueError('window must be 6..32')
        self.recent = deque(maxlen=window)

    def choose(self, obs, raw, adapted):
        legal = obs['legal']
        if (not isinstance(legal, list) or not legal or len(set(legal)) != len(legal)
                or not set(legal) <= ACTIONS or raw not in legal or adapted not in legal):
            raise ValueError('Host-legal proposals required')
        cells = obs['local_5x5']
        if (not isinstance(cells, list) or len(cells) != 5 or
                any(not isinstance(row, str) or len(row) != 5 for row in cells) or
                cells[2][2] != '@'):
            raise ValueError('Malformed local observation')
        if (type(obs['x']) is not int or type(obs['y']) is not int or
                type(obs['heading']) is not int or obs['heading'] not in range(4)):
            raise ValueError('Malformed pose')
        history = obs['last_actions']
        if not isinstance(history, list) or any(a not in ACTIONS for a in history):
            raise ValueError('Malformed action history')
        pose = (obs['x'], obs['y'], obs['heading'])
        repeat_count = self.recent.count(pose)
        suffix = history[-4:]
        oscillating = (len(suffix) == 4 and suffix[0] == suffix[2] and
                       suffix[1] == suffix[3] == 'RETRACE' and
                       suffix[0] in ('STRAIGHT', 'LEFT', 'RIGHT'))
        visible_wall = any('#' in row for row in cells)
        shield_wanted = visible_wall and raw not in ('STOP', 'RETRACE') and raw != adapted
        escape = shield_wanted and repeat_count >= 2 and oscillating
        choice = adapted if escape or not shield_wanted else raw
        # Record only observed, actual positions. Do not fabricate counterfactual trajectories.
        self.recent.append(pose)
        return choice, {
            'raw': raw, 'learned': adapted, 'chosen': choice,
            'shield_candidate': bool(shield_wanted),
            'escape': bool(escape), 'pose_repeats': repeat_count,
            'alternating_retrace': bool(oscillating), 'wall_visible': visible_wall,
        }
