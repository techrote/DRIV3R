"""Offline on-policy-prefix shadow replay over verified original Wave-17 ZIP.

The comparison is VALID only up to the first choice that differs from the
recorded Wave-17 wall-shield trajectory. Anything after that would require new
model inference. No synthetic counterfactual gameplay performance is claimed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

try:
    from .loop_guard import LoopAwareGate
    from .audit_wave17_evidence import verify_zip
except ImportError:
    from loop_guard import LoopAwareGate
    from audit_wave17_evidence import verify_zip

EXPECTED_SHA = 'dcaf7539d39c7b1a28b4f1be00ad427e71cffe8a6378500b928d123c914db0d9'
VARIANTS = ('east_open', 'east_blocked', 'north_blocked', 'south_blocked', 'far_open', 'far_blocked')


def shadow_replay(path):
    original = Path(path).read_bytes()
    sha = hashlib.sha256(original).hexdigest()
    if sha != EXPECTED_SHA:
        raise ValueError('Original Wave-17 ZIP pin differs')
    with zipfile.ZipFile(path) as z:
        verified = verify_zip(z)
        report = json.loads(z.read('inference/comparison.json'))
        if report['schema'] != 'frankenlayers-wave17-causal-shield-ablation/1':
            raise ValueError('Not the pinned Wave-17 experimental contract')
        results = {}
        for variant in VARIANTS:
            name = f'inference/eval-q4-mutant-guarded/{variant}/aero4_no2-greebles-choices.json'
            calls = json.loads(z.read(name))
            gate = LoopAwareGate()
            first_difference = None
            candidate_count = 0
            for i, call in enumerate(calls):
                if call['status'] != 'ok':
                    raise ValueError('Non-successful recorded decision')
                obs = call['observation']
                if obs['tick'] != i:
                    raise ValueError('Nonconsecutive original decisions')
                p = call['adapter']
                if p['base'] != call['base_model_choice'] or p['adapted'] != call['choice']:
                    raise ValueError('Historical policy provenance mismatch')
                predicted, rationale = gate.choose(obs, p['base'], p['pre_shield_adapted'])
                candidate_count += int(rationale['shield_candidate'])
                if predicted != call['choice']:
                    first_difference = {
                        'tick': i, 'pose': [obs['x'], obs['y'], obs['heading']],
                        'historical': call['choice'], 'shadow': predicted,
                        'raw': p['base'], 'learned': p['pre_shield_adapted'],
                        'repeats': rationale['pose_repeats'],
                        'oscillating': rationale['alternating_retrace'],
                    }
                    break  # No claims after off-policy divergence.
            results[variant] = {
                'historical_decisions': len(calls),
                'validated_on_policy_prefix': i + 1 if first_difference else len(calls),
                'wall_shield_candidates_in_prefix': candidate_count,
                'first_shadow_divergence': first_difference,
            }
        return {
            'schema': 'frankenlayers-wave18-original-trace-shadow/1',
            'original_archive_sha256': sha, 'verified_payloads': verified,
            'source_run': 43947574492,
            'basis': 'original Wave-17 guarded observations and real-model sidecar proposals',
            'results': results, 'new_model_inference': False,
            'counterfactual_trajectory_measured': False,
            'limitations': 'Shadow replay stops at first changed action; later positions, legality, novelty, survival and geometry response are unknown until a full inference rerun.',
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('wave17_zip', type=Path)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    result = shadow_replay(a.wave17_zip)
    text = json.dumps(result, sort_keys=True, indent=2) + '\n'
    if a.output:
        if a.output.exists():
            raise ValueError('Refusing to overwrite immutable evidence')
        a.output.write_text(text)
    print(text)


if __name__ == '__main__':
    main()
