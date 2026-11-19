#!/usr/bin/env python3
"""Cert-only, evaluator-side v76 action ambiguity controls (no neural training)."""
import collections
import hashlib
import json
from pathlib import Path
import sys


def score(root: Path):
    mf = json.loads((root / 'manifest.json').read_text())
    assert mf['schema'] == 'driv3r/corpus_factory/material-specific-risk-v76/1'
    scores = {}
    for split in ('TRAIN', 'DEV', 'HELDOUT'):
        file = root / f'{split}_material_specific_risk_certificates_EVALUATOR_ONLY.jsonl'
        body = file.read_bytes()
        assert hashlib.sha256(body).hexdigest() == mf['files'][file.name]['sha256']
        current_only = collections.defaultdict(collections.Counter)
        paired = collections.defaultdict(list)
        samples = 0
        for line in body.splitlines():
            cert = json.loads(line)
            assert cert['split'] == split
            current_only[cert['decision_sha256']][cert['decision_action']] += 1
            key = (cert['world_index'], tuple(sorted(cert['calibration'].items())),
                   cert['fault']['domain'], tuple(sorted(cert['passes'].items())),
                   tuple(cert['sequence']), cert['fault']['flip'])
            paired[key].append(cert)
            samples += 1
        assert samples == mf['splits'][split]['episodes']
        for certs in paired.values():
            assert len(certs) == 4
            assert len({(c['incident_counts']['A'], c['incident_counts']['B']) for c in certs}) == 4
            assert len({c['decision_sha256'] for c in certs}) == 1
        scores[split] = {
            'synthetic_not_native_CyberSand': True,
            'decision_samples': samples,
            'complete_material_risk_counterfactual_groups': len(paired),
            'current_input_only_action_accuracy_ceiling': sum(max(c.values()) for c in current_only.values()) / samples,
            'instantaneous_groups_with_conflicting_labels': sum(len(c) > 1 for c in current_only.values()),
            'A_specific_intervention_contrasts': mf['splits'][split]['groups_with_different_optimal_probe_across_A_risk'],
            'B_specific_intervention_contrasts': mf['splits'][split]['groups_with_different_optimal_probe_across_B_risk'],
            'uniform_half_prior_strict_mistakes': mf['splits'][split]['uniform_half_prior_mistakes'],
            'uniform_half_prior_mean_regret': mf['splits'][split]['uniform_half_prior_regret'],
        }
    return scores

if __name__ == '__main__':
    print(json.dumps(score(Path(sys.argv[1])), indent=2, sort_keys=True))
