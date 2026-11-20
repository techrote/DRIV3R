#!/usr/bin/env python3
"""Cert-only, evaluator-side v76 action ambiguity controls (no neural training)."""
import collections
import hashlib
import json
from pathlib import Path
import sys



def audit_causal(rows, split, stats):
    """Recompute independent-risk matched groups and the uniform-prior control.

    A 50/50 mean is the average of the four symmetric A/B corners, because
    physical-outcome expected costs are bilinear in independent risk priors.
    Certificates and manifest aggregates are evaluator-only, NEVER features.
    """
    rates = {'TRAIN': (2, 6), 'DEV': (3, 5), 'HELDOUT': (1, 7)}[split]
    modes = ('NONE', 'A', 'B', 'BOTH')
    def cheapest(means):
        assert set(means) == set(modes) and all(isinstance(means[m], (int, float)) for m in modes)
        best = 'NONE'
        for m in modes[1:]:
            if means[m] < means[best] - 1e-9:
                best = m
        return best
    def signature(v):
        return json.dumps(v, sort_keys=True, separators=(',', ':'))
    corners = collections.defaultdict(dict)
    futures = collections.defaultdict(list)
    profiles = collections.defaultdict(set)
    selections = collections.Counter()
    worlds = {}
    for idx, c in enumerate(rows):
        assert c['split'] == split and c['episode'] == idx
        assert c['terminal_success'] is True and c['native_CyberSand'] is False
        assert c['decision_step'] >= 0 and c['critical_step'] > c['decision_step']
        assert c['selected'] == cheapest(c['policy_means'])
        assert abs(c['cost_regret_to_none'] -
                   (c['policy_means']['NONE'] - c['policy_means'][c['selected']])) < 1e-8
        if c['selected'] == 'NONE':
            assert c['decision_action'] in ('MOVE_N', 'MOVE_E', 'MOVE_S', 'MOVE_W')
        else:
            assert c['decision_action'] == 'PAINT_' + ('A' if c['selected'] == 'BOTH' else c['selected'])
        counts = c['incident_counts']
        assert set(counts) == {'A', 'B'} and counts['A'] in rates and counts['B'] in rates
        world = c['world_index']
        assert world not in worlds or worlds[world] == c['geometry_sha256']
        worlds[world] = c['geometry_sha256']
        base = (world, signature(c['calibration']), c['fault']['domain'], signature(c['passes']))
        risk = counts['A'], counts['B']
        futures[(base, risk)].append(c)
        if risk not in corners[base]:
            corners[base][risk] = c
        else:
            previous = corners[base][risk]
            assert c['selected'] == previous['selected'] and c['decision_sha256'] == previous['decision_sha256']
            assert all(abs(c['policy_means'][m] - previous['policy_means'][m]) < 1e-8 for m in modes)
        selections[c['selected']] += 1
    for arms in futures.values():
        assert len(arms) == 8
        assert len({(tuple(c['sequence']), c['fault']['flip']) for c in arms}) == 8
    a_contrasts = b_contrasts = varying = uniform_mistakes = 0
    regret = 0.0
    for base, group in corners.items():
        lo, hi = rates
        assert set(group) == {(lo, lo), (lo, hi), (hi, lo), (hi, hi)}
        chosen = {r: c['selected'] for r, c in group.items()}
        uniform = {m: sum(c['policy_means'][m] for c in group.values()) / 4 for m in modes}
        baseline = cheapest(uniform)
        varying += len(set(chosen.values())) > 1
        a_contrasts += (chosen[lo, lo] != chosen[hi, lo]) + (chosen[lo, hi] != chosen[hi, hi])
        b_contrasts += (chosen[lo, lo] != chosen[lo, hi]) + (chosen[hi, lo] != chosen[hi, hi])
        profiles[base[:3]].update(chosen.values())
        for c in group.values():
            excess = c['policy_means'][baseline] - c['policy_means'][c['selected']]
            assert excess >= -1e-8
            if excess > 1e-8:
                uniform_mistakes += 1
                regret += excess
    expected = {
        'counterfactual_expected_cost_groups': len(corners) * 4,
        'groups_with_different_optimal_probe_across_risk': varying,
        'groups_with_different_optimal_probe_across_A_risk': a_contrasts,
        'groups_with_different_optimal_probe_across_B_risk': b_contrasts,
        'groups_with_different_optimal_probe_across_cost_profiles': sum(len(s) > 1 for s in profiles.values()),
        'uniform_half_prior_mistakes': uniform_mistakes,
    }
    assert len(rows) == len(corners) * 32 == len(futures) * 8
    assert len(worlds) == len(stats['worlds'])
    assert {v['geometry_sha256'] for v in stats['worlds']} == set(worlds.values())
    assert all(stats[k] == value for k, value in expected.items()), ('manifest causal-metric tamper', split, expected)
    assert abs(stats['uniform_half_prior_regret'] - regret / (len(corners) * 4)) < 1e-8
    assert dict(selections) == {m: stats['probe_choices'][m] for m in selections}
    return set(worlds.values())


def score(root: Path):
    mf = json.loads((root / 'manifest.json').read_text())
    assert mf['schema'] == 'driv3r/corpus_factory/material-specific-risk-v76/1'
    scores = {}
    seen_geometries = set()
    for split in ('TRAIN', 'DEV', 'HELDOUT'):
        file = root / f'{split}_material_specific_risk_certificates_EVALUATOR_ONLY.jsonl'
        body = file.read_bytes()
        assert hashlib.sha256(body).hexdigest() == mf['files'][file.name]['sha256']
        current_only = collections.defaultdict(collections.Counter)
        paired = collections.defaultdict(list)
        samples = 0
        certificates = []
        for line in body.splitlines():
            cert = json.loads(line)
            certificates.append(cert)
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
        geometry = audit_causal(certificates, split, mf['splits'][split])
        assert not (seen_geometries & geometry), 'cross-split geometry leakage'
        seen_geometries.update(geometry)
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
