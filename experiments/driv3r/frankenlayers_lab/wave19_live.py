#!/usr/bin/env python3
"""Wave 19: bounded, manual-only, live-model paired loop-guard evaluation.

No automatic workflow. Never infer counterfactual trajectories from historic traces.
The imported parent run_arm takes FIVE arguments and reads module-global SEED;
older lab scripts passed SIX. Bind its actual API in a reversible fold context.
"""
from contextlib import contextmanager
import argparse
import json
import math
from pathlib import Path
import statistics

from experiments.driv3r.backend import PINS, sha256_file
from experiments.driv3r.frankenlayers import infer as paired
from experiments.driv3r.frankenlayers.patch_runtime import FLOWS
from experiments.driv3r.frankenlayers_lab import counterfactual as cf
from experiments.driv3r.frankenlayers_lab.flows import LAB_FLOWS, validate_flows
from experiments.driv3r.frankenlayers_lab.infer_quant import validate_quant_receipt
from experiments.driv3r.frankenlayers_lab.loop_guard import LoopAwareGate
from experiments.driv3r.frankenlayers_lab.repair_adapter import predict, train, digest

MODE = 'aero4_no2'
STEPS = 24
VARIANTS = ('east_open', 'east_blocked', 'north_blocked', 'south_blocked')
PAIRS = {'forward_gate': ('east_open', 'east_blocked'),
         'side_obstacle': ('north_blocked', 'south_blocked')}
FOLDS = ((6089, 'pipework', 2), (7127, 'greebles', 3))
TRAIN = (2718, 'architecture', 0)
ARMS = ('q8_stock', 'q4_stock', 'q4_mutant_raw', 'q4_mutant_strong',
        'q4_mutant_guarded', 'q4_mutant_loop')


def obstacles(variant, shift):
    if variant not in VARIANTS or type(shift) is not int or shift not in (0, 2, 3):
        raise ValueError('Unregistered intervention or shift')
    y = cf.SIZE // 2
    return {'east_open': set(), 'east_blocked': {(10 + shift, y)},
            'north_blocked': {(8 + shift, y - 2), (9 + shift, y - 2)},
            'south_blocked': {(8 + shift, y + 2), (9 + shift, y + 2)}}[variant]


@contextmanager
def fold_context(seed, family, shift):
    """Pin all legacy global inputs, restoring even on exceptions."""
    if (seed, family, shift) not in (TRAIN, *FOLDS):
        raise ValueError('Unregistered calibration/evaluation fold')
    old = (paired.SEED, paired.STEPS, paired.FAMILIES, cf.SEED, cf.STEPS,
           cf.FAMILY, cf.ALL_VARIANTS, cf.PAIRS, cf.variant_obstacles)
    try:
        paired.SEED, paired.STEPS, paired.FAMILIES = seed, STEPS, (family,)
        cf.SEED, cf.STEPS, cf.FAMILY = seed, STEPS, family
        cf.ALL_VARIANTS, cf.PAIRS = VARIANTS, PAIRS
        cf.variant_obstacles = lambda v, size=cf.SIZE: obstacles(v, shift) if size == cf.SIZE else (_ for _ in ()).throw(ValueError('size'))
        yield
    finally:
        (paired.SEED, paired.STEPS, paired.FAMILIES, cf.SEED, cf.STEPS,
         cf.FAMILY, cf.ALL_VARIANTS, cf.PAIRS, cf.variant_obstacles) = old


@contextmanager
def model_policy(checkpoint, kind):
    """One stateful loop detector per server/episode, never across variants."""
    if kind not in ('strong', 'guarded', 'loop'):
        raise ValueError('Unexpected policy arm')
    original = paired.Server

    class Adapted(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.gate = LoopAwareGate()

        def choose(self, obs, records):
            if obs['tick'] == 0:
                self.gate = LoopAwareGate()
            raw = super().choose(obs, records)
            rec = records[-1]
            learned, note = predict(obs, raw, rec['parsed']['probabilities'], checkpoint)
            wall = any('#' in row for row in obs['local_5x5'])
            shield = wall and raw not in ('STOP', 'RETRACE') and raw != learned
            if kind == 'strong':
                chosen = learned
                loop = None
            elif kind == 'guarded':
                chosen = raw if shield else learned
                loop = None
            else:
                chosen, loop = self.gate.choose(obs, raw, learned)
                if loop['shield_candidate'] != shield:
                    raise AssertionError('Guard provenance mismatch')
            if chosen not in obs['legal']:
                raise AssertionError('Policy emitted action outside host menu')
            rec['base_model_choice'] = raw
            rec['choice'] = chosen
            rec['adapter'] = {'base': raw, 'adapted': chosen,
                              'pre_shield_adapted': learned,
                              'overrode': chosen != raw,
                              'shielded': shield and chosen == raw,
                              'wall_visible': wall,
                              'residual': note, 'loop_guard': loop}
            return chosen

    paired.Server = Adapted
    try:
        yield
    finally:
        paired.Server = original


def run_fold(binary, model, w0, mode, root, fold):
    seed, family, shift = fold
    with fold_context(seed, family, shift):
        runs = {}
        for variant in VARIANTS:
            out = root / variant
            # Imported paired.run_arm has five parameters; seed is bound above.
            arm = paired.run_arm(binary, model, cf.CausalHost(w0, variant), mode, out)
            calls = json.loads((out / (mode + '-' + family + '-choices.json')).read_text())
            if (len(calls) != arm['model_decisions'] or any(
                    c['status'] != 'ok' or c['choice'] not in c['observation']['legal']
                    for c in calls)):
                raise ValueError('Model/host legal-action provenance mismatch')
            runs[variant] = {'arm': arm, 'calls': calls}
        diagnostics = {name: cf.pair_diagnostic(runs[left], runs[right])
                       for name, (left, right) in PAIRS.items()}
        episodes = [runs[v]['arm']['experiments'][0] for v in VARIANTS]
        n = sum(len(runs[v]['calls']) for v in VARIANTS)
        summary = {
            'decisions': n,
            'budget': len(VARIANTS) * STEPS,
            'new_cells': sum(e['metrics']['new_steps'] for e in episodes),
            'revisits': sum(e['metrics']['total_revisit_contacts'] for e in episodes),
            'early_stop_episodes': sum(bool(e['metrics']['terminated_by_stop'])
                                       and e['decisions'] < STEPS for e in episodes),
            'invalid_actions': sum(e['metrics']['invalid_raw_choices'] for e in episodes),
            'exposed_pairs': sum(x['geometry_exposure_observed'] for x in diagnostics.values()),
            'responded_pairs': sum(x['action_divergence_after_observation'] for x in diagnostics.values()),
            'unexplained_divergences': sum(x['unexplained_early_action_divergence'] for x in diagnostics.values()),
            'median_decision_seconds': statistics.median(
                runs[v]['arm']['p50_seconds'] for v in VARIANTS),
            'choice_counts': {a: sum(e['raw_choices'].count(a) for e in episodes)
                              for a in paired.ACTION_DESCRIPTIONS},
        }
        if sum(summary['choice_counts'].values()) != n:
            raise ValueError('Choice accounting mismatch')
        return {'seed': seed, 'family': family, 'geometry_shift': shift,
                'pairs': diagnostics, 'summary': summary}


def compare(folds):
    result = {}
    for arm in ARMS:
        rows = [fold[arm]['summary'] for fold in folds]
        decisions = sum(x['decisions'] for x in rows)
        budget = sum(x['budget'] for x in rows)
        choices = {a: sum(x['choice_counts'][a] for x in rows)
                   for a in paired.ACTION_DESCRIPTIONS}
        result[arm] = {'decisions': decisions, 'budget': budget,
                       'legal_fraction': (decisions-sum(x['invalid_actions'] for x in rows))/decisions if decisions else None,
                       'novel_cells': sum(x['new_cells'] for x in rows),
                       'novel_per_budget': sum(x['new_cells'] for x in rows)/budget,
                       'revisits': sum(x['revisits'] for x in rows),
                       'premature_stop_episodes': sum(x['early_stop_episodes'] for x in rows),
                       'exposed_pairs': sum(x['exposed_pairs'] for x in rows),
                       'responded_exposed_pairs': sum(x['responded_pairs'] for x in rows),
                       'unexplained_divergences': sum(x['unexplained_divergences'] for x in rows),
                       'median_fold_p50_seconds': statistics.median(x['median_decision_seconds'] for x in rows),
                       'action_entropy_bits': -sum((n/decisions)*math.log2(n/decisions)
                                                   for n in choices.values() if n and decisions),
                       'choice_counts': choices}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('binary', 'model-q8', 'model-q4', 'quant-receipt', 'w0', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--source-sha', required=True)
    a = p.parse_args()
    if a.output.exists() or len(a.source_sha) != 40 or any(c not in '0123456789abcdef' for c in a.source_sha):
        raise ValueError('Immutable evidence destination or exact source SHA missing')
    if sha256_file(a.model_q8) != PINS['model']['sha256']:
        raise ValueError('Q8 donor integrity mismatch')
    receipt = json.loads(a.quant_receipt.read_text())
    validate_quant_receipt(receipt, a.model_q8, a.model_q4)
    validate_flows()
    if any(k in FLOWS for k in LAB_FLOWS):
        raise ValueError('Historical model flow overwritten')
    FLOWS.update(LAB_FLOWS)
    w0 = paired.load_w0(a.w0)
    a.output.mkdir(parents=True)
    source = []
    run_fold(a.binary, a.model_q4, w0, MODE, a.output/'train', TRAIN)
    for variant in VARIANTS:
        src = a.output/'train'/variant/(MODE+'-'+TRAIN[1]+'-choices.json')
        for c in json.loads(src.read_text()):
            source.append({'observation': c['observation'], 'choice': c['choice'],
                           'probabilities': c['parsed']['probabilities']})
    if len(source) < 48:
        raise ValueError('Insufficient real training fold decisions')
    checkpoint = train(source)
    paired.write_json(a.output/'adapter-checkpoint.json', checkpoint)
    paired.write_json(a.output/'train-samples.json', source)
    folds = []
    for i, fold in enumerate(FOLDS):
        root = a.output/('heldout-'+str(i))
        rows = {}
        rows['q8_stock'] = run_fold(a.binary, a.model_q8, w0, 'stock', root/'q8-stock', fold)
        rows['q4_stock'] = run_fold(a.binary, a.model_q4, w0, 'stock', root/'q4-stock', fold)
        rows['q4_mutant_raw'] = run_fold(a.binary, a.model_q4, w0, MODE, root/'q4-raw', fold)
        for kind in ('strong', 'guarded', 'loop'):
            with model_policy(checkpoint, kind):
                rows['q4_mutant_'+kind] = run_fold(
                    a.binary, a.model_q4, w0, MODE, root/('q4-'+kind), fold)
        folds.append(rows)
    result = {'schema': 'frankenlayers-wave19-live-loop-ablation/1',
              'source_sha': a.source_sha, 'flow': LAB_FLOWS[MODE],
              'q8_sha256': PINS['model']['sha256'],
              'q4_sha256': sha256_file(a.model_q4),
              'w0_sha256': sha256_file(a.w0),
              'quant_receipt': receipt,
              'adapter_weights_sha256': checkpoint['weights_sha256'],
              'train_data_sha256': digest(source), 'train_fold': TRAIN,
              'evaluation_folds': FOLDS, 'steps_per_variant': STEPS,
              'variants': VARIANTS, 'pairs': PAIRS,
              'folds': folds, 'pareto_axes': compare(folds),
              'real_model_inference': True, 'external_adapter_trained': True,
              'base_model_weights_changed': False, 'native_game': False,
              'limits': 'Synthetic W0 only; no native game, materials chemistry, protected-cell reactions or goal success. '
                        'Exposure-normalised response is not proven causal reasoning; per-decision latency is not episode throughput.'}
    paired.write_json(a.output/'comparison.json', result)
    print(json.dumps(result['pareto_axes'], sort_keys=True))


if __name__ == '__main__':
    main()
