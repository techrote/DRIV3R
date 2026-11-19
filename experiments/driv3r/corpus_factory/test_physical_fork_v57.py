#!/usr/bin/env python3
"""Independent, no model-training v57 fork-output/data contract auditor.

Host fork execution is performed by the JavaScript generator against the pinned
v53 host modules. This validator independently checks lineage, source labels,
branch outcomes, per-pair properties, output hashes, and all split statistics.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

SPLITS = ('TRAIN', 'DEV', 'HELDOUT')


def check(x, message):
    if not x:
        raise AssertionError('V57 independent audit: ' + message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def rows(path):
    return [json.loads(x) for x in path.read_text(encoding='utf8').splitlines()]


def audit(source, output, modules):
    source, output, modules = map(Path, (source, output, modules))
    v56 = json.loads((source / 'manifest.json').read_text())
    manifest = json.loads((output / 'manifest.json').read_text())
    check(manifest['schema'] == 'driv3r/corpus_factory/physical-fork-v57/1', 'schema')
    check(manifest['synthetic_not_native_CyberSand'] and manifest['data_only'], 'safety labels')
    check(manifest['parent_v56_manifest_sha256'] == digest((source / 'manifest.json').read_bytes()), 'parent hash')
    check(manifest['generator_sha256'] == digest((modules / 'physical_fork_v57.js').read_bytes()), 'generator hash')
    for f, sha in manifest['host_module_sha256'].items():
        check(digest((modules / f).read_bytes()) == sha, 'host file hash ' + f)
    check(manifest['parent_v56_shard_sha256'] == {f: d['sha256'] for f, d in v56['files'].items()}, 'parent shards manifest')
    for f, d in v56['files'].items():
        b = (source / f).read_bytes()
        check(digest(b) == d['sha256'] and len(b) == d['bytes'] and len(b.splitlines()) == d['rows'], 'v56 parent file ' + f)
    counts = Counter()
    geo_owners = {}
    for split in SPLITS:
        prov = rows(source / f'{split}_provenance_EVALUATOR_ONLY.jsonl')
        episodes = rows(source / (f'{split}_causal_stream.jsonl' if split == 'TRAIN' else f'{split}_EVALUATOR_ONLY_causal_stream.jsonl'))
        older = rows(source / f'{split}_material_pairs_EVALUATOR_ONLY.jsonl')
        shard = f'{split}_host_forks_EVALUATOR_ONLY.jsonl'
        cert = rows(output / shard)
        desc = manifest['files'][shard]
        b = (output / shard).read_bytes()
        check(digest(b) == desc['sha256'] and len(b) == desc['bytes'] and len(cert) == desc['rows'], 'output file ' + shard)
        check(len(cert) == len(older), 'lost pair ' + split)
        by_id = {p['pair_id']: p for p in older}
        check(len(by_id) == len(older), 'duplicate parent pair id')
        counts_split = Counter()
        for row in cert:
            check(row['pair_id'] in by_id and row['split'] == split, 'unexpected pair/split')
            orig = by_id[row['pair_id']]
            check(row['same_instant_input_sha256'] == orig['instant_observation_sha256'], 'instant hash')
            check(row['identical_past_input_action_suffix_steps'] == orig['identical_past_input_action_suffix_steps'], 'suffix')
            check(len(row['counterfactuals']) == 2, 'pair missing fork')
            observations = []
            for q, ref in enumerate(row['counterfactuals']):
                oldref = orig['references'][q]
                source_id = oldref['v56_causal_episode']
                idx = oldref['source_step']
                p = prov[source_id]
                step = episodes[source_id]['steps'][idx]
                geometry = p['geometry_sha256']
                check(geometry not in geo_owners or geo_owners[geometry] == split, 'cross-split geometry leak')
                geo_owners[geometry] = split
                check(ref['world_index'] == p['world_index'] and ref['arm'] == p['arm'], 'scenario provenance')
                check(ref['episode_index'] == source_id and ref['source_step'] == idx, 'episode-step provenance')
                check(ref['input_sha256'] == orig['instant_observation_sha256'], 'input hash')
                check(ref['calibration_memory_age_steps'] == oldref['calibration_memory_age_steps'], 'memory age')
                check(ref['action'] == step['action'] == orig['actions'][q], 'source action')
                check(ref['correct_result'] == oldref['surface'] == step['input']['surface'], 'correct result not physical target')
                check(ref['wrong_result'] != oldref['surface'], 'wrong result matches target')
                check({ref['action'], ref['wrong_action']} == {'PAINT_A', 'PAINT_B'}, 'not opposite legal actions')
                check(ref['wrong_action'] in step['input']['legal'], 'wrong action illegal')
                check(ref['correct_post']['stage'] > ref['pre']['stage'], 'correct action did not advance')
                check(ref['wrong_post']['stage'] == ref['pre']['stage'], 'wrong action advanced')
                check(ref['wrong_post']['accepted'] == ref['pre']['accepted'] + 1, 'wrong action not accepted')
                check(not ref['wrong_post']['success'], 'wrong action successful')
                check(ref['wrong_caused_terminal_failure'] != ref['wrong_caused_nonterminal_bad_paint'], 'bad fork classification')
                if ref['wrong_caused_terminal_failure']:
                    check(ref['wrong_post']['done'] and not ref['oracle_recovery_events'] and not ref['oracle_recovery_terminal_success'], 'terminal fork recovery')
                    counts_split['wrong_terminal_failures'] += 1
                else:
                    check(not ref['wrong_post']['done'] and ref['oracle_recovery_terminal_success'], 'recoverable fork not recovered')
                    actions = ref['oracle_recovery_actions']
                    check(actions[:1] == ['ERASE'] and actions[-1] == 'STOP', 'invalid oracle recovery sequence')
                    check(ref['oracle_recovery_events'][-1]['terminal'] and ref['oracle_recovery_events'][-1]['success'], 'no recovered terminal')
                    check(len(actions) <= 8 and len(actions) >= 2, 'recovery length')
                    counts_split['wrong_nonterminal_bad_paints'] += 1
                    counts_split['wrong_nonterminal_oracle_recoveries'] += 1
                counts_split['host_confirmed_correct_mission_advances'] += 1
                counts_split['correct_action_forks'] += 1
                counts_split['wrong_action_forks'] += 1
                observations.append(step['input'])
            check(observations[0] == observations[1], 'not identical current observations')
            check(row['counterfactuals'][0]['action'] != row['counterfactuals'][1]['action'], 'same causal action')
            check(row['counterfactuals'][0]['correct_result'] == row['counterfactuals'][1]['correct_result'], 'different target surface')
            counts_split['paired_contrasts'] += 1
            counts_split['equal_past_32step_pairs'] += int(row['identical_past_input_action_suffix_steps'] >= 32)
        got = manifest['splits'][split]
        check(set(got) == set(counts_split) | {'host_confirmed_wrong_mission_advances'}, 'stat keys')
        check(got['host_confirmed_wrong_mission_advances'] == 0, 'wrong action mission progress')
        for k, v in counts_split.items():
            check(got[k] == v, f'stat {split}/{k}')
        counts.update(counts_split)
    return {'audit': 'PASS', 'scope': 'synthetic host evidence, NOT neural or native CyberSand',
            'host_confirmed': dict(counts), 'cross_split_geometry_owned': len(geo_owners)}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--modules', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(audit(a.source, a.output, a.modules), indent=2))
