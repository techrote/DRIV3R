#!/usr/bin/env python3
"""DRIV3R v54: TRAIN/DEV/HELDOUT synthetic event-stream corpus from pinned v53.
DATA ONLY; no policy training. Native CyberSand NOT exercised.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

SPLITS = ('TRAIN', 'DEV', 'HELDOUT')
def sha(data): return hashlib.sha256(data).hexdigest()
def enc(x): return json.dumps(x, separators=(',', ':'), ensure_ascii=False)
def must(condition, message):
    if not condition: raise ValueError('v54: ' + message)
def load_rows(p): return [json.loads(t) for t in p.read_text().splitlines()]
def difference(current, previous):
    old = Counter(map(enc, previous)); new = Counter(map(enc, current))
    must(not (old - new), 'a historical event disappeared or changed')
    remaining = new - old; output = []
    for event in current:
        key = enc(event)
        if remaining[key]: output.append(event); remaining[key] -= 1
    must(not +remaining, 'event-delta extraction failed')
    return output

def generate(src, dest):
    src = Path(src).resolve(); dest = Path(dest).resolve()
    must(not dest.exists(), 'destination exists')
    original = json.loads((src / 'manifest.json').read_text())
    must(original['schema'] == 'driv3r/corpus_factory/deferred-cross-surface-v53/manifest'
         and not original['quick'], 'requires complete v53 corpus')
    for file, entry in original['shards'].items():
        b = (src / file).read_bytes()
        must(sha(b) == entry['sha256'] and len(b) == entry['bytes'], 'v53 source hash mismatch '+file)
    dest.mkdir(parents=True)
    files, stats, ambiguous, geometry = {}, {}, {}, {}
    for split in SPLITS:
        records = load_rows(src / (split + '_policy.jsonl'))
        worlds = load_rows(src / (split + '_challenge_EVALUATOR_ONLY.jsonl'))
        must(len(records) == original['splits'][split]['policy_decisions'], 'row count mismatch')
        must(len(worlds) == original['splits'][split]['worlds'], 'world count mismatch')
        name = split + ('_stream.jsonl' if split == 'TRAIN' else '_EVALUATOR_ONLY_stream.jsonl')
        side = split + '_provenance_EVALUATOR_ONLY.jsonl'
        cursor = 0; state = defaultdict(Counter); full = {}; prefixes = {}
        totals = Counter(); hist = Counter()
        with (dest/name).open('x') as out, (dest/side).open('x') as provenance:
            for wi, world in enumerate(worlds):
                group = world['geometry_sha256']; prior_split = geometry.get(group)
                must(prior_split is None or prior_split == split, 'cross-split topology overlap')
                geometry[group] = split
                for arm, n in enumerate(world['episode_steps']):
                    episode = records[cursor:cursor+n]; cursor += n
                    must(len(episode) == n and n > 0, 'truncated source episode')
                    must(episode[-1]['action'] == 'STOP' and all(r['action'] != 'STOP' for r in episode[:-1]), 'invalid episode boundary')
                    prev = []; prefix = sha(b'v54-start'); steps = []
                    for i, row in enumerate(episode):
                        obs, action = row['input'], row['action']
                        must(action in obs['legal'], 'illegal synthetic host label')
                        must(set(obs) == {'v','p','legal','local','survey','painted','recent','recent_positions','durable','surface'}, 'unexpected source fields')
                        delta = difference(obs['durable'], prev)
                        if i == 0: must(delta == obs['durable'], 'missing initial historical receipts')
                        prev = obs['durable']
                        clean = {key: value for key,value in obs.items() if key != 'durable'}
                        clean['v'] = 'synthetic-v54-event-stream'
                        clean['recent'] = clean['recent'][-8:]
                        clean['recent_positions'] = clean['recent_positions'][-8:]
                        clean['events_delta'] = delta
                        signature = sha(enc(clean).encode()); state[signature][action] += 1
                        raw = sha(enc(obs).encode()); must(raw not in full or full[raw] == action, 'contradictory full-state label'); full[raw] = action
                        current_prefix = sha((prefix+enc(clean)).encode())
                        must(current_prefix not in prefixes or prefixes[current_prefix] == action, 'identical observable sequence requires opposite actions')
                        prefixes[current_prefix] = action
                        prefix = sha((current_prefix+action).encode())
                        steps.append({'input':clean,'action':action,'weight':row['weight']})
                        hist[action] += 1; totals['decisions'] += 1
                        totals['new_event_receipts'] += len(delta)
                        totals['no_new_event_steps'] += int(not delta)
                        totals['zero_weight_repeats'] += int(row['weight'] == 0)
                    out.write(enc({'steps':steps})+'\n')
                    provenance.write(enc({'split':split,'world_index':wi,'source_seed':world['source_seed'],
                        'geometry_sha256':group,'arm':world['arms'][arm],'episode_steps':n})+'\n')
                    totals['episodes'] += 1
        must(cursor == len(records), 'unconsumed policy decisions')
        for target in (name,side):
            b = (dest/target).read_bytes(); files[target] = {'sha256':sha(b),'bytes':len(b)}
        clashes = [v for v in state.values() if len(v)>1]
        must(clashes, 'no memory-dependent contradictions found')
        all_rows = sum(sum(v.values()) for v in state.values())
        clash_rows = sum(sum(v.values()) for v in clashes)
        ambiguous[split] = {'identical_instant_input_opposite_label_keys':len(clashes),
           'decision_rows_in_conflicting_keys':clash_rows,
           'stateless_majority_ceiling_in_conflicting_rows':round(sum(max(v.values()) for v in clashes)/clash_rows,6),
           'stateless_majority_ceiling_all_rows':round(sum(max(v.values()) for v in state.values())/all_rows,6),
           'observable_prefix_conflicts':0,'full_history_label_conflicts':0}
        stats[split] = {'worlds':len(worlds),**totals,'actions':dict(sorted(hist.items()))}
    result = {'schema':'driv3r/corpus_factory/stream-memory-v54/1','date':'2026-11-18',
      'synthetic_not_native_CyberSand':True,
      'derived_from_v53_manifest_sha256':sha((src/'manifest.json').read_bytes()),
      'derived_from_v53_generator_sha256':original['generator_sha256'],
      'source_shard_sha256':{k:v['sha256'] for k,v in original['shards'].items()},
      'contract':{'train':'TRAIN_stream.jsonl only; feed each complete ordered episode with fresh model state',
       'dev':'DEV_EVALUATOR_ONLY_stream.jsonl; evaluation-only',
       'heldout':'HELDOUT_EVALUATOR_ONLY_stream.jsonl; no optimization/tuning',
       'sidecars':'All *_provenance_EVALUATOR_ONLY.jsonl kept entirely outside model input',
       'context':'All existing legal actions unchanged. Replace replayed durable history by events_delta, only eight most recent action/position receipts.',
       'bootstrap':'First events_delta carries full synthetic prelude; later events_delta contains strictly newly observed receipts.',
       'weight':'Zero-weight TRAIN repeats preserved for sequential state progression; optimizer must ignore their supervised loss.',
       'grouping':'World topologies/arms remain in original v53 split. Do not mix related worlds across corpus versions.'},
      'stats':stats,'ambiguous_instantaneous_observations':ambiguous,'files':files,
      'limitations':['Synthetic event-stream transformation, no native CyberSand execution',
       'Preludes scripted by v53 and delivered all at first model step',
       'Recurrent memory requirement proven by conflicting instantaneous labels, not learned policy competence',
       'No LFM/Sprocket training or neural latency/RSS measurements',
       'No immutable GitHub Actions publication; export remains a local reproducible archive.']}
    (dest/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();print(json.dumps(generate(args.source,args.output)['ambiguous_instantaneous_observations'],indent=2))
