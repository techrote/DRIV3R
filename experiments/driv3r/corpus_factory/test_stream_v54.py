#!/usr/bin/env python3
"""Independent v54 corpus audit: byte-level provenance, event-delta reconstruction,
no model-visible metadata, family split isolation and instantaneous contradiction proof."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stream_v54 as v54

def must(x, why):
    if not x: raise AssertionError(why)

def audit(source, exported):
    source, exported = Path(source), Path(exported)
    result = json.loads((exported / 'manifest.json').read_text())
    must(result['schema'] == 'driv3r/corpus_factory/stream-memory-v54/1', 'schema')
    orig = json.loads((source / 'manifest.json').read_text())
    must(v54.sha((source/'manifest.json').read_bytes()) == result['derived_from_v53_manifest_sha256'], 'source identity')
    for p, data in result['files'].items():
        b = (exported / p).read_bytes()
        must(v54.sha(b) == data['sha256'] and len(b) == data['bytes'], 'v54 shard integrity '+p)
    worlds_seen = {}
    for split in v54.SPLITS:
        episodes = v54.load_rows(exported / (split + ('_stream.jsonl' if split == 'TRAIN' else '_EVALUATOR_ONLY_stream.jsonl')))
        sidecars = v54.load_rows(exported / (split+'_provenance_EVALUATOR_ONLY.jsonl'))
        original = v54.load_rows(source / (split+'_policy.jsonl'))
        must(len(episodes) == len(sidecars) == result['stats'][split]['episodes'], 'episode counts')
        index = 0; clashes = defaultdict(set)
        for ep, meta in zip(episodes,sidecars):
            group = meta['geometry_sha256']
            must(group not in worlds_seen or worlds_seen[group] == split, 'cross-split world leak')
            worlds_seen[group] = split
            must(set(ep) == {'steps'}, 'episode data leaks metadata')
            must(len(ep['steps']) == meta['episode_steps'], 'episode length')
            last = []
            for step in ep['steps']:
                record = original[index]; index += 1
                host = record['input']; clean = step['input']
                must(set(step) == {'input','action','weight'}, 'policy row field leak')
                must(set(clean) == v54.INPUT_FIELDS if hasattr(v54,'INPUT_FIELDS') else
                     set(clean) == {'v','p','legal','local','survey','painted','recent','recent_positions','surface','events_delta'}, 'input field leak')
                must('durable' not in clean and 'geometry_sha256' not in clean, 'full history or topology leaked')
                must(step['action'] == record['action'] and step['weight'] == record['weight'], 'source label changed')
                delta = v54.difference(host['durable'],last)
                must(clean['events_delta'] == delta, 'receipt stream divergence')
                must(clean['recent'] == host['recent'][-8:] and clean['recent_positions'] == host['recent_positions'][-8:], 'bounded recent differs')
                must(host['legal'] == clean['legal'] and step['action'] in clean['legal'], 'illegal action')
                last = host['durable']
                key = hashlib.sha256(v54.enc(clean).encode()).hexdigest()
                clashes[key].add(step['action'])
            must(ep['steps'][-1]['action'] == 'STOP', 'missing final STOP')
        must(index == len(original), 'orphaned labels')
        must(sum(len(x)>1 for x in clashes.values()) == result['ambiguous_instantaneous_observations'][split]['identical_instant_input_opposite_label_keys'], 'ambiguous-state count')
    # Direct algorithm-level tamper checks; source-shard hash gate rejects actual tampered archives.
    must(v54.difference([[1,'a'],[1,'b']], [[1,'a']]) == [[1,'b']], 'same timestamp delta')
    try:
        v54.difference([[1,'a']], [[1,'a'],[2,'b']])
        raise AssertionError('mutated history should be rejected')
    except ValueError: pass
    return {'status':'PASS','worlds':len(worlds_seen),
            'episodes':sum(result['stats'][s]['episodes'] for s in v54.SPLITS),
            'decisions':sum(result['stats'][s]['decisions'] for s in v54.SPLITS),
            'heldout_conflicting_instant_states':result['ambiguous_instantaneous_observations']['HELDOUT']['identical_instant_input_opposite_label_keys']}

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--export',type=Path,required=True)
    a=p.parse_args();print(json.dumps(audit(a.source,a.export),indent=2))
