"""C68: audit whether synthetic memory corruption changes host-executed policy behavior.

Reads only previously committed C65/C66/C67 executed ledgers. No new training,
rollouts, Git writes or native acceptance. Family and graph identities are preserved.
Usage: python c68_memory_cue_audit.py LAB_DIRECTORY OUTPUT.json
"""
from __future__ import annotations
import argparse, base64, collections, hashlib, json, random, zlib
from pathlib import Path

LEDGERS = {
    'C65': 'C65-executed-ledger.zlib.b64',
    'C66': 'C66-executed-ledger.zlib.b64',
    'C67': 'C67-executed-ledger.zlib.b64',
}
DROPS = (0.0, 0.5)
STRENGTHS = (0.35, 1.0)

def read(path):
    raw = path.read_bytes()
    return json.loads(zlib.decompress(base64.b64decode(raw))), hashlib.sha256(raw).hexdigest()

def effect_interval(deltas, seed):
    # Graph-cluster bootstrap: one graph contributes one paired delta.
    assert deltas
    r = random.Random(seed)
    z = sorted(sum(r.choice(deltas) for _ in deltas) for _ in range(4000))
    return [z[99], z[3899]]

def records_for(name, data):
    rows = data['rows' if name == 'C67' else 'records']
    by = collections.defaultdict(dict)
    for x in rows:
        key = (float(x['drop']), None if name == 'C67' else float(x['arm'][1]))
        assert key not in by[x['graph']], (name, x['graph'], key)
        by[x['graph']][key] = x
    assert by
    return by

def outcome(name, row, policy):
    return row['policies'][policy] if name == 'C67' else row['outcomes'][policy]

def audit(root):
    cohorts = {}; hashes = {}; identities = set()
    for name, fname in LEDGERS.items():
        data, hashes[name] = read(root / fname)
        by = records_for(name, data)
        assert not identities.intersection(by), 'cross-cohort graph reuse'
        identities.update(by)
        policies = ('WEAK', 'STRONG', 'ALIGNED', 'SHUFFLED') if name == 'C67' else (
            'C18-A0', 'C34-RESET', 'C60-FRONTIER', 'C60-FROZEN')
        parity = {}
        for pol in policies:
            for strength in ([None] if name == 'C67' else STRENGTHS):
                paired = [(by[g].get((0., strength)), by[g].get((.5, strength))) for g in by]
                assert all(a is not None and b is not None for a, b in paired)
                pairs = [(outcome(name, a, pol), outcome(name, b, pol)) for a, b in paired]
                same = sum(a['action_sha'] == b['action_sha'] and a['ink_sha'] == b['ink_sha'] for a, b in pairs)
                key = pol if strength is None else f'{pol}@{strength:g}'
                parity[key] = {'same_action_and_ink_hashes': same, 'total_graphs': len(by),
                               'changed_graphs': len(by) - same,
                               'goal_changes': sum(a['goal'] != b['goal'] for a, b in pairs)}
        contrasts = {}
        for drop in DROPS:
            paired = []
            for graph, cases in sorted(by.items()):
                if name == 'C67':
                    w, s = cases[(drop, None)]['policies']['WEAK'], cases[(drop, None)]['policies']['STRONG']
                else:
                    w = cases[(drop, .35)]['outcomes']['C60-FROZEN']
                    s = cases[(drop, 1.)]['outcomes']['C60-FROZEN']
                paired.append((graph, w, s))
            delta = [s['goal'] - w['goal'] for _, w, s in paired]
            contrasts[str(drop)] = {'weak_goals': sum(w['goal'] for _, w, _ in paired),
                'strong_goals': sum(s['goal'] for _, _, s in paired),
                'strong_only_wins': sum(x == 1 for x in delta),
                'strong_only_losses': sum(x == -1 for x in delta),
                'paired_goal_delta': sum(delta), 'graph_cluster_bootstrap95_descriptive': effect_interval(delta, 680118),
                'weak_new_sites': sum(w['new_sites'] for _, w, _ in paired),
                'strong_new_sites': sum(s['new_sites'] for _, _, s in paired),
                'paired_goals_sha256': hashlib.sha256(json.dumps([[g, w['goal'], s['goal']] for g,w,s in paired],
                    separators=(',',':')).encode()).hexdigest()}
        cohorts[name] = {'graphs': len(by), 'record_rows': len(data['rows' if name == 'C67' else 'records']),
            'memory_corruption_hash_parity': parity, 'cue_contrast': contrasts}
    return {'schema': 'driv3r-c68-memory-cue-control-audit/1',
        'date': '2026-11-18', 'source_ledgers_sha256': hashes,
        'unique_graphs': len(identities), 'cohorts': cohorts,
        'interpretation_boundary': [
            'Exact action/ink hash parity means memory erasure had no observable action or paint effect for that policy and graph; it does not prove injection was skipped.',
            'C66 is the C67 cue-gate training cohort; do not pool C66 with C67 as independent held-out gate evaluation.',
            'A graph has two correlated dropout arms; bootstrap is graph-clustered and descriptive, not confirmatory.',
            'C67 records selected C60-FROZEN weak/strong counterfactual outcomes, not newly rolled-out adaptive decisions.',
            'Synthetic CyberSand-like host only; no native gameplay ABI, personality or checkpoint custody certification.',
            'No new optimizer run in C68: this is an intervention-potency and evidence-quality audit.'
        ]}

def main():
    p = argparse.ArgumentParser();p.add_argument('directory', type=Path);p.add_argument('output', type=Path)
    a = p.parse_args();result = audit(a.directory)
    a.output.write_text(json.dumps(result, sort_keys=True, indent=2)+'\n')
    print(json.dumps({'unique_graphs': result['unique_graphs'], 'cohorts': result['cohorts']},indent=2))
if __name__ == '__main__':main()
