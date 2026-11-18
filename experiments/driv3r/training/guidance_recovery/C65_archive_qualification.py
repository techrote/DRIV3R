"""Independent C65 archive/trajectory integrity verifier, standard library only.

Usage: python C65_archive_qualification.py path/to/DRIV3R_C65_*.zip [rerun_directory]
The second argument verifies independently executed source-matched C65 rerun output.
This verifies synthetic-host evidence only; it does NOT establish native CyberSand ABI compliance.
"""
import base64
import hashlib
import json
import sys
import zipfile
import zlib
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def evidence(encoded):
    return json.loads(zlib.decompress(base64.b64decode(encoded)))


def validate(summary, ledger):
    assert ledger['schema'] == 'driv3r-c65-executed-ledger/1'
    rows = ledger['records']
    assert len(rows) == summary['matched_condition_rows'] == 72
    assert summary['host_executed_suffixes'] == 432
    assert summary['training_actions_each'] == 4184
    assert summary['optimizer']['ALIGNED']['shared_training_actions'] == 4184
    assert summary['optimizer']['SHIFT']['shared_training_actions'] == 4184
    assert summary['optimizer']['ALIGNED']['updates'] == 48
    assert summary['optimizer']['SHIFT']['updates'] == 48
    assert summary['replayed_traces'] == len(ledger['traces']) == 36
    assert len(rows) * len(rows[0]['outcomes']) == 432
    assert len({r['graph'] for r in rows}) == 12
    guided = {d: {} for d in (0., 0.5)}
    no_guidance = 0
    for r in rows:
        vals = r['outcomes']
        assert len(vals) == 6
        assert len({v['protected_sha'] for v in vals.values()}) == 1
        assert all(v['host_rejections'] == 0 for v in vals.values())
        if r['arm'][0] == 'none':
            no_guidance += 1
            assert len({(v['action_sha'], v['ink_sha']) for v in vals.values()}) == 1
        else:
            counts = guided[r['drop']]
            for policy, v in vals.items():
                counts[policy] = counts.get(policy, 0) + int(v['goal'])
    assert no_guidance == 24
    for drop, counts in guided.items():
        expected = summary['aggregate'][str(drop)]['policies']
        assert all(counts[k] == expected[k]['goals'] for k in counts)
        assert [counts[k] for k in ('C34-RESET','C60-FRONTIER','C60-FROZEN','C65-ALIGNED','C65-SHIFT')] == [10,12,12,10,10]
    for t in ledger['traces']:
        row = next(r for r in rows if (r['graph'],r['drop'],r['arm']) == (t['graph'],0.,['backtrack',1.]))
        v = row['outcomes'][t['policy']]
        assert (v['action_sha'],v['ink_sha'],v['goal']) == (t['action_sha'],t['ink_sha'],t['goal'])
    return {"rows":len(rows),"suffixes":432,"unguided_parity":no_guidance,
            "trace_matches":len(ledger['traces']),"guided_goals":guided}


def main(archive, rerun=None):
    with zipfile.ZipFile(archive) as z:
        m=json.loads(z.read('MANIFEST.sha256.json'))
        assert all(digest(z.read(path)) == sha for path,sha in m.items())
        s=json.loads(z.read('publication/C65-summary.json'))
        e=evidence(z.read('publication/C65-executed-ledger.zlib.b64'))
        for p,h in s['checkpoint_sha256'].items():
            assert digest(z.read('checkpoints/'+p+'.safetensors')) == h
        assert digest(z.read('source/c65_paired_delayed_credit.py')) == s['source_sha256']['c65_paired_delayed_credit.py']
    r=validate(s,e)
    if rerun:
        d=Path(rerun)
        rs=json.loads((d/'C65-summary.json').read_bytes())
        re=evidence((d/'C65-executed-ledger.zlib.b64').read_bytes())
        validate(rs,re)
        assert len(re['records']) == len(e['records'])
        assert all(a==b for a,b in zip(e['records'],re['records']))
        assert e['traces'] == re['traces']
        assert rs['aggregate'] == s['aggregate']
        r['rerun_exact_executed_rows']=len(e['records'])
        r['rerun_exact_trace_rows']=len(e['traces'])
        r['checkpoint_bytes_identical']={p:rs['checkpoint_sha256'][p]==s['checkpoint_sha256'][p] for p in s['checkpoint_sha256']}
    print(json.dumps(r,sort_keys=True,indent=2))


if __name__ == '__main__':
    if len(sys.argv) not in (2,3):
        sys.exit('usage: python C65_archive_qualification.py C65.zip [rerun_dir]')
    main(sys.argv[1],sys.argv[2] if len(sys.argv)==3 else None)
