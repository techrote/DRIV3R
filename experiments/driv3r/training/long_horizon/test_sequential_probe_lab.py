"""SEQUENTIAL-PROBE-001: source replay, budget, equal-menu and optimizer tests."""
import copy
import json
import numpy as np
import pytest
from . import sequential_probe_lab as lab, goal_recovery_lab as host
from .world import specs


def test_initial_alias_and_budget():
    for sp in specs('train',1):
        for turns in (0,1,2,3):
            pair=lab.matched(sp,turns)
            a=pair['trap'];b=pair['open']
            assert a['world'].menu(12)==b['world'].menu(12)
            assert a['world'].recoveries==b['world'].recoveries==0
            for q in (a,b):
                w=q['world'];s=lab.fresh_state()
                first=lab.observe(w,'sequential7',q['forward'],q['retreat'],s)
                assert first['reads']==1 and first['retreating'] is False
                for t in range(3):
                    ev=w.apply(q['forward'],12)
                    lab.advance_state(w,q['forward'],q['retreat'],s,q['forward'],tuple(ev['cursor_before']))
                    obs=lab.observe(w,'sequential7',q['forward'],q['retreat'],s)
                    assert obs['reads']==(3 if t in (0,2) else 0)
                    assert obs['spent']<=7
                assert obs['spent']==7
                assert obs['retreating'] == (q['kind']=='trap')
                assert 'BACKTRACK' in w.menu(12)
    with pytest.raises(ValueError):lab.scan(pair['open']['world'],pair['open']['forward'],8)


def test_train_data_is_split_sealed_and_replayable():
    inputs,masks,labels,rows,meta=lab.train_cases(811,36,16)
    assert set(inputs)==set(lab.ARMS)
    assert all(inputs[a].shape==(len(labels[a]),7,lab.WIDTH) for a in lab.ARMS)
    assert len({len(labels[a]) for a in lab.ARMS})==1
    assert all(np.all(masks[a][np.arange(len(labels[a])),labels[a]]) for a in lab.ARMS)
    assert meta['natural_sensor_zero_padded'] is True
    assert lab.verify_train_ledger(rows)[1]==len(rows)
    bad=copy.deepcopy(rows);bad[0]['obs']['reads']=100
    with pytest.raises(ValueError):lab.verify_train_ledger(bad)
    bad=copy.deepcopy(rows);bad[0]['spec']['split']='heldout'
    with pytest.raises(ValueError):lab.verify_train_ledger(bad)


def test_equal_capacity_and_checkpoint(tmp_path):
    base=host.Ranker(811)
    models=[lab.ProbeRanker.from_base(base) for _ in lab.ARMS]
    assert len({sum(p.size for p in m.p.values()) for m in models})==1
    assert sum(p.size for p in models[0].p.values())==1033
    for m in models:
        assert np.array_equal(m.p['w1'][:host.WIDTH],base.p['w1'])
        f=tmp_path/'weights.npz';m.save(f)
        restored=lab.ProbeRanker.load(f)
        assert all(np.array_equal(m.p[k],restored.p[k]) for k in m.p)


def test_immutable_quick_campaign_and_evidence_tamper(tmp_path):
    out=tmp_path/'real'
    report=lab.run(out,seed=811,base_updates=2,adapt_updates=3,natural_limit=16,quick=True)
    assert report['episodes']==6 and report['training_rows']==2880
    assert lab.verify(out)['episodes']==6
    with pytest.raises(FileExistsError):lab.run(out,seed=811,quick=True)
    t=next((out/'traces').glob('*.json'))
    orig=t.read_text();ep=json.loads(orig)
    ep['rows'][0]['probabilities'][next(iter(ep['rows'][0]['probabilities']))]=.001
    t.write_text(json.dumps(ep)+'\n')
    with pytest.raises(ValueError):lab.verify(out)
    t.write_text(orig)
    assert lab.verify(out)['episodes']==6
