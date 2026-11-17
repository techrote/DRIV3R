"""Focused source-faithful synthetic sensor ablation tests (no native claims)."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import pytest
import numpy as np

from . import sensing_probe_lab as lab, menu_alias_lab as alias, goal_recovery_lab as host
from .world import specs


def test_equal_host_menus_and_sensor_range_isolation():
    for sp in specs('train', 1):
        cases, features = alias.matched(sp, 711, 0, 0)
        far, open_ = cases['far_trap'],cases['open']
        assert np.array_equal(features['far_trap'],features['open'])
        assert far['world'].menu(12) == open_['world'].menu(12)
        assert lab.ray(far['world'], far['forward'], 0) == (0.,0.)
        assert lab.ray(far['world'], far['forward'], 3) == lab.ray(open_['world'], open_['forward'], 3)
        assert lab.ray(far['world'], far['forward'], 7) != lab.ray(open_['world'], open_['forward'], 7)
        for radius in (0,3,7):
            fmenu, fx = lab.matrix(far['world'],far['goal'],128,radius)
            omenu, ox = lab.matrix(open_['world'],open_['goal'],128,radius)
            assert fmenu == omenu
            assert fx.shape == ox.shape == (7,host.WIDTH+lab.EXTRA)
            assert np.array_equal(fx[:,:host.WIDTH],ox[:,:host.WIDTH])
            assert np.array_equal(fx,ox) == (radius!=7)
            assert np.all(np.isfinite(fx))
    with pytest.raises(ValueError):lab.ray(cases['open']['world'],cases['open']['forward'],99)


def test_train_only_split_and_unchanged_menu():
    inputs,masks,y,ledger,meta=lab.train_cases(711,36,natural_limit=16)
    assert ledger and all(r['split']=='train' for r in ledger)
    assert meta['natural_sensor_missing_zero_padded'] is True
    assert all(lab.verify_training_row(r) for r in ledger)
    assert all(x.shape==(len(y),7,host.WIDTH+2) for x in inputs.values())
    assert np.array_equal(masks[np.arange(len(y)),y], np.ones(len(y),bool))
    assert np.array_equal(inputs['blind'][:,:,-2:],0*inputs['blind'][:,:,-2:])
    for r in ledger:
        with pytest.raises(ValueError):
            bad=copy.deepcopy(r);bad['split']='heldout';lab.verify_training_row(bad)


def test_matched_architecture_and_restore(tmp_path):
    base=host.Ranker(711)
    arms=[lab.SensorRanker.from_base(base) for _ in range(3)]
    assert len({sum(v.size for v in m.p.values()) for m in arms})==1
    for m in arms:
        assert np.array_equal(m.p['w1'][:host.WIDTH],base.p['w1'])
        assert sum(v.size for v in m.p.values()) == 1009
        fn=tmp_path/'model.npz';m.save(fn)
        restored=lab.SensorRanker.load(fn)
        assert all(np.array_equal(restored.p[k],m.p[k]) for k in m.p)


def test_quick_source_replay_and_tamper(tmp_path):
    path=tmp_path/'out'
    rep=lab.run(path,seed=711,base_updates=2,adapt_updates=3,natural_limit=16,quick=True)
    assert rep['episodes']==12 and rep['aliases']==1
    assert lab.verify(path)['episodes']==12
    first=sorted((path/'traces').glob('*.json'))[0]
    saved=first.read_text();row=json.loads(saved)
    row['rows'][0]['requested']='STOP' if row['rows'][0]['requested']!='STOP' else 'MOVE_E'
    first.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError):lab.verify(path)
    first.write_text(saved)
    assert lab.verify(path)['episodes']==12
