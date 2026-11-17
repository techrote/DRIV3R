"""No fake model test: pure preflight/budget/host-contract assertions only."""
import argparse
import importlib.util
from pathlib import Path

import run


def test_source_pin():
    assert run.MODEL == 'ibm-granite/granite-4.0-h-350m'
    assert len(run.PIN) == 40 and all(c in '0123456789abcdef' for c in run.PIN)


def test_budget_limit():
    a = argparse.Namespace(train_seeds=6,eval_seeds=2,steps=4,updates=32,
        rollout_seeds=2,horizon=12,batch=8,threads=2,max_context=256)
    run.budget(a)
    a.updates = 999999
    try: run.budget(a)
    except ValueError: pass
    else: raise AssertionError('unbounded training accepted')


def test_host_legality():
    scores = [999, 3, 2, 0, 0, 0, 1]
    assert run.host_choice(scores, ['LEFT','STOP']) == 'LEFT'
    assert run.host_choice(scores, ['STOP']) == 'STOP'
    try: run.host_choice(scores, [])
    except ValueError: pass
    else: raise AssertionError('empty legal action menu accepted')


def test_split_leak_guard():
    run.validate_splits([('a',0,400)], [('b',0,900)], [('c',0,1200)])
    for bad in (([('a',0,400)],[('b',0,400)],[('c',0,1200)]),
                ([('a',0,400)],[('b',0,900)],[('a',0,1200)])):
        try: run.validate_splits(*bad)
        except ValueError: pass
        else: raise AssertionError('leak accepted')


def test_preflight_fails_closed():
    r=run.preflight()
    assert r['revision']==run.PIN
    assert r['status'] in ('BLOCKED','CACHE_PRESENT_NOT_LOADED')
    assert 'pretrained' in ' '.join(r['missing']).lower() or r['status']=='CACHE_PRESENT_NOT_LOADED'


if __name__ == '__main__':
    for name in sorted(x for x in globals() if x.startswith('test_')):
        globals()[name]()
        print(name, 'PASS')
