#!/usr/bin/env python3
"""QL-018: seed-held-out, score-level low-bit BRANCH calibration audit.

Uses immutable QL-017 W0 first-pass Actions evidence; never substitutes an
unplayed counterfactual for a real trajectory. No CyberSand code modifications.
"""
import argparse
import hashlib
import json
import math
import resource
import struct
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ACTIONS = ('STRAIGHT', 'LEFT', 'RIGHT', 'RETRACE', 'BRANCH', 'STOP')
B, R, S = (ACTIONS.index(x) for x in ('BRANCH', 'RETRACE', 'STOP'))
TRAIN_SEEDS = (10611, 10623, 10647, 10651)
TEST_SEEDS = (10667, 10679)
POLICIES = ('recovery-q1-branch12', 'recovery-q1-branch08')


def verify_and_load(path):
    with zipfile.ZipFile(path) as z:
        manifest = json.loads(z.read('MANIFEST.json'))
        for item in manifest:
            b = z.read(item['path'])
            assert len(b) == item['bytes']
            assert hashlib.sha256(b).hexdigest() == item['sha256'], item['path']
        report = json.loads(z.read('results.json'))
        assert report['schema'] == 'ql017-separated-recovery-branch-v1'
        raw = json.loads(z.read('raw_decisions.json'))
        assert len(raw) == report['first_pass_trajectories']
        assert len(manifest) == 336
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return raw, report, digest, len(manifest)


def observations(raw):
    entries = []
    legality_errors = 0
    for ep in raw:
        assert ep['first_pass'] is True
        seen = set()
        past = []
        for step in ep['steps']:
            assert step['tick'] == len(past)
            legal = [a in step['legal'] for a in ACTIONS]
            assert all(a in ACTIONS for a in step['legal'])
            if step['chosen'] not in step['legal']:
                legality_errors += 1
            s = np.asarray(step['scores'], dtype=np.float64)
            assert s.shape == (6,) and np.isfinite(s).all()
            assert ACTIONS[int(np.argmax(np.where(legal, s, -1e9)))] == step['masked_top']
            xy = tuple(step['position'])
            seen.add(xy)
            repeats = 1 - len(seen) / (len(past) + 1)
            recent = past[-8:]
            # All features depend only on the current observable trajectory prefix.
            features = [1., min(step['tick'] / 96., 1.), repeats,
                        recent.count('BRANCH') / 8., recent.count('RETRACE') / 8.,
                        float(legal[R])]
            entries.append(dict(policy=ep['policy'], seed=ep['seed'], family=ep['family'],
                                rate=ep['override_rate'], tick=step['tick'], xy=xy,
                                feature=features, scores=s.tolist(), legal=legal,
                                chosen=step['chosen'], override=step['host_override']))
            past.append(step['chosen'])
    assert legality_errors == 0
    return entries


def samples(rows, policy, seeds, nonoverride=True):
    selected = [a for a in rows if a['policy'] == policy and a['seed'] in seeds
                and a['legal'][B] and (not nonoverride or not a['override'])]
    assert selected, (policy, seeds)
    x = torch.tensor([r['feature'] for r in selected], dtype=torch.float32)
    logits = torch.tensor([r['scores'] for r in selected], dtype=torch.float32)
    mask = torch.tensor([r['legal'] for r in selected], dtype=torch.bool)
    y = torch.tensor([r['chosen'] == 'BRANCH' for r in selected], dtype=torch.float32)
    return selected, x, logits, mask, y


def fake_quant(w, bits):
    if bits == 32:
        return w
    if bits == 1:
        q = torch.where(w >= 0, 1., -1.) * w.detach().abs().mean().clamp_min(.0001)
    else:
        limit = 2 ** (bits - 1) - 1
        scale = w.detach().abs().max().clamp_min(.0001) / limit
        q = torch.round(w / scale).clamp(-limit, limit) * scale
    return w + (q - w).detach()  # straight-through QAT


def train_teacher(rows):
    _, x, _, _, y = samples(rows, 'original-q1-q4middle', TRAIN_SEEDS)
    torch.manual_seed(18018)
    w = torch.nn.Parameter(torch.zeros(x.shape[1]))
    optim = torch.optim.AdamW([w], lr=.04, weight_decay=0.)
    for _ in range(250):
        loss = F.binary_cross_entropy_with_logits(x @ w, y) + .002 * w.square().mean()
        optim.zero_grad(); loss.backward(); optim.step()
    return w.detach(), float(loss.detach()), len(y)


def branch_scores(logits, legal, x, w):
    scores = logits.masked_fill(~legal, -1e9).clone()
    scores[:, B] += x @ w[:x.shape[1]]
    return scores


def train_calibration(rows, teacher, policy, bits, constant=False):
    _, x, scores, legal, _ = samples(rows, policy, TRAIN_SEEDS)
    if constant:
        x = x[:, :1]
    with torch.no_grad():
        goal = torch.sigmoid(samples(rows, policy, TRAIN_SEEDS)[1] @ teacher)
    torch.manual_seed(18018 + bits + int(constant) * 100)
    w = torch.nn.Parameter(torch.zeros(x.shape[1]))
    opt = torch.optim.AdamW([w], lr=.07, weight_decay=0.)
    trace = []
    for step in range(200):
        q = fake_quant(w, bits)
        adjusted = branch_scores(scores, legal, x, q)
        old = scores.masked_fill(~legal, -1e9).softmax(1)
        p = adjusted.softmax(1)
        # No STOP/retrace target replacement: only a BRANCH logit correction.
        loss = ((p[:, B] - goal) ** 2).mean() + .001 * q.square().mean() + \
               .07 * ((p[:, S] - old[:, S]) ** 2).mean() + \
               .03 * ((p[:, R] - old[:, R]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 40 == 0:
            trace.append({'update':step+1, 'loss':round(float(loss.detach()), 7)})
    with torch.no_grad():
        quant = fake_quant(w, bits).detach()
    assert torch.isfinite(quant).all()
    return quant, trace


def evaluate(rows, teacher, policy, bits, weights):
    recs, x, s, legal, _ = samples(rows, policy, TEST_SEEDS)
    xx = x[:, :len(weights)]
    base = s.masked_fill(~legal, -1e9)
    new = branch_scores(s, legal, xx, weights)
    before, after = base.argmax(1), new.argmax(1)
    p0, p1 = base.softmax(1), new.softmax(1)
    target = torch.sigmoid(x @ teacher)
    visits = x[:, 2] > .25
    out = {'n':len(recs), 'bits':bits,
           'packed_weight_bytes_estimate': math.ceil(len(weights) * bits / 8) + (0 if bits == 32 else 4),
           'orig_branch_greedy': int((before==B).sum()),
           'shadow_branch_greedy': int((after==B).sum()),
           'orig_retrace_greedy': int((before==R).sum()),
           'shadow_retrace_greedy': int((after==R).sum()),
           'orig_stop_greedy':int((before==S).sum()),
           'shadow_stop_greedy':int((after==S).sum()),
           'decision_flips':int((before!=after).sum()),
           'mean_p_branch_before':float(p0[:,B].mean()),
           'mean_p_branch_shadow':float(p1[:,B].mean()),
           'teacher_mean_p_branch':float(target.mean()),
           'branch_brier_before':float(((p0[:,B]-target)**2).mean()),
           'branch_brier_shadow':float(((p1[:,B]-target)**2).mean()),
           'repeat_context_n':int(visits.sum()),
           'repeat_context_branch_before':int(((before==B)&visits).sum()),
           'repeat_context_branch_shadow':int(((after==B)&visits).sum()),
           'legal_shadow':int(legal.gather(1, after[:,None]).sum()),
           'weights':[float(v) for v in weights]}
    for key in ('mean_p_branch_before','mean_p_branch_shadow',
                'teacher_mean_p_branch','branch_brier_before','branch_brier_shadow'):
        out[key]=round(out[key],7)
    return out


def pack_exact(weights, bits):
    """Write actual compact residual codes; scales are little-endian FP32."""
    arr = np.asarray(weights, dtype=np.float32)
    if bits == 32:
        payload = arr.astype('<f4').tobytes()
        restored = np.frombuffer(payload, dtype='<f4').copy()
    else:
        limit = 1 if bits == 1 else 2 ** (bits - 1) - 1
        scale = max(float(np.max(np.abs(arr))) / limit, 0.0001)
        code = np.sign(arr).astype(np.int32) if bits == 1 else np.rint(arr / scale).astype(np.int32)
        code = np.clip(code, -limit, limit)
        unsigned = (code > 0).astype(np.uint8) if bits == 1 else (code + limit).astype(np.uint8)
        values = bytearray(math.ceil(len(arr) * bits / 8))
        for i, v in enumerate(unsigned):
            offset = i * bits
            values[offset // 8] |= int(v) << (offset % 8)
            if offset % 8 + bits > 8:
                values[offset // 8 + 1] |= int(v) >> (8 - offset % 8)
        payload = bytes(values) + struct.pack('<f', scale)
        decoded = []
        for i in range(len(arr)):
            offset = i * bits
            v = payload[offset // 8] >> (offset % 8)
            if offset % 8 + bits > 8:
                v |= payload[offset // 8 + 1] << (8 - offset % 8)
            v &= (1 << bits) - 1
            decoded.append((1 if v else -1) * scale if bits == 1 else (v - limit) * scale)
        restored = np.asarray(decoded, dtype=np.float32)
    assert np.allclose(restored, arr, atol=2e-6, rtol=2e-6)
    return payload


def latency_samples(rows, policy, weights):
    _, x, scores, legal, _ = samples(rows, policy, TEST_SEEDS)
    x = x[:, :len(weights)]
    # Same process, single-thread CPU, same stored logit input; reference Python/Torch only.
    elapsed=[]
    with torch.no_grad():
        for i in range(min(120,len(x))):
            j=i % len(x)
            t=time.perf_counter_ns()
            branch_scores(scores[j:j+1],legal[j:j+1],x[j:j+1],weights).argmax(1)
            elapsed.append(time.perf_counter_ns()-t)
    return round(float(np.median(elapsed))/1e6,6)


def run(archive, out):
    torch.set_num_threads(1)
    raw, old, sha, verified = verify_and_load(archive)
    rows=observations(raw)
    teacher, teacher_loss, nt=train_teacher(rows)
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    models_dir=out.parent/'ql018-models';models_dir.mkdir(exist_ok=True)
    report={'schema':'ql018-archived-score-shadow-calibration-v1',
            'input_ql017_archive_sha256':sha,
            'ql017_source_sha':old['source_sha'],
            'ql017_parent_q1_sha256':old['parent_q1_sha256'],
            'verified_manifest_items':verified,
            'verified_legal_logged_decisions':len(rows),
            'train_seeds':TRAIN_SEEDS,'test_seeds':TEST_SEEDS,
            'teacher_updates':250,'teacher_n':nt,
            'teacher_final_loss':teacher_loss,
            'teacher_weights':teacher.tolist(),'results':{},
            'limitations':[
               'Strictly shadow re-scoring already visited W0 states: not a counterfactual on-policy rollout.',
               'Statistical original-parent behaviour teacher is NOT an oracle for other policies on their different states.',
               'Raw original trajectories and novelty/retrace personality remain unchanged.',
               'Q1/Q2/Q4 refer to small learned correction weights, not GGUF IQ codecs or accelerated kernels.',
               'Held-out seeds are distinct, but only two seed clusters; no significance or native generalization claim.',
               'Only legal original recorded action menus used; STOP/RETRACE logits are not directly modified.',
               'No new runtime/workbench/production provider or external GPU spend.'
            ]}
    for policy in POLICIES:
        report['results'][policy] = {}
        for name,bits,constant in [('bias-fp32',32,True),('context-fp32',32,False),
                                   ('context-q4',4,False),('context-q2',2,False),
                                   ('context-q1',1,False)]:
            w, trace=train_calibration(rows,teacher,policy,bits,constant)
            metrics=evaluate(rows,teacher,policy,bits,w)
            packed=pack_exact(w.numpy(),bits)
            packed_path=models_dir / (policy+'-'+name+'.bin')
            packed_path.write_bytes(packed)
            metrics['actual_packed_bytes']=len(packed)
            metrics['actual_packed_sha256']=hashlib.sha256(packed).hexdigest()
            metrics['reference_median_ms_per_decision']=latency_samples(rows,policy,w)
            metrics['optimizer_updates']=200
            metrics['train_trace']=trace
            report['results'][policy][name]=metrics
    report['rss_peak_kib_shared_process']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':'QL018_COMPLETE','verified':verified,
        'decisions':len(rows),'teacher_n':nt,
        'test':{p:{k:(v['orig_branch_greedy'],v['shadow_branch_greedy'],
                        v['branch_brier_before'],v['branch_brier_shadow'])
                    for k,v in group.items()} for p,group in report['results'].items()}},indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--ql017',required=True,help='Unmodified QL-017 evidence ZIP')
    ap.add_argument('--out',default='ql018-results.json')
    a=ap.parse_args();run(a.ql017,a.out)
