"""D023: source-pinned original Q8 teacher receipt audit (NO training).

Research-only. Archives must be supplied explicitly; never download weights or
relabel synthetic D007/D017 rollouts as original-model inference.
"""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

ACTIONS = ('STRAIGHT', 'LEFT', 'RIGHT', 'BRANCH', 'RETRACE', 'STOP')
D022_ZIP_SHA = '80d8ffde7d4aba32234f6c2030a6a2078c5eaac97c424ac89c54353a4f802fb2'
D016_ZIP_SHA = '238fa017dc0c3ee8da9def8397c54f59ec595d88696ee1b362c125c9a8cfa1d1'
D001_LINES_SHA = '140342c16a0b8c6c8d2c3bc6071767885b15096d5a53bbf304979f3015c86522'
D016_LINES_SHA = '8e4d2a2fa7a9f4ddfdb4a2d9e4f5108c82278a5173c9c120755c8a1300288727'


def digest(data: bytes) -> str:
    return sha256(data).hexdigest()


def check_archive(path: Path, expected: str) -> ZipFile:
    if digest(path.read_bytes()) != expected:
        raise ValueError(f'archive SHA mismatch: {path}')
    z = ZipFile(path)
    if z.testzip() is not None:
        raise ValueError('broken zip entry')
    return z


def jsonlines(data: bytes, expected: str) -> list[dict]:
    if digest(data) != expected:
        raise ValueError('teacher JSONL SHA mismatch')
    rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    if not rows:
        raise ValueError('empty teacher corpus')
    return rows


def validate_receipt(action: str, obs: dict, probs: dict[str, float]) -> None:
    legal = obs['legal']
    if len(legal) != len(set(legal)) or not legal or not set(legal) <= set(ACTIONS):
        raise ValueError('invalid legal menu')
    if action not in legal:
        raise ValueError('illegal recorded action')
    if set(probs) != set(ACTIONS) or any(not isinstance(v, (float,int)) or v < 0 or v > 1 for v in probs.values()):
        raise ValueError('invalid action probabilities')
    if abs(sum(probs.values()) - 1) > 1e-5:
        raise ValueError('unnormalised action probabilities')
    if any(probs[a] > 1e-8 for a in ACTIONS if a not in legal):
        raise ValueError('probability assigned to illegal action')


def obs_key(obs: dict) -> str:
    return digest(json.dumps(obs, sort_keys=True, separators=(',', ':')).encode())


def audit(d022: Path, d016: Path) -> dict:
    with check_archive(d022, D022_ZIP_SHA) as z:
        source = jsonlines(z.read('inputs/original_teachers.jsonl'), D001_LINES_SHA)
    for r in source:
        if r['source_kind'] != 'original_model_inference':
            raise ValueError('synthetic teacher masquerading as original')
        validate_receipt(r['action'], r['observation'], r['probabilities'])
    with check_archive(d016, D016_ZIP_SHA) as z:
        variants = jsonlines(z.read('evidence/source_parent_records.jsonl'), D016_LINES_SHA)
        inner = {digest(z.read(n)) for n in z.namelist()
                 if n.startswith('immutable_original_source_zips/') and n.endswith('.zip')}
        if len(inner) != 10:
            raise ValueError('D016 expected exactly 10 original source ZIPs')
    for r in variants:
        if r['source_sha256'] not in inner:
            raise ValueError('D016 source ZIP not verified')
        validate_receipt(r['action'], r['obs'], dict(zip(ACTIONS, r['ps'])))
    if len(source) != 144 or len(variants) != 483:
        raise ValueError('unexpected original-receipt counts')
    original_keys = {obs_key(r['observation']) for r in source}
    variant_keys = {obs_key(r['obs']) for r in variants}
    seed_counts = Counter(r['source_seed'] for r in source)
    return {
        'schema': 'driv3r.distillation.d023-teacher-provenance/1',
        'evidence_kind': 'original frozen Q8 inference receipts, NOT fresh inference or native game replay',
        'source_archives': {
            'D022_evidence_sha256': D022_ZIP_SHA,
            'D016_evidence_sha256': D016_ZIP_SHA,
            'D001_original_jsonl_sha256': D001_LINES_SHA,
            'D016_original_jsonl_sha256': D016_LINES_SHA,
            'D016_verified_inner_source_zips': len(inner),
        },
        'original_anchor': {
            'rows': len(source),
            'source_seed_counts': dict(seed_counts),
            'unique_observations': len(original_keys),
            'teacher_counts': dict(sorted(Counter(r['teacher'] for r in source).items())),
            'action_counts': dict(sorted(Counter(r['action'] for r in source).items())),
            'model_sha256': sorted({r['model_sha256'] for r in source}),
        },
        'franken_variants': {
            'rows': len(variants),
            'unique_observations': len(variant_keys),
            'variant_counts': dict(sorted(Counter(r['variant'] for r in variants).items())),
            'action_counts': dict(sorted(Counter(r['action'] for r in variants).items())),
            'source_seed_from_normalized_rows': None,
        },
        'cross_archive_unique_observations': len(original_keys | variant_keys),
        'shared_observations_between_archives': len(original_keys & variant_keys),
        'legal_recorded_actions': len(source) + len(variants),
        'illegal_recorded_actions': 0,
        'seed_independent_teacher_split_established': False,
        'training_ready': False,
        'caveats': [
            'D001 original parent rows have only source seed 570; row random splits leak world conditions.',
            'D016 normalized rows do not expose source seed; archive provenance does not establish independent worlds.',
            'Cross-parent overlapping observations require group-by-world provenance before splitting.',
            'These receipts are action samples, not complete changing-observation teacher rollouts.',
            'Synthetic D018/D021/D022 student-state labels are NOT original teacher inference.',
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--d022-zip', required=True, type=Path)
    p.add_argument('--d016-zip', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args()
    result = audit(args.d022_zip, args.d016_zip)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
