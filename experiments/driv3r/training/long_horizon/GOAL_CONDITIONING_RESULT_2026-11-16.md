# GOAL-COND-001 — explicit spatial-goal control versus goal-blind imitation

**16 November 2026 · DRIV3R Draft/HOLD PR #1 · model-side synthetic DEVELOPMENT only.**

## Falsifiable claim and boundary

Does an autonomous 769-parameter policy receiving an explicit, host-observable current
spatial goal respond meaningfully to a changed goal command over 128–160 steps,
compared with an exactly matched policy whose six goal features are zero-masked?

Two independent CPU seeds (613, 614), each arm initialized identically within
seed. Same 36 → 72 → 128 TRAIN teacher examples and labels, same 60 Adam
updates/stage = 180 genuine optimizer steps per checkpoint. No goal information
is available to the masked control. Both arms receive **exactly the same host
menu**. All action probabilities include legal STOP and recovery actions.

GOAL-COND is **not** d1/LFM/Sprocket training, not native CyberSand, not the
unmodified native 64-move ceiling, not a held-out structural qualification.
The synthetic host/terrain and source pins are `world.py`, `policy.py`,
`train.py` in this owned lane. External goals are explicit command inputs, not
A* routes or hidden environment labels. Goal-feature arithmetic includes the
one-step Manhattan improvement to the current target: this is **engineered
geometric conditioning**, not evidence that a general model discovered planning.

TRAIN = eight source-generated training worlds × two independent goal schedules;
DEVELOPMENT = four separate seeds/structures × three goal schedules × horizons
128 and 160. **Sealed archipelago/staggered_vault heldout families were never opened.**
The DEVELOPMENT families share structural classes with TRAIN; they are not a
substitute for sealed or native generalization.

## Results: 96 fully autonomous DEVELOPMENT episodes

| Seed | Arm | Goals reached (24 episodes, up to 4 each) | Mean unique ink | Early STOP | Mean turns |
| --- | --- | ---: | ---: | ---: | ---: |
| 613 | Goal-blind | 12 | 143.625 | 0/24 | 35.500 |
| 613 | Goal-conditioned | **82** | 130.833 | **5/24** | 46.875 |
| 614 | Goal-blind | 16 | 141.625 | 3/24 | 40.000 |
| 614 | Goal-conditioned | **75** | 129.792 | **6/24** | 46.250 |
| **Both** | **Goal-blind** | **28/192 potential goals** | **142.625** | **3/48** | **37.750** |
| **Both** | **Goal-conditioned** | **157/192 potential goals** | **130.313** | **11/48** | **46.563** |

*Goals reached* = a goal cell painted by the policy at least once, over the
fixed four-command list; goals can be genuinely inaccessible after painting.
Both policies are neural for **every DEVELOPMENT decision**, without a teacher
continuation after the initial state. Legal forced alternative MOVE requests at
fixed positions 27/68/107 are re-executed and independently validated.

**Mixed result:** conditioning produces a large, reproducible increase in
commanded goal-cell visitation on DEVELOPMENT (+129), but *reduces* mean unique
paint by 12.31 cells/episode and *increases* premature STOPs by 8. The
conditioned controllers chase commanded cells and turn more; they are **not
validated as better 128–160-decision survival policies**.

Counterfactual initial-state goal swap across east/west/north/south commands:
within each world the blind arm's distribution is exactly identical (maximum
L1 = 0.0); conditioned models have maximum L1 1.9963 (seed 613) and 1.9996
(seed 614), and select all four cardinal actions somewhere in four DEVELOPMENT
starting worlds. This establishes sensitivity to explicit commands, not
unbiased route competence.

## Original evidence and exact identities

Four checkpoint SHA-256 identifiers (NPZ files remain **outside Git**):

| Seed | Goal-blind | Goal-conditioned |
| --- | --- | --- |
| 613 | `f01fc7b45f4afe18bc508d3a0e0d614184fb82de5daeb426b2a40f5737515a1b` | `d5f3837a0dee2e29874db8493dcc46447e354ed4ec13111ee6ef2ca5b7240860` |
| 614 | `77cc15770a54aafa29f55bd4e364b7a9584bf696791862e3a8b5e29873140a3a` | `d1b48e7e7d4f8e6024051ed31fb57ab516b5289b714a4227579f911b6d4c2ab4` |

Original `goal_conditioning_lab.py` SHA-256:
`6e00bfdfbaecc8195a86cb626bbbcbb65184a449adc5b7272d5642553cf02659`.
Dependency SHA-256: `world.py` `6801c401c356869ed6535fec5da4a627871cb05fff9c95d6726224a492db678a`;
`policy.py` `f3e6469b592e68c0d3410dca88e107b215f1bdb1c16f7ebd73232ea9470cbb82`;
`train.py` `9ed2c9ac49fa397c12f0b8256e4d4439bd3a9294b002b54276b702a1e3ddcacb`.
These dependency Git blobs were compared against the exact DRIV3R research branch
before this experiment; local hashes were identical.

Each campaign was independently verified by reloading the checkpoints,
regenerating each host-observable goal sequence, rerunning all neural
probabilities and legal action menus, reproducing state hashes, forced
interventions, painted cells and metrics, and validating SHA-256 manifests.
**96/96 episodes, 13,373/13,373 executed decisions, 118/118 file hashes,
4/4 checkpoints passed**. Four focused tests passed, including a tampering
rejection test. All state edits were host-legal; no protected or occupied
material edits were observed. Action distributions are in each original
`report.json` and trajectory JSON. No native receipts.

| Seed | CPU runtime | Process peak RSS | Average neural decision latency (blind / conditioned) |
| --- | ---: | ---: | ---: |
| 613 | 13.606 s | 113,560 KiB | 0.1889 / 0.2070 ms |
| 614 | 11.922 s | 113,444 KiB | 0.1620 / 0.1748 ms |

Original checkpoint/trace ZIPs and contact sheet supplied separately as local
conversation artifacts; **not** yet immutable GitHub Actions artifacts.
No new workflows, routine-push training, paid inference, model downloads,
DRIV3R main changes, or CyberSand/Neurounner mutations.

## Next falsification

Hold goal conditioning fixed while training an explicit **return/recovery-aware
continuation head**; separately measure target achievement, paint continuity,
recovery cost and STOP inflation. Before any promotion, require multi-seed
blind structural held-out families, fresh native host-contract comparison,
and a matched goal-information-ablation study with identical computational
capacity. A strong goal-visit count alone is insufficient.
