# P17 — Closed-loop, changing-goal painting and recovery population (16 November 2026)

**Repository scope:** `techrote/DRIV3R`, existing Draft/HOLD PR #1 only. All research source and committed evidence are confined to `experiments/driv3r/training/population/`. CyberSand and Neurounner are read-only historical references. No game or production settings were changed.

## Question and intervention

After P16's 100% first-branch result with only 95.31% four-decision continuation, P17 tested whether small trained policies can complete **three changing spatial targets in a 96-action synthetic P0-style world** with three branch corridors, random obstacle fields, protected cells, four synthetic material classes, one dynamically injected obstruction, painting and parent-tree retracing. This is new closed-loop gameplay synthesis, **not** an extension of the P16 engineered order comparator, and not native Workbench acceptance. Target paths are explicitly scaffolded, and the supervised teacher is a privileged shortest-route BFS over the legal catalog.

Four independently executed CPU cohorts used seeds **672–675**. Each cohort trained **36 distinct scratch-initialized small MLP candidates**: six labelled behavioural niches × three controlled treatments × two candidate initializations. Each candidate received exactly **240 genuine Adam updates** (34,560 total). Selection-only worlds chose one of two candidates for each niche/treatment; the second checkpoint was retained. Training, selection and held-out world seeds were hashed and disjoint. Each treatment was evaluated on **192 held-out complete missions**, 8 worlds × 6 niches × 4 cohorts, without test-time teacher forcing.

Treatments: `goal_aware_curriculum` received changing goal observations and training-world debris; `goal_aware_static` received those goal observations but had debris disabled during training; `goal_blind_curriculum` received debris but goal progress, stage and goal coordinate features were removed (action direction and material observations retained). Training examples are on-policy/teacher-augmented and rewards are handcrafted proxy utilities, not a human-preference model. Evaluation worlds all contain fresh debris and withheld goal-order permutations.

## Held-out results

| Treatment | Complete 3-goal missions | Paint cells | Retraces | Early-stop fraction | Pairwise action distance |
| --- | ---: | ---: | ---: | ---: | ---: |
| Goal-aware + debris curriculum | **25.52%** | 48.21 | 24.85 | 35.94% | 0.7092 |
| Goal-aware + static training | 22.92% | 46.63 | 24.48 | 41.15% | 0.6955 |
| Goal-blind + debris curriculum | **0.00%** | 61.38 | 26.53 | 13.02% | 0.7166 |
| Privileged BFS teacher (no neural training) | **100.00%** | — | — | — | — |
| Nonstop-random (host legal) | 0.00% | — | — | — | — |
| Persistent heading baseline | 0.00% | — | — | — | — |

Cohort-level mission completion for the primary goal-aware curriculum treatment was **25.00%, 29.17%, 25.00%, 22.92%** across seeds 672–675. Static-train success was **8.33%, 20.83%, 29.17%, 33.33%** respectively. Curriculum exceeded static training on only two of four seeds. The mean advantage of **2.60 percentage points** is therefore **not convincing evidence of a robust improvement**. Goal observation, by contrast, was causally important within this controlled task: all goal-blind cohorts failed all complete missions. This does not establish generalized spatial reasoning.

The six labels each condition a separate trained neural population, but the task does not prove durable artistic personality. Goal-blind policies can exhibit large trajectory dispersion while achieving no goals, demonstrating why diversity alone is insufficient. No selected trained population approaches privileged expert performance, so **multi-stage continuation, retracing and timely stopping remain unsolved**.

## Qualification and reproducibility

Six `test_population_p17.py` guardrail tests passed locally. `p17_replay_audit.py` independently reconstructed **144 checkpoint weight files**, replayed **576 held-out missions**, checked **44,824 host-menu-legal actions**, verified **144 saved trace JSONs**, and confirmed zero illegal-action or obstacle/protected painting violations. All **444 cohort manifest entries** matched their original SHA-256 digests. `P17_RESULTS.json` contains source hashes, disjoint split provenance and numeric aggregates. `P17_GALLERY.svg` is composed from six actual model-executed synthetic traces, not AI-generated visuals.

Reproduce each cohort separately from this directory (Python 3.11+, NumPy 2.3.x):

```bash
python -m unittest test_population_p17 -v
python population_p17.py --output /tmp/p17-672 --seeds 672 --updates 240 --offspring 2
python population_p17.py --output /tmp/p17-673 --seeds 673 --updates 240 --offspring 2
python population_p17.py --output /tmp/p17-674 --seeds 674 --updates 240 --offspring 2
python population_p17.py --output /tmp/p17-675 --seeds 675 --updates 240 --offspring 2
python p17_replay_audit.py /tmp/p17-672 /tmp/p17-673 /tmp/p17-674 /tmp/p17-675 --output /tmp/p17-replay.json
```

Checkpoint byte hashes are documented but may vary by numerical environment, even when outcomes match. Each local run includes every candidate's weights, optimizer steps, selection metrics, held-out winner outcomes, role identifiers, gallery SVG/JSON and a complete `SHA256SUMS`. **These locally produced evidence packages are not GitHub Actions artifacts; do not claim GitHub exact-HEAD CI or immutable artifact retention until an owner-approved opt-in DRIV3R artifact workflow is available.** No new single-use workflow or game-side change was created. Earlier pre-correction experiments and interrupted 320-update campaigns are unqualified and are retained separately rather than blended into the qualified aggregates.

## Next discriminating experiment

Prioritize dense failure diagnosis: distinguish premature STOP, irrecoverable self-paint traps, failure to resume a branch after RETRACE, goal-memory confusion and material-choice errors. Compare fixed-horizon imitation against step-aware DAgger/reward-trained continuation; add matched observation and reward-model ablations. Introduce legal-action/mat ABI checks against current CyberSand read-only source before treating synthetic material decisions as transferable. Do not merge or promote the Draft PR.
