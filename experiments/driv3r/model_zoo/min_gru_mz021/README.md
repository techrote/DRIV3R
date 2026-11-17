# MZ-021 — 20.61M minGRU recurrence, first game-policy training probe

**Status: REJECT PROMOTION, retain as a trained architectural specimen.**

This is a **scratch-initialized** (NOT pretrained) 20,612,807-parameter, four-layer minGRU-family recurrent network. The recurrence follows the equations discussed in Feng et al., *Were RNNs All We Needed?*, and the independently authored source is reference-pinned to `lucidrains/mingru-pytorch@9fe95d623b2a30f5cbc689e4640dc62403da0df5` (MIT). The model is not the published upstream checkpoint/implementation; no upstream weights were loaded.

The target is the **existing common MZ-003/Mamba130 11x11 synthetic host** and 114-dimensional structured prompt projection (`experiments.driv3r.model_zoo.mamba130.world` and `.counterfactual`), NOT the canonical CyberSand runtime or an authoritative native action/observation ABI. Offered actions: STRAIGHT, LEFT, RIGHT, BRANCH, RETRACE, PENUP, STOP. Protected cells are excluded by the host menu. In this host `PENUP` is a two-cell unpainted movement, not a true CyberSand PAINT action. This probe establishes no native painting skill or safety.

## Completed bounded local CPU trial

`python -m experiments.driv3r.model_zoo.min_gru_mz021.run --out /tmp/mz021 --updates 64 --train-seeds 24 --steps 8 --window 8 --batch 2 --threads 2 --rollout-steps 24`

- **64 actual AdamW updates across all 20.61M weights**, paired with 64 updates to a 805-parameter linear structured-feature control on the same sampled minibatches. Training seeds 400–423; DEV 900–903, TEST 1200–1203, closed-loop 1807–1810 (disjoint). Each world has repeated counterfactual target substitutions; the model learns 7-class raw logits before host masking.
- TRAIN loss 1.99982 → 1.25084; independent TEST raw action accuracy 29/124 (23.387%) versus linear control 23/124 (18.548%). TEST raw illegal 7/124 (5.645%) versus control 12/124 (9.677%). Fork both-correct 4/161 (2.484%) versus control 1/161 (0.621%). These four worlds are **weak evidence** and not a statistically robust win.
- Four held-out 24-turn closed-loop worlds: **0/4 goals reached** for minGRU, linear, random and the teacher. The teacher is not an optimal route planner on this host, so goal-reaching is a poor competence endpoint. Unique cumulative painted visitation: minGRU 30, linear 33, random non-STOP 47, teacher 84 across four worlds. No raw illegal proposals in the minGRU rollouts, but this is a small, seed-dependent sample.
- Cached full-prefix numerical parity 0.0 absolute difference on the tested 10-token prefix. CPU cached decision p50 2.706 ms / p95 5.5542 ms (local torch CPU, two threads); 10-token full-history replay p50 29.6985 ms. Peak **whole-process RSS** 769.5 MiB is not runtime-resident model RSS.
- Trained checkpoint **stayed local only** (82,474,613 bytes), SHA-256 `3d1c343fa70958ccf1353ab802e50cff086205ec356f27bb9852e2967f3557d1`. It is **not** stored in Git and **not** in an immutable Actions artifact, so this is unqualified/locally reproducible evidence rather than an independently replicated published checkpoint.

## Caveats and disposition

The local training sandbox used a manually reconstructed semantic copy of the current `mamba130/world.py` and `counterfactual.py`, not byte-identical originals. Those repository modules are the direct imports in the published runner; exact-HEAD re-execution could produce different metrics. This prevents claiming source-identity qualification. Training samples contained **no STOP class**, because the supervised teacher always moved when legal. The recurrent model is scratch-initialized and only trained for 64 updates; architecture-vs-pretraining conclusions are unwarranted. No compute-matched pretrained comparator was run. Neither 4-step truncated-memory intervention nor closed-loop data shows a substantive robust benefit of recurrence.

**Disposition:** trained scratch minGRU architectural control, **not promoted**. Next useful step for this lane is an actual source-pinned *pretrained* new family (prefer a compact xLSTM/RWKV/Mamba variant not already catalogued) with 64+ bounded real adapter/block updates on the same fixed seed splits. If weights remain unreachable, explicitly report acquisition/dependency blockers instead of another scratch architecture. SIMD, quantization and kernel optimization belong to Extreme Quant.

Files: `run.py` is the complete reproducible research experiment, `metrics.json` contains measured losses/splits/latencies/hashes, `trajectories.jsonl` preserves each raw and masked action/event, and `trajectories.svg` is the original seeded gallery. No native CyberSand changes, Actions workflow, expensive auto-training, PR creation, or source changes outside `model_zoo/**`.
