# C31 — Visited-goalward recovery and deep false-progress controls

**17 November 2026 · DRIV3R guidance/recovery laboratory · synthetic-only / Draft-HOLD / NOT PROMOTED.**

## Question

C29's probe controller treats only **fresh** goalward cells as evidence of a legitimate route. That rule can mistakenly abandon a real shortcut when it intersects an *already executed* host trail. C30's preliminary work (separate, source-incomplete) explored deeper probes but still reported true-tunnel regressions. C31 tests whether consulting the *current host-authorised action menu* for **any legal goalward continuation, even one crossing visited cells**, resolves this failure without clairvoyant geometry.

C31 separates three elements: (1) a deterministic legal-progress rule, (2) a trained 6→8→1 move-ranking head with a fixed bounded exploration bonus, and (3) the same head with a learned bonus. This is deliberately a **causal ablation**, not an attempt to attribute all gains to a neural optimizer.

## Execution and training

The synthetic host executes 27 actual legal movements around an enclosed U corridor and then changes the goal to the first visited cell. The shortest route back is **27 steps** if a direct side branch is a false spur, versus **9 steps** if it is a real tunnel. False spurs have two, three or four legal goal-improving moves before terminating; BFS verifies solvability offline but its information never enters policy features. The first `depth-1` action menus are identical for true and false branches. Training uses eight rotations/reflections with unshifted coordinates and depths 1 and 3; evaluation shifts the layouts and withholds depth 2 and 4. This is constrained synthetic OOD, not broad native geometry transfer.

The move-ranking head was trained on 192 host-observed menus / 736 labelled legal actions with **240 CPU AdamW optimizer steps**. Loss sum per epoch declined **5.9529603 → 0.9364626**, in-sample accuracy **95.65%** (not a generalisation estimate). Same-seed optimizer replicas produced byte-identical safetensors (checkpoint SHA-256 `490c1d8a1dd6f30aa2ae76e3dd37b78c30a74409bff6f0cfa82f07bfc30eb8f5`). All training labels reflect currently observable host legal progress, not an oracle's future solution.

The host determines legality and protected-cell material permissions. Each action is freshly drawn from a finite legal menu; active exploration caps the selected action's sampling mass at 88%, and persistent BACKTRACK caps at 98.5%. No advice yields the exact frozen C18-A0 policy. Goal changes and explicit advice cancellation release the option.

## Matched independent U-corridor evaluation

Two independent fresh-seed cohorts, each 64 guided + 16 cue-off episodes **per policy**. Four outcome families with matched arm/guidance controls and two personality variants were not used to fit the head. Results are **completed goals out of 64 guided episodes per cohort**, with condition-level gates relative to C18-A0.

| Policy | A | B | Strict group nonregression both? | Decision |
|---|---:|---:|---|---|
| Frozen C18-A0 | 13 | 12 | reference | reference |
| C29-GUARD | 13 | 12 | yes | fails deep spurs |
| C29-PROBE | 34 | 34 | **no** | true-tunnel loss in B |
| C31-FRESH_ONLY | 31 | 34 | **no** | true-tunnel losses in both |
| C31-LEARNED | 40 | 43 | yes | learned, not superior |
| C31-RULE (fixed logit bonus, learned ranking) | 41 | 45 | yes | ablation |
| **C31-DIRECT (no learned ranking or bonus)** | **41** | **45** | **yes** | strongest simplicity control |

Withheld **four-step false spurs**: C31-DIRECT recovered 8/16 in A and 6/16 in B; C29-PROBE recovered 1/16 and 0/16; C18-A0 never reached either goal. Real tunnel: DIRECT 16/16 in each cohort, baseline 13/16 and 12/16. Direct and RULE have equal aggregate outcomes but *different paths* (70/80 identical action+ink hashes in A, 68/80 in B): their equal totals do not imply identical policies.

Across the 32 cue-off matched evaluations per policy, C31 candidates retained 32/32 exact baseline action and ink hashes. Zero synthetic host rejections or raw illegal top choices were observed.

## Historical standard-fixture regression audit

Separately evaluated the standard C29 synthetic outcomes (reverse, short, near, tunnel, one-step decoy) at a 90-decision horizon, **two cohorts of 160 guided and 40 no-guidance trials per policy**. Each matched condition uses the same host-executed prefix, personality, guidance arm and random seed.

| Policy | Standard A /160 | Standard B /160 | Regressed guidance groups? |
|---|---:|---:|---|
| C18-A0 | 82 | 77 | reference |
| C31-LEARNED | 137 | 144 | none |
| C31-RULE | 144 | 149 | none |
| **C31-DIRECT** | **144** | **149** | **none** |

All 80/80 matched cue-off trials per C31 candidate had exact baseline action and ink hashes. No host-rejected commands or raw illegal top-choice actions.

## Interpretation and limits

**Primary positive result:** a host-observed *visited-but-legal goalward* continuation prevents premature rollback on genuine tunnels, while an eventually exhausted branch can still trigger stochastic persistent recovery. This improved aggregate completions across the tested fixtures without measured group-level regression.

**Primary negative result:** the trained six-feature head **did not outperform the simpler no-head DIRECT rule**. This is evidence for an action-menu semantics / recovery-policy improvement, **not learned long-horizon reasoning**. The sensor-level rule itself is nontrivial but transparent; a neural explanation for the gain would be unjustified. Training uses action-local labels and one restrictive corridor generator, and the evaluation has few independent map families. Exact no-guidance behaviour says nothing about active-guidance drawing personality. Native CyberSand action/observation ABI compatibility and physical Workbench acceptance remain untested. Checkpoints are kept outside Git; uploaded local archive is interim custody, *not* immutable GitHub Actions artifact custody.

**Disposition:** retain C31-DIRECT as a research-only policy control and C31-LEARNED as a negative neural ablation. No promotion. Maintain DRIV3R PR #1 Draft/HOLD without merge; external CyberSand/Neurounner repositories and DRIV3R main are read-only. The C30 preliminary evidence and source-incomplete test files are separate pre-existing work, not modified by C31.
