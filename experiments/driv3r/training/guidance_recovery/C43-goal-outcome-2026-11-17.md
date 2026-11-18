# C43 — Goal-outcome recovery targets versus short-suffix ranking (17 November 2026)

**Disposition: DRAFT / HOLD; negative neural result. Synthetic host evidence only.** No promotion, native runtime change, or claim of full reproducibility. Source is packaged losslessly in `C43-extract-sources.py` in this directory. Running `python C43-extract-sources.py ./c43_restore` restores ordinary readable `c43_goal_outcome.py` and `test_c43_goal_outcome.py` with enforced SHA-256 checks. This avoids committing model binaries; the compressed source is not itself a model checkpoint.

## Question and intervention

C40 suggested that ten-step executed-return rankings might improve recovery, but C41/C42 could not reliably transfer that finding. C43 instead trains a 34-feature host-legal action head on **executed goal outcomes** under a fixed ReArmedWatch continuation. Each legal candidate is executed as the first action in two paired stochastic, host-legal 54-step suffixes (BACKTRACK strength 0.35 and 1.0). The target weights actual goal achievement most heavily, with a small best-observed Manhattan-progress backoff and stop penalty. Shortest-path/oracle graph data are not exposed to inference.

The four training families are DFS tree/braid and randomized Prim/Prim braid; the blind and online families are independently constructed Wilson and Kruskal mazes. Training, blind and online graph hashes are disjoint. Three heads start from the same architecture: aligned outcome targets, within-state shuffled targets, and untrained random initialization. Frozen C40-PAIR, C34-RESET and C18-A0 provide independent controls. The two trained heads ran nine deterministic AdamW epochs and exact repeated optimizer training before saving.

## Held-out results

**1,536 matched synthetic episodes** across four cohorts, 64 online graphs and six policies; 192 guided and 64 unguided episodes per policy. Blind test: 32 further graph identities.

| Policy | Guided goals | Strict condition regressions vs RESET | Blind mean goal-outcome oracle regret (lower better) |
|---|---:|---:|---:|
| C18-A0 baseline | 49 | 9 | — |
| C34-RESET | 48 | 0 | — |
| C40-PAIR | 48 | 6 | 0.526036 |
| **C43-OUTCOME** | **48** | **0** | **0.281398** |
| C43-SHUFFLED | 50 | 0 | 0.158270 |
| C43-RANDOM | 49 | 0 | 0.125381 |

C43-OUTCOME ties RESET, trails both shuffled and untrained heads, and scores worse than both on blind outcome regret. Its paired goal-event difference versus RESET is 0 (graph-cluster descriptive 95% interval [0,0]); versus SHUFFLED it is -2 ([-5,0]). Learned superiority is **not established**. The nonregression gate passes for C43-OUTCOME on this bounded fixture, but no actual recovery improvement follows.

All 64 matched unguided trajectories per model preserve baseline action and painting hashes. **No synthetic host-rejected execution** was recorded; four raw illegal top-choice proposals by C43-OUTCOME were filtered by the legal-menu mask. Protected-set hashes are asserted unchanged during the synthetic rollouts. C43-OUTCOME altered 6,183 cue-gated decision states, so its ineffectiveness is not due to a completely inactive head. Aggregate preference/ink ratios are insufficient for whole-trajectory personality acceptance.

Three targeted unit tests passed locally: deterministic executed suffix, host legality, and illegal first-action rejection. These tests require `C43_ROOT` to point to existing prerequisite files; they skip without the fixtures.

## Custody and limitations

* Readable Python source SHA-256: `46203e540b95d44c4b581da2c8bec1bc3b0e12a1bf920841bf28cbab315dd746`.
* Test SHA-256: `f37f30ae0149fa536bf8cba56417bc00d576faf2b8396fa8bdca22f2e3a50ae4`.
* C43-OUTCOME checkpoint SHA-256: `df4e70c5644b327429bca64ef2a8c8f9c4d11ee89984cb8ad49dee34b4018bc5`.
* C43-SHUFFLED: `63e00f3f593067dcf0ad47ca3cebd4a798c24292daffefbf07857e1e7ab63828`.
* C43-RANDOM: `824b139348d166c770196dcfe34687452ea0a521b90d46d1524bf31e29204655`.
* Compact summary SHA-256: `1144fa9b6156adf94fde17a59e52aab964842f706c5d0b721c59e9667c7b6ba1`.
* Full blind JSON SHA-256: `b043373901386a5b0a607516e30dd97232ea5d11ee53998c42a8a7ddff363f0d`.
* Full selected path JSON SHA-256: `08afc81407a93e96599eea0f9fcf41da79df51eaa6e6e679675af60f6cb68a5b`.

The first full evaluation invocation was interrupted **after training, repeated-optimizer comparison and checkpoint writing**. A second evaluation invocation was interrupted after blind evaluation and first cohort; the four cohorts were completed as separate deterministic batches, and then recombined. The initial unlogged training-state labels and optimizer curves were not retained. The code regenerates them, but end-to-end reconstruction from immutable GitHub Actions artifacts **has not occurred**. Therefore do not describe the study as fully scientifically reproduced. Local source, binaries, blinded cases, complete cohorts, eight visualized executed trajectories and a 15-file SHA-256 manifest are retained in an external local ZIP, pending immutable Actions custody. Binaries are not in Git history.

This experiment is a synthetic research fixture, **not** CyberSand's authoritative native actor or action/observation ABI acceptance. Stronger goal-outcome labels did not solve the long-horizon problem. Future experiments should first examine where sparse long-horizon targets collapse to identical outcomes and whether training differs meaningfully from shuffled control at action-decision level. No paid GPU/API spend; no changes outside the owned laboratory path.
