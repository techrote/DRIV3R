# DRIV3R Guidance & Recovery — C44 on-policy pairwise ranking

**17 November 2026 · Draft/HOLD · Research only · REJECT for promotion**

**Ownership:** All new files are exclusive to `experiments/driv3r/training/guidance_recovery/**` in the existing [DRIV3R research PR #1](https://github.com/techrote/DRIV3R/pull/1). Neither CyberSand, Neurounner nor DRIV3R main was modified.

## Question and intervention

C43's longer goal-outcome oracle failed against non-neural controls. C44 asks whether C40-style action ranking becomes useful by **training on active-guidance states actually visited by the host-legal recovery controller**, rather than hand-selected maze-prefix states.

Collected **138 visited training states** on **48 DFS/tree/braid/Prim/Prim-braid training graphs**, yielding **722 legal candidate examples**, including **101 nonconstant return vectors**. Labels are independently executed **20-step host-legal continuations** from cloned states. The learner receives only the existing **34-dimensional observation-plus-candidate-action features**, not the graph oracle, future continuation, or any new action type.

Controls: C44-ONPOLICY trained on actual returns, C44-SHUFFLED trained on returns shuffled within each state, C44-RANDOM untrained, alongside fixed C34-RESET, frozen C40-PAIR and C18-A0. The aligned and shuffled heads each completed **7 AdamW epochs / 966 CPU optimizer updates**; independent repeat executions reproduced both checkpoints and optimizer curves byte-identically.

Blind evaluation uses **48 active-guidance decision states** on **24 withheld Wilson/Kruskal graphs**. Online evaluation uses **64 other, mutually disjoint Wilson/Kruskal graph identities**, split into four cohorts. Six policies × four guidance arms (no cue, weak BACKTRACK, strong BACKTRACK, strong CONTINUE) × 64 graphs = **1,536 synthetic episodes**, of which each policy has 192 guided episodes. All 48 selected executed trajectories were independently replayed and verified against movement/action and ink hashes.

## Held-out results: no demonstrated neural recovery advantage

| Policy | Completed guided goal events / 192 | Regressed condition groups vs RESET | Blind oracle regret (lower better) |
|:--|--:|--:|--:|
| C18-A0 | **40** | 9 | — |
| C34-RESET | 36 | 0 | — |
| C40-PAIR | 33 | 7 | 0.478125 |
| **C44-ONPOLICY** | **36** | **1** | **0.387500** |
| C44-SHUFFLED | 34 | 2 | **0.265625** |
| C44-RANDOM | 34 | 2 | 0.408333 |

ONPOLICY ties RESET and loses to C18-A0. Its graph-cluster goal delta against RESET is **0**, with descriptive 95% bootstrap interval **[-3,+3]** from 4,000 resamples. Its +2 goal events against SHUFFLED come from only **two favorable graph clusters**; interval [0,+5] is insufficient to claim robust neural superiority. **ONPOLICY fails a condition-level nonregression gate**. Blind offline oracle regret actually favors SHUFFLED over the aligned learner, despite learning and altered candidate rankings.

## Host legality, personality and caveats

No synthetic host-rejected execution or protected-cell mutation. Every policy retained all **64 unguided movement and painting hashes** from its matched frozen baseline. All **48** audited synthetic-host paths reproduced both recorded hashes.

Across guided episodes, preferred-ink fraction was **0.5955 RESET**, **0.5925 ONPOLICY**, **0.5923 SHUFFLED**, **0.5937 RANDOM**; mean within-state projected directional-preference shift for ONPOLICY was **0.00086**. These pooled proxies do **not** demonstrate whole-trajectory personality preservation, because interventions change state visitation and artistic choices. Native CyberSand acceptance is not claimed. The synthetic host is not authoritative game-side verification.

## Source and custody

C44 source SHA-256: `7eb0a0472c59d9a24561428fb314db083ec733995b32fe20472c44ce7e09c153`.
The three `.zlib` files under this laboratory are lossless, SHA-256-checked packages for the ordinary readable Python experiment and two JSON evidence ledgers; run `python C44-unpack.py` within the laboratory to restore original bytes without training. The test `test_c44_onpolicy_pairwise.py` passed **4/4** local source-only cases.

Compact results SHA-256: `454bbbc04b399e55591b14b033b75444f3f47c103792097fddaf8e76e03c67d4`; curated independently replayed path receipts SHA-256: `9ebe227b8a097a4d22c3266b96d2371b629f00ac7050155eab2fbf636287149c`.

Source-matched prerequisite Git blob SHA-1: `c39_action_value.py` `61cfdb7fabd7fc82fcac94d7492fd351d145d838`; `c40_pairwise_rank.py` `c58de2825a66ae5ea5e6c8318efff8e85bdc2d7d`; `c41_topology_transfer.py` `34f7448da4f900b0784ec0780d4759e0c3beb1cc`; `c42_wilson_transfer.py` `dc095edd7e8a9d87007f5e18ef7cadf0a4836790`; `c34_independent_maze.py` `569d4b5d0d6074d48989b353f71729cdf4f2afb0`; `c28_temporal_option.py` `4ad56a930bbb8040c84e2a444046f59b1e357de0`.

Checkpoint SHA-256: ONPOLICY `dfb20022bdcae496acfb80331feb61171320e40ee20b4128cac03596b1e6ccba`; SHUFFLED `f2393e2aee4b8513310ed4cef9843b2765a4af00e0522c5163fdf2a14d7cd461`; RANDOM `7e4e65126d0eef49116a083bda73f2c2faf1aa7d84c2f3c5785cab2b76893125`.

Full per-episode outcomes, training and repeat receipts, 48 executed trace records, PNG/HTML gallery, checkpoints and immutable hash manifest are retained in this run's downloadable source-matched archive, **not Git history**. Immutable GitHub Actions checkpoint custody, complete repository-only reproduction, whole-trajectory personality acceptance and native CyberSand qualification **remain outstanding**. CPU research does not justify scientific reproducibility from the historic import alone.

## Disposition

**HOLD / REJECT C44 for promotion.** More on-policy labels did not improve held-out goal completion relative to RESET; the shuffled control wins on blind offline regret. The next experiment needs stronger causal credit assignment on disjoint graph-generating processes and a predeclared acceptance gate: superiority to both shuffled and non-neural controls, **zero per-condition regressions**, and direct whole-trajectory personality evaluation.
