# C71 — Unseen-remote-goal transfer and route-feature ablation (2026-11-18)

**Scientific disposition: negative / HOLD. No policy promotion.** This is a CPU-only, synthetic-CyberSand-like host experiment, not CyberSand-native gameplay or a replacement for the authoritative host ABI. Research source and evidence belong exclusively to DRIV3R consolidated Draft/HOLD PR #1.

## Question and pre-specified selection

Does on-policy training of the C70 six-feature route-conditioned recovery actor transfer to genuinely unobserved remote goals? Does its observed-route match feature cause changes that matter, compared with frozen neural policies, non-neural frontier search, RESET and a wrong-credit control?

The fixture world is generated independent of policy suffix outcomes. Source C71 seed `710119`, deterministic five-family geometry generation (`tree`, `braid`, `prim`, `wilson`, `kruskal`), prefix tick 9; for each family the first 2 strict-eligible graphs train and the next 2 test. Each target must be absent from visited memory, absent from stored menu-neighbor keys, and must have **no previously observed ledger route** at prefix time. Historical graph identities from frozen C60 exclusions and C61/C64–C70 receipts were excluded before generation; generated graph and prefix SHA-256 identities are in the summary. Families are small independent graph clusters; test graphs were not selected by outcomes.

The policy receives the host-offered legal action menu, current position/published goal, observed movement/painting history, and issued corrective cue. No geodesic oracle, future rollout, protected-cell mask or hidden graph edges are used in policy features or inference. The historical d_common code is a fixture only. The synthetic host retains action legality and protected-cell checks.

## Real optimizer work and controls

Two fresh **stochastic linear actor heads**, ALIGNED and SHIFTED, each received 36 CPU AdamW updates in three on-policy epochs at the identical predefined graph/episode/optimizer schedules. The shifted-credit negative control cyclically rolls delayed returns within each episode, retaining some within-episode reward autocorrelation; its different on-policy rollouts produced different action exposure. Training actions per epoch:

| Head | Epoch 0 | Epoch 1 | Epoch 2 | Updates |
|---|---:|---:|---:|---:|
| C71-ALIGNED | 1552 | 1594 | 1438 | 36 |
| C71-SHIFTED | 1552 | 1738 | 1604 | 36 |

Fresh independent CPU refits produced **byte-identical checkpoints**, matching published SHA-256 identities. No paid API, GPU or external training resource was used. Checkpoint binaries are excluded from Git history pending immutable Actions artifact custody.

## Matched held-out results

10 genuinely unseen target graphs, two per topology family. Each graph was evaluated with no guidance, weak correction (0.35), strong correction (1.0), and intact or 100%-erased historical menus. 60 matched condition rows, seven policies, **420 actual executed suffixes** and **13,540 host-legal actions**, with zero action rejections or protected-cell mutations. All 20 no-guidance rows preserved identical action and painting hashes across all seven policies.

**Goal completions / 10** (goal means the published remote target was reached within the 70-tick horizon):

| Controller | Weak intact | Weak fully erased | Strong intact | Strong fully erased |
|---|---:|---:|---:|---:|
| RESET | 3/10 | 3/10 | 3/10 | 3/10 |
| FRONTIER | 4/10 | 4/10 | 4/10 | 4/10 |
| C60-FROZEN | 3/10 | 3/10 | 3/10 | 3/10 |
| C70-FROZEN | 3/10 | 3/10 | 3/10 | 3/10 |
| C71-ALIGNED | 3/10 | 3/10 | 3/10 | 3/10 |
| C71-SHIFTED | 3/10 | 3/10 | 4/10 | 4/10 |
| C71-NO-ROUTE | 3/10 | 3/10 | 3/10 | 3/10 |

ALIGNED tied C70-FROZEN and RESET under every guided condition at 3/10; non-neural FRONTIER completed 4/10. Under strong guidance, shifted-credit completed 4/10. The paired aligned-minus-frontier result was zero wins, one loss and nine ties for each guided arm; there is no demonstrated delayed-credit learning advantage. No confidence interval can turn this small, largely tied exploratory cohort into an affirmative generalization claim.

ALIGNED, C70-FROZEN and SHIFTED did not change executed action/ink hashes after full erasure in any of the 10 graphs under either guidance strength; this is **not** affirmative evidence of resilience. Initial remote goal routes were absent by construction. FRONTIER changed two of ten trajectories under each strength, consistent with its continued use of exploration history.

The C71-NO-ROUTE action-feature ablation forced the route-match feature to zero while keeping other observations and head parameters unchanged. It produced identical actions/painting and goals to C71-ALIGNED in all guided test conditions. Its `route_matching_actions` counter must be interpreted under feature ablation (always zero), not as an independent behavioural choice. This suggests the observed-route feature had no realized effect in this selected remote-goal test, not that route memory never matters elsewhere.

## Verification and custody

- 8/8 source-matched evidence and legality regression tests passed, including reconstruction of all 20 strict-unknown train/test prefixes, protected integrity, no-guidance parity, complete ledger/aggregate reconciliation, and route ablation.
- Two independent CPU optimizer fits yielded byte-identical final safetensors for each trained head, with 36 updates per head.
- Eight selected host-executed gallery trajectories across four distinct test graphs were replayed and action, ink, goal and full path records matched exactly. These replays are *not* independent experimental cases.
- Complete source, summary, compressed full executed ledger, SVG paths, replay receipt and digest ledger are text files intended for this laboratory only. Original tiny safetensors remain only in the owner archive pending immutable GitHub Actions artifact custody; no binary is committed to Git.

Source SHA-256: `bf19f954e5596b43bb9dd60e970a5826101c19b7f471f81a2fc6c77da28b346a`  
Frozen C70 source SHA-256: `cac1f5a31b3d1c322a219b19e3987461ff67f4921957fc8325cd4860b2324436`
  
C71-ALIGNED checkpoint SHA-256: `043dcec178cfc6ae88acd6926ef88b2d986ab9c12071d3ec06a19e381f402e94`
  
C71-SHIFTED checkpoint SHA-256: `01312966d2213a5772802a24b254c511af377258098ee12f2ea9164db29a7314`

## Limitations and next gate

The world and movement/painting simulation are synthetic; even host-legal synthetic actions are not independently native CyberSand-qualified. Remote targets are unobserved at prefix selection but the published target coordinate is known, and path-memory features can become available only later. The 10-test-graph cohort and 70-tick horizon are intentionally bounded, with some difficult topologies yielding no goal completions. On-policy action exposure is not exactly matched after models diverge, and the shifted-return control preserves temporal correlations. No-guidance parity and limited movement/ink counters are **not** complete personality preservation. Full guided personality and CyberSand native acceptance remain outstanding.

**Decision:** HOLD C71, no promotion, no merge. The next substantive experiment should improve goal-directed frontier navigation rather than fit another cue selector, with a larger preregistered remote-goal cohort, matching optimizer *and action exposure* where possible, stronger graph-level wrong-credit controls, and explicit long-horizon personality signatures. Maintain strict DRIV3R-only source ownership. No CyberSand/Neurounner mutations.
