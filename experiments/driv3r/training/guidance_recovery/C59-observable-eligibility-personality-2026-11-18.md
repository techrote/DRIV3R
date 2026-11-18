# C59 — Observable guidance eligibility and short-horizon personality proxies

**Disposition: RESEARCH HOLD · 18 November 2026 · synthetic-host only.** Owner: `experiments/driv3r/training/guidance_recovery/**` in techrote/DRIV3R Draft/HOLD PR #1. No changes to CyberSand native/Workbench, Neurounner, DRIV3R main or immutable historical interfaces.

## Question and method

C58's aligned neural action head improved **observed-waypoint** recovery after joint memory damage but harmed **unobserved remote-goal** recovery. C59 isolates whether the failure can be mitigated by *abstaining from learned action bias when no host-observed legal route is recoverable*, rather than claiming the action model can plan through unseen terrain. It also measures a bounded directional movement-personality proxy instead of equating unguided hash parity with preserved guided character.

The **frozen C58-ALIGNED** checkpoint is reused unchanged, and a separate one-layer six-input **C59 eligibility head** is trained for **240 CPU AdamW updates** to imitate an explicit host-visible directed-route test. The teacher uses only previously observed neighbor menus, retained authorized action traces, current goal, and previously visited positions. No shortest-path oracle, remote geometry or future goal outcomes enter training or inference. This eligibility target is much simpler than navigation, and the learned gate must be compared with the trivial non-neural `STRICT` rule.

Six matched policies: C18-A0 unguided, C34-RESET, original C58-ALIGNED, **C59-STRICT** (only bias when a remembered route to the published goal exists), **C59-SOFT** (same restriction with half the extra logits), and **C59-LEARNED** (gate probability at least 0.5). Guidance strengths are none, weak (0.35), and strong (1.0); joint historical-menu/action-log erasure is 0%, 50%, 100%. Current host-legal actions are always reobserved and revalidated. Random generator and global clock continue across the host-issued correction at tick 9.

Training used **72 eligibility examples from 13 contributing graph identities** within an independently seeded 16-graph familiar-family source cohort; **14** examples had an observed legal route. Train accuracy was **98.61%** with Brier **0.0185**, but this measures imitation of a simple route-presence rule, not model intelligence or long-horizon navigation.

Held-out research evaluated **24 new graph identities** (4 Wilson, 4 Kruskal, 4 tree, 4 braid, 4 Prim, 4 Prim-braid), all excluded from C56–C58 train and evaluation graph hashes and from C59 training graphs. Sixteen of those graphs produced eligible matched states of both target classes. The 288 matched conditions × 6 policies yielded **1,728 policy suffix executions**, including **96 unguided conditions per policy**, identical action and painting hashes across all policies. Zero synthetic-host rejected actions or protected-cell mutations. Trajectory panels were rerun in-source against actual recorded action and painting hashes; separate native-host replay is not claimed.

## Held-out goal completion

Each cell gives goals out of **32 guided episodes per policy** (16 eligible graphs × weak/strong correction), on the same graph identity within each memory condition. These are paired, *not* independent samples across memory levels.

| Goal | Memory erasure | RESET | Frozen C58 | C59-STRICT | C59-SOFT | C59-LEARNED |
|:--|--:|--:|--:|--:|--:|--:|
| Observed waypoint | 0% | 24 | **26** | **26** | 25 | **26** |
| Observed waypoint | 50% | 24 | **26** | 24 | 25 | 24 |
| Observed waypoint | 100% | 24 | **27** | 24 | 24 | 24 |
| Unknown remote goal | 0% | **12** | 10 | **12** | **12** | **12** |
| Unknown remote goal | 50% | **12** | 10 | **12** | **12** | **12** |
| Unknown remote goal | 100% | **12** | 10 | **12** | **12** | **12** |

**Effect:** With unknown remote goals, abstention eliminates the two C58 regressions per corruption condition and ties RESET: **12/32**. It does **not** improve remote navigation; it simply stops the harmful intervention. At intact memory, STRICT/LEARNED retain C58's +2 observed-waypoint gain over RESET. At 50% and 100% joint memory loss, their observed-waypoint advantage disappears. The original unconditional C58 remains +2/+3 on those easier waypoints, but has remote regressions. Soft gating offers +1 observed recovery at 50% damage while tying RESET remotely, though this is a single-episode effect.

Regressions persist in observed intact-memory conditions: STRICT and LEARNED have **4 paired wins and 2 paired losses** versus RESET among 32 guided episodes. **No controller is eligible for promotion.** The study is small, adaptive, and not preregistered; goal differences and bootstrap intervals are descriptive, not proof of generalization.

The fitted gate's **96 held-out pre-correction state labels** produced 22 true positives, 70 true negatives, four false positives and zero false negatives against the observed-route test. Its classification success cannot justify retaining a neural eligibility gate when the explicit test is both simpler and more transparent. Some remote runs later discover a legitimate route during execution; STRICT allows only those revalidated route-guided turns, not hidden-maze access.

## Personality and limitations

For each policy, a separate first-20-action movement proxy was calculated only when **both paired trajectories lasted at least 20 actions**: turn frequency, immediate reversals, unique visited positions, directional entropy and painting fraction. This removes obvious unequal-duration bias but produces small, selected matched samples: **six observed** and **24 remote** cases per memory level for the reported turn-frequency comparison. The strict/soft gate tied RESET's measured remote-goal proxy in the retained samples; frozen C58 differed in turn frequency by +0.0469 and unique-position fraction by −0.0317 on remote-goal suffixes. That is not an objective human assessment of meandering, branch exploration, or aesthetics. Early successes are censored. These limited metrics cannot establish whole-trajectory personality preservation.

No exploration breakthrough was observed. The eligibility gate is deliberately an imitative neural head over a trivial human-readable rule, and the conservatively gated policies forfeit some recovery benefits from local fallback precisely where historical route information is most damaged. Evidence is from an external synthetic host, not CyberSand native gameplay. The historical `d_common` fixture is not a newly approved host ABI. This series remains adaptively selected; confidence intervals are exploratory and no paid APIs/GPUs were used.

## Evidence and reproducibility

Repository evidence is text-only: `c59_eligibility_personality.py`, `test_c59_eligibility_personality.py`, `c59_gallery.py`, `C59-summary.json`, `C59-gallery.svg`, `C59-executed-ledger.zlib.b64`, this report, and SHA-256 receipt and extractor. The compressed complete ledger encodes all 288 matched conditions and their six execution outcomes, plus the action-hash-verified galleries. The Git repository does **not** contain checkpoint binaries. The full owner archive carries the C59 checkpoint, frozen C58 model and predecessor source/evidence with a SHA-256 manifest. **Immutable Actions checkpoint artifact custody and native CyberSand acceptance remain outstanding.**

To validate from the owner archive, extract it, set `PYTHONPATH` to the listed frozen source directories, run the six unit tests, then execute `c59_eligibility_personality.py <prior> <new_output> --train-reps 1 --online-reps 1`. Exact checkpoint hashes, online evidence and graph separation should be verified rather than assumed from the import. The frozen C58 source was checked against Git blob `63c85e63a4067f8b6cd7580c8c999e6f8486c0c6` on the DRIV3R research branch. An independent second local CPU optimizer/evaluation run reproduced the C59 gate checkpoint, summary, full online ledger and trace files **byte-for-byte**; six unit tests also passed. This is deterministic external synthetic-host rerun evidence, not host-native qualification.

**Decision: HOLD.** Next research: find a remote-goal exploration mechanism that improves completed goals beyond RESET without relying on an already discovered route; independently validate whole-trajectory character on host-legal gameplay. Avoid adding another label-imitating classifier without a measurable gameplay improvement.
