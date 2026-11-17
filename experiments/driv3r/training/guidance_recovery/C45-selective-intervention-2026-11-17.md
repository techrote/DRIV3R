# C45 — Counterfactual selective guidance, held-out topology test

**Repository/ownership:** `techrote/DRIV3R` → existing Draft/HOLD PR #1, `research/import-consolidated-worm-labs`. Owned lab `experiments/driv3r/training/guidance_recovery/**`. This experiment is **synthetic-host research**, NOT a native CyberSand result. No external repository changes or model promotion are authorized.

## Decision

**HOLD / reject C45-GATE for promotion.** A trained abstention gate reduces the use of the C40 action ranker and weakly improves held-out *counterfactual label classification*, but makes **no improvement in goal completion** over RESET or a shuffled-label gate on 48 independent Wilson/Kruskal maze graphs. Despite 75/144 guided action+ink trajectories changing relative to RESET, the learned gate finishes **21 vs 21** guided goal events, with no condition regressions but no gains. It is premature to replace the non-neural controller or to claim personality preservation.

## Intervention and experimental design

C45 inherits the C40 pairwise action ranker (frozen) and trains a tiny 38-input/16-hidden-unit binary gate deciding whether C40 may perturb the RESET legal-action logits at a guided decision. Its inputs are 34 host-visible candidate statistics plus four frozen-ranker dispersion/margin statistics; they do not contain graph identity, oracle return, future path, or protected-cell bypass. When the gate declines, action-ranker score adjustment is precisely zero, leaving RESET. When it accepts, the original C39/C40 bounded, personality-projected adjustment applies. Existing host legality masking remains authoritative.

The supervision is the *sign* of the difference between expected **20-action executed-candidate returns** under two local action distributions (C40-adjusted versus approximated RESET). This is deliberately an approximation to future recovery, not actual differential goal success. The RESET phase is reconstructed at stored host-observed states; its full recurrent history is not reconstructed and may introduce label noise. Training actions are sampled from RESET-visited host trajectories, with no teleportation. The shuffled control receives the same architecture, initialization, training order, and target prevalence, but permutation-destroyed labels. A simple ranker-margin gate and ungated C40 are independent controls.

- Training: 64 disjoint DFS, braided DFS, Prim and braided Prim graph instances; 186 actively visited decision states (81 positive, 105 negative counterfactual-advantage labels).
- Two trained heads: **2,232 AdamW optimizer updates each**, 12 epochs, fixed initialization and deterministic CPU repetition. Both repeat-trained state dictionaries match byte-for-byte.
- Blind evaluation: 48 decisions on 24 independent Wilson/Kruskal graphs, with labels from newly executed legal candidate continuations.
- Online testing: 48 other Wilson/Kruskal graphs in three independently seeded cohorts, each with a new/unchanged goal and no hint, weak BACKTRACK, strong BACKTRACK and strong CONTINUE; six policies, **1,152 matched episodes** (144 guided per policy, 48 unguided per policy). Train/blind/online graph IDs are pairwise disjoint. Seeds, controls and threshold (0.5) are fixed in the source, not selected after viewing held-out performance.

## Held-out policy outcomes

| Policy | Guided goal events / 144 episodes | Conditions below RESET | Exact unguided movement+ink histories |
|---|---:|---:|---:|
| C18-A0 | 28 | 3 | 48/48 |
| C34-RESET | 21 | 0 | 48/48 |
| C40-PAIR | 22 | 2 | 48/48 |
| **C45-GATE** | **21** | **0** | **48/48** |
| C45-SHUFFLE | 21 | 0 | 48/48 |
| C45-MARGIN | 22 | 2 | 48/48 |

For C45-GATE minus RESET, the paired graph-cluster goal difference and its descriptive 4,000-bootstrap 95% interval are both **0 [0,0]**. This **is not equivalence proof**: the small number of successful goals makes the test low-powered. C45-GATE minus ungated C40 is **−1** goal (descriptive 95% interval **−6 to +3**). The only C40 gain over RESET is one extra strong-BACKTRACK goal; its interval is **−3 to +5**. The simple margin rule behaves exactly like C40 on these goal outcomes, despite suppressing a minority of action-head calls.

C45-GATE altered 75 of 144 guided movement+painting trajectories relative to RESET, yet no additional goals. The shuffled gate altered only 8 of 144. The learned gate opened on 1,816 of 5,847 eligible calls (~31%); shuffled on 122 of 5,777 (~2%); simple margin on 5,756 of 6,084 (~95%). These activation counts refer to actual model calls, not unique episodes.

Blind counterfactual classification: C45-GATE accuracy **70.83%**, Brier score **0.2085**, enabled **14.58%** of states; C45-SHUFFLE **64.58%**, Brier **0.2216**, enabled **4.17%**. The label prevalence is **31.25% positive**, so the trivial all-negative classifier achieves **68.75% accuracy**; C45's 2.08-point improvement over that baseline is weak and cannot establish useful calibration. Offline predictive gains do **not** validate online benefit.

Guidance itself should not be presumed beneficial in every test cohort: the unguided A0 baseline completes 28 events versus RESET's 21. This is a property of these synthetic Wilson/Kruskal cohorts and does not establish that all guidance harms gameplay.

## Safety, reproducibility and independent checks

Four focused C45 local tests pass, covering a 38-feature host-visible gate input, exact fallback-to-RESET score delta on rejection, exact frozen-C40 scores on acceptance, and deterministic optimizer replication with a distinct shuffled control. All executed synthetic-host actions are legal, no protected-mask modifications occur, and zero host-rejected actions are executed. One C40 and one SHUFFLE raw top-choice proposal were illegal before the host legality filter. The 12 retained example paths were **executed a second time through the same host fixture**, matching action and ink hashes; this is replay confirmation, not an independent host implementation or native gameplay validation.

The complete local ZIP retains full 1,152 episode records, 186 training-state receipts, 48 blind decision records, checkpoint binaries and hashes, 12 selected executed trajectories, an HTML gallery, source dependencies, four regression tests, training stdout and SHA-256 custody manifest. The repository includes the trainer, tests, gallery renderer, compact measured summary and retained trajectory records. Binary checkpoints are **not** committed to Git. **Immutable GitHub Actions artifact custody is still missing**, and no new workflow or training-on-push was created.

The existing `C44-unpack.py` restores historical `c44_onpolicy_pairwise.py` from its SHA-verified `.zlib` fixture. Full local reproduction additionally requires prerequisite safetensors and the exact older research modules; these remain bundled in the local archive but are not represented as a GitHub Actions checkpoint artifact. Do not claim repository-only scientific reproduction from a source-smoke pass.

## Interpretation and next test

C45 is a **falsification of naive selective abstention**: choosing when to deploy a short-return ranker does not remedy the mismatch between local proxy and whole-trajectory goal recovery. The next bounded study should avoid adding another classifier around this proxy. A stronger evaluation would gather **paired full-goal outcomes for decisions under the same upstream state**, preserve exact host observation histories, and compare the learned policy against RESET/shuffled gates on independent graph clusters with predeclared minimum nonregression and whole-path style thresholds. If that study cannot generate a reliable positive signal, pause neural recovery training in favor of diagnosing the recovery objective and synthetic-to-native gap.
