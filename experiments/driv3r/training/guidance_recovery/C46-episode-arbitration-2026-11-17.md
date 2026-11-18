# C46 — Goal-paired guidance arbitration (17 November 2026)

**Disposition: REJECT neural promotion / retain negative research evidence.** This is a synthetic CyberSand-like host study, **not** native CyberSand gameplay or a replacement for its host ABI. This report belongs only to `techrote/DRIV3R` Draft/HOLD PR #1. No game, Neurounner, or main-branch modifications.

## Question and causal design

Could complete, host-executed goal outcomes provide a better supervisory signal for guidance than C40–C45 short-horizon distance/return proxies?

For each training maze-and-cue instance, execute three frozen policies on **separate copies of the same host state**, using the same RNG seed: C18-A0 (unguided controller), C34-RESET (non-neural recovery), and C40-PAIR (frozen neural action-value head). Record both goal outcomes (including a scheduled second goal) and exact action hashes; reward goal events before optional speed tie-breaks. A small CPU-trained episode-start classifier predicts which controller to use from the **current host observation, legal action menu and issued cue metadata**. There is no oracle map or future return feature at inference.

The learned selector has a graph-block-shuffled-label control, a frozen random-initialized control and all three unselected controllers as exact matched controls. Fixed tests contain no-guidance, weak BACKTRACK, strong BACKTRACK and strong CONTINUE arms, both personality signs, detour/forward challenges, switched goals, and disjoint held-out Wilson/Kruskal maze families. Labels deliberately break ties in favour of C18-A0; this known design bias is analyzed rather than concealed. Selection occurs once at episode start, not mid-trajectory.

## Exact experiment

- **Train:** 64 graph identities spanning DFS/DFS-braid/Prim/Prim-braid; 128 graph-cue records, 3 complete matched counterfactual rollouts per record, 18 epochs and **144 AdamW updates per trained head**. 91/128 training labels chose A0, 16 RESET and 21 PAIR. Aligned/shuffled heads were independently retrained and their weight tensors matched byte-for-byte.
- **Blind:** 16 disjoint Wilson/Kruskal graphs, 32 graph-cue records. Both trained heads matched 28/32 majority-A0 labels (**87.5%**), exactly the majority-class baseline, without showing any learned differential. Random head matched 2/32.
- **Online:** 48 fresh Wilson/Kruskal graphs, 3 cohorts, 192 matched episodes per policy (48 no-guidance, 144 guided), **1,152 policy-labeled episodes**. A selected policy reuses an identical executed base trajectory rather than rerunning and introducing sampling noise; all 18 gallery traces were independently re-executed and action/ink hashes matched.

| Policy | Guided goal events | Condition regressions against RESET | Selection behaviour |
|---|---:|---:|---|
| C18-A0 | 34 | 4 | Always A0 |
| C34-RESET | 30 | 0 | Always RESET |
| C40-PAIR | 30 | 3 | Always C40 |
| **C46-ALIGNED** | **34** | **4** | **192/192 A0** |
| C46-SHUFFLED | 34 | 4 | 192/192 A0 |
| C46-RANDOM | 29 | 3 | 60 A0 / 71 RESET / 61 C40 |

The aligned selector's paired goal difference versus RESET is +4, but its **95% descriptive graph-cluster bootstrap interval is −6 to +15**, and it loses on six graphs while winning on seven. Its delta against SHUFFLED is exactly zero on all 48 graphs (descriptive interval [0,0]). The aligned selector never activates recovery, so this is **A0 baseline replication, not neural guidance success**. Four condition-level regressions disqualify it outright.

## Qualification and failure modes

Four targeted regression tests pass: host-observation input/determinism, goal dominance over speed, optimizer/label-shuffle repeat determinism, and exact unguided action/ink parity. All recorded host executions report zero rejections. Every policy preserves 48 no-guidance movement and painting hashes. The model's whole-trajectory personality preservation is **not** established: preferred/opposed turn counts are descriptive only and confounded by visited-state differences.

This is a useful negative result: outcome-derived supervision did not prevent a small selector from collapsing to the majority controller. That collapse is structurally encouraged by the 91/128 A0 labels and A0-favouring tie policy; the 87.5% blind score is therefore uninformative despite appearing high. Full-goal labels and a held-out topology split are not sufficient by themselves to demonstrate adaptive recovery.

**Further research:** Pre-register a much larger, balanced training dataset and separately held-out evaluation before considering weighted or cost-sensitive learning. Include strong CONTINUE labels in training, true mid-trajectory feedback, per-condition nonregression, wall-clock recovery latency, whole-trajectory personality retention and native CyberSand validation. Do not tune to the current Wilson/Kruskal evaluation cohorts.

## Evidence custody and limitations

`c46_episode_arbitration.py` is the exact executable source. `test_c46_episode_arbitration.py` is the regression qualification. `c46_gallery.py` independently renders replayed synthetic-host paths from the retained trajectory receipts. The accompanying `C46-summary.json` is the compact result, with source/checkpoint SHA-256 values, paired graph-cluster intervals and model selection frequencies. Full training, blind, online, traceback, checkpoints, prior sources and dependencies are retained in a local SHA-256-manifested archive, **not immutable GitHub Actions artifacts**; no reproducibility or custody claim stronger than the actual local verification is made. None of the checkpoint binaries is committed to Git history.
