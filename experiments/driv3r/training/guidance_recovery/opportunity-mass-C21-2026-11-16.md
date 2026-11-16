# C21 — opportunity-conditioned personality under host guidance

**16 November 2026 · DRIV3R PR #1 · Draft/HOLD · research-only negative result.** No CyberSand or Neurounner writes. No promotion of C21 checkpoints.

## Research question

C20 attempted to protect drawing personality using short-horizon left/right turn gaps, but those gaps were sparse and sensitive to recovery trajectory changes. C21 instead asks whether a bounded BACKTRACK-only residual can preserve **the probability mass assigned to host-legal, personality-favoured INK turns on exactly the same candidate observation**, while improving full-episode outcome returns under true and false host recovery instructions.

The controlled comparison is against frozen trained C12, untrained C18-A0 (+1.0 attested BACKTRACK logit), and the rejected C20-STYLE head. No hint may override host legal permissions. The historical synthetic 32-feature menu is *not* the canonical CyberSand runtime ABI.

## Actual optimizer work

- Trained two independent **32→12→1** correction heads, C21-RETURN and C21-OPPORTUNITY, both initialized at the untrained C18-A0 gate. C21-RETURN optimizes host-executed counterfactual return advantages with a frozen-policy trust region. C21-OPPORTUNITY adds a penalty when the proposed policy reduces the probability of legal personality-favoured INK moves beyond a 0.5% relative margin at the same state.
- **324 matched training decision menus, 1,296 actual host-executed counterfactual suffixes**, from train orientations of reverse, ahead, shortcut and bridge worlds; scalar hint strengths across three arms; signed BACKTRACK/CONTINUE advice; both personality labels. **156 menus** expose a legal preferred INK opportunity. The narrower *both-sided-turn* requirement is absent at the training starting states and must not be used as a denominator.
- **144 CPU AdamW steps per variant, 288 total**. The objective for RETURN decreased from +0.0000656 to −0.0003581; OPPORTUNITY decreased from +0.0000656 to −0.0003014. The objectives differ, so their values are not direct model rankings.
- A separate complete optimizer repetition gave **bit-identical tensors** for each variant. This verifies local deterministic optimization, not cross-platform or full-chain scientific reproducibility.
- C21-RETURN checkpoint SHA-256: `22a8bf1b6efe3719ecb696a4d1d51867d7af65fd7b7bdd9dd427d834b56b0472`.
- C21-OPPORTUNITY checkpoint SHA-256: `6d2c8e5c7e96c83a3f18f9da6b8f3d648a6420f4672e4a22215448440bd8a028`.

## Independent held-out outcomes

**3,240 stochastic host-applied episodes** (5 models × 648 episodes each), seed `240201`, up to 56 decisions per episode; train/validation/test seeds disjoint. Additional 1,440 validation episodes used seed `230201`.

| Held-out condition | Frozen C12 | C18-A0 | C20-STYLE | C21-RETURN | C21-OPPORTUNITY |
|---|---:|---:|---:|---:|---:|
| Reversed goal, valid BACKTRACK | 39/54 | 51/54 | 51/54 | 51/54 | 51/54 |
| Reversed goal, misleading CONTINUE | 47/54 | 54/54 | 54/54 | 54/54 | 54/54 |
| Ahead goal, misleading BACKTRACK | 50/54 | 50/54 | 50/54 | 50/54 | 50/54 |
| Shortcut, misleading BACKTRACK | 26/54 | 32/54 | 32/54 | 32/54 | 32/54 |
| Shortcut, legitimate CONTINUE | 41/54 | 45/54 | 46/54 | 46/54 | 46/54 |
| Bridge, misleading BACKTRACK | 36/54 | 37/54 | 38/54 | 37/54 | 37/54 |

C21 has **no demonstrated recovery advantage over C18-A0** on these measured conditions. The one extra shortcut/CONTINUE completion also occurs under rejected C20-STYLE. The bridge/BACKTRACK result remains one worse than C20-STYLE. This is not a promotion candidate.

## Opportunity-conditioned evaluation: what changed

The fixed-state audit scores every model on **the exact same 1,704 legal-favoured-INK candidate menus** encountered during independently executed C18-A0 traces: 609 BACKTRACK, 620 CONTINUE, 475 no-cue. Unlike comparisons of divergent paths, these probabilities are directly comparable. Under active BACKTRACK advice, the mean preferred-INK probability mass is:

| Model | Preferred-INK mass | BACKTRACK mass |
|---|---:|---:|
| Frozen C12 | 0.250352 | 0.516758 |
| C18-A0 | 0.222057 | 0.586436 |
| C20-STYLE | 0.221810 | 0.586880 |
| C21-RETURN | 0.222661 | 0.584827 |
| C21-OPPORTUNITY | **0.222758** | 0.584640 |

C21-OPPORTUNITY preserves only **+0.0007015 absolute probability mass** relative to A0: around **0.07 percentage points**. Under active CONTINUE advice, its mass is **0.215348 vs A0's 0.216319**, a *regression*. Both candidates and baseline retain exactly 0.422062 preferred-INK mass on the no-guidance reference menus.

A second audit replays all 3,240 recorded trajectories through the same synthetic host, reconstructing the executed action history, protected geometry and live host advice. It counts only times a preferred turn was **actually available**, and separates active advice from after-advice. Active BACKTRACK reference: C18-A0 selected preferred INK in **129/574** opportunities; C21-OPPORTUNITY **133/576**. Because executed paths diverge, these denominators are *not matched-state causal evidence*. After-cue two-sided opportunities are scarce (9 BACKTRACK and 6 CONTINUE for A0), so no robust after-guidance personality conclusion is possible. The more informative same-state probability audit is the primary style metric.

## Safety, custody and limitations

- On the 3,240 held-out episodes: **zero synthetic host-rejected actions**, zero raw illegal top-1, and no protected-cell modifications; execution always applies a legal host candidate.
- C21's **216 cue-off matched episodes per candidate** reproduce C12's full action traces and final ink hashes exactly. Drawing-action logits remain invariant even under guidance; only the host-cue-active BACKTRACK score changes.
- The 148 available local source tests passed. This is **not native CyberSand**, a real Workbench run, an authoritative ABI validation, physical Windows GPU acceptance, or full exact-HEAD CI.
- The checkpoint binaries, complete raw trajectories, prerequisite model chain, C20 source snapshot and C21 full reproduction receipts remain in a manifest-verified external run bundle, **not Git history**. Immutable DRIV3R Actions artifact custody is still not established. The source files in this lane and this ledger are research-only.

**Disposition:** Reject C21-RETURN and C21-OPPORTUNITY for promotion. Retain **C12** as preferred trained checkpoint and **C18-A0** as stronger untrained comparator. The C21 style metric is more interpretable than C20's raw turn-gap, but the tested penalty provides negligible benefit and does not solve misleading-cue robustness or long-horizon personality retention.

**Next discriminator:** construct richer legal branching episodes with verified preferred and opposing INK opportunities *during active correction*, and compare source-matched post-cue path personality with adequate opportunity counts. Avoid another optimization loop on geometry that cannot express the behaviour under test.
