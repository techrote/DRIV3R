# C20 — host-guidance personality hard gate (16 November 2026)

**Research-only; negative result. Neither C20 head is promoted.** DRIV3R consolidated PR #1 remains Draft/HOLD, with no CyberSand/Neurounner mutations. C12 remains the preferred trained checkpoint; the untrained C18-A0 remains the stronger host-attestation control.

## Question and implementation

Can a learned, locally observable penalty for suppressing personality-favoured ink turns improve active-guidance style while retaining C18-A0 recovery success and resistance to false cues?

Two 32→12→1 heads start at the same frozen C18-A0 (+1 BACKTRACK-logit) policy. Both optimize actual paired, host-executed future-return advantages. The **STYLE** arm adds a differentiable cost on positive BACKTRACK adjustments where there are host-legal personality-preferred INK opportunities and the paired futures are near-ties. **RETURN** omits that cost. Neither changes any other candidate score or legal permission. No hidden global map or future-outcome labels enter inference.

## Executed bounded CPU experiment

- **360** matched candidate menus, with no/weak/strong scalar hint arms, signed BACKTRACK/CONTINUE advice, both personality settings and hairpin/staircase/shortcut/bridge training worlds.
- **1,440** counterfactual synthetic host-executed suffixes: **138** favour BACKTRACK, **44** favour forward and **178** are near-ties; no host-rejected training actions.
- **192 CPU AdamW updates per head (384 total)**. RETURN loss +0.000082 → −0.000856; STYLE +0.000081 → −0.000596. Two full optimizer replays produced identical weight tensors for each head.
- RETURN checkpoint SHA-256: `564dbbc4a956fe41937dec68093813940329af3ebc32ac801673daf2bf628abc`.
- STYLE checkpoint SHA-256: `bd358be59c842f386c691e9434f2e9c45df2e52ef3ca1ee518f1c1a135d3a0a7`.

The fully executed trainer, frozen checkpoint prerequisites, complete source dependencies, raw held-out trajectories, metrics, HTML/SVG gallery and SHA-256 manifest reside in the external owner archive `driv3r_guidance_recovery_C20_personality_hard_gate_2026-11-16.zip` (archive SHA-256: `ba49e25edc7817bcde94b37e90e97cc099d9d3c7ba26fc539e4ab079aa192cbb`). Model binaries are **not committed to Git**, and immutable Actions artifact custody remains outstanding. Existing C19 repository inference source was not rewritten; the complete experiment depends on the archived synthetic C19 host/replay harness.

## Held-out results

Independent seed **221000**, horizon 48, with **2,304** host-executed stochastic episodes (576 per model) and withheld hook/zigzag plus rotated shortcut and bridge geometries. Separate pilot seed **211000**, **864** episodes. Successful goals out of 48 per row:

| Situation | Frozen C12 | Untrained C18-A0 | C20-RETURN | C20-STYLE |
|---|---:|---:|---:|---:|
| Reverse / valid BACKTRACK | 35 | 39 | 39 | 39 |
| Reverse / misleading CONTINUE | 38 | 45 | 45 | 45 |
| Shortcut / BACKTRACK | 26 | 29 | **28** | 29 |
| Bridge / BACKTRACK | 36 | 37 | 38 | 38 |
| Shortcut / CONTINUE | 40 | 46 | 46 | 46 |
| Bridge / CONTINUE | 40 | 44 | 44 | 44 |
| Ahead / misleading BACKTRACK | 45 | 45 | 45 | 45 |
| Ahead / valid CONTINUE | 37 | 37 | 37 | 37 |

STYLE adds one bridge success against A0, with no change in the other seven reported conditions. RETURN loses one shortcut completion. **There is no convincing improvement over A0.**

## Active personality gate — failed

The measured gap is the normalized signed-turn bias for personality +1 minus that for −1, computed separately for each topology/advice stratum. A0 → C20-STYLE:

- Reverse / misleading CONTINUE: **+0.205 → +0.000**; total observed turning events **16 → 12**.
- Reverse / valid BACKTRACK: **+0.268 → +0.250**.
- Shortcut / valid CONTINUE: **+0.326 → +0.197**.
- Bridge / misleading BACKTRACK: **−0.011 → −0.091**.

These style measurements involve sparse event counts and should not be mistaken for definitive stylistic semantics. Nevertheless, the hard personality gate **fails**: a small completion improvement does not justify the observed deterioration in active-guidance character. Both heads are rejected for promotion.

## Safety, source scope and unresolved qualification

All **192 cue-off held-out paired traces per candidate** matched C12 exactly, including ink hashes. Held-out synthetic host rejections **0**, illegal raw top-1 **0** and protected-material mutations **0**. Source/menu/seed comparisons are initially matched; different policies no longer remain same-state comparisons after divergence.

This is **external synthetic CyberSand-like research**, not live Workbench or native actor acceptance. No actual native painting/Undo, legal ABI replacement, production provider modification or paid GPU/API cost. Complete repository exact-HEAD tests, immutable GitHub Actions checkpoint custody and physical evaluation remain outstanding.

**Disposition:** negative result retained, no model promoted. Next investigate opportunity-conditioned multi-turn style measurement and an explicit policy-level style constraint rather than protecting only non-BACKTRACK logits.
