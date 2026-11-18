# C49 — Mid-trajectory, host-issued recovery/goal correction

**Status: RESEARCH HOLD; learned C49-ALIGNED is not accepted for promotion.** 17 November 2026.

**Ownership:** only `techrote/DRIV3R` PR #1, branch `research/import-consolidated-worm-labs`; all new source and evidence under `experiments/driv3r/training/guidance_recovery/**`. No native CyberSand, Neurounner, or DRIV3R `main` modifications. No paid external compute.

## Question and causal design

After C40–C48 repeatedly produced promising offline predictions but unreliable goal recovery, test whether labels on *actually encountered intermediate states* contain more discriminative information than episode-start labels. A candidate model must recognize when the frozen `C34-RESET` response to an authenticated `BACKTRACK` cue is beneficial relative to a cue-visible `C18-A0` policy without RESET's additional state machine. To avoid implying the cue is absent for C18, this is a comparison of *recovery controllers receiving the same cue*, not a clean guidance-versus-no-guidance causal estimate.

C49 executes 9 or 19 further ordinary host-legal stochastic baseline actions from each independently generated maze world. **Only the host** then changes or reissues a reachable goal and sets its explicit goal-change/recovery context. The target is `second_goal` when unvisited; otherwise the original still-unvisited goal is reissued. No teleportation, protected-cell edit or preloaded fake prefix history is used. This is a synthetic, host-controlled event, not a change to the CyberSand ABI. On each resultant state the baseline, RESET and frozen C40 controllers run separately on deep copies of the same world under matched cue arms. The outcome label is `goals(RESET) - goals(C18-A0)` over a 78-step suffix.

The 19-float inference vector consists of synthetic host-visible state/cue properties, legal candidate statistics and frozen C40 action-score dispersion. It does **not** contain future goal outcomes, full maze topology, graph IDs, protected-cell locations or oracle shortest paths. Inference only chooses between two previously qualified host-legal controllers; it cannot create a new action or bypass the host legality menu. Non-`BACKTRACK` cues make the selector abstain to C18-A0. All long-horizon results are host-executed, not predicted oracle returns.

## Training and controls

- **64 generated training graph identities**, of which **46** contributed at least one nonterminal mid-trajectory state; 69 usable states, 138 paired training examples. Frozen DFS/braid/Prim/Prim-braid generator families, disjoint from blind and online graph hashes. The training targets have **21 positive, 15 negative, 102 tied** outcomes: 36/138 (26.1%) discriminate, a material improvement over episode-start C47's 18/192 (9.4%).
- Two 19→12→1 heads initialized from identical parameters: `C49-ALIGNED` trained against actual labels and `C49-SHUFFLED` trained on source-matched, nonidentity label permutations preserving the global label histogram. Both underwent **24 AdamW epochs** (training update counts in `C49-summary.json`), and independent repeated optimizer runs checked byte-identical parameters and loss receipts.
- Fixed inference threshold: logit > 0.75 selects RESET under an authenticated backtrack cue; otherwise select A0. No tuning against online goal outcomes. Also compare original C18-A0, C34-RESET, frozen C40-PAIR, deterministic random RESET admission and always-abstain controls.
- Blind set: **8 independent Wilson/Kruskal graph identities**. Online: **24 further graph identities** (16 in-family; 8 Wilson/Kruskal), 27 usable prefix states and four cue arms per state (no guidance, backtrack 0.35, backtrack 1.0, continue 1.0), **756 policy-suffix episodes** across seven controllers. Only 81 of the 108 per-controller episodes carry nonzero cue strength. No graph identity is shared among train, blind and online sets.

## Actual held-out goal recovery

| Policy | Goals on 81 guided suffixes | Condition groups worse than RESET | RESET choices out of 81 |
|---|---:|---:|---:|
| C18-A0 | 21 | 3 | 0 |
| C34-RESET | 23 | 0 | 81 |
| Frozen C40-PAIR | **32** | **0** | not applicable |
| **C49-ALIGNED** | **23** | **1** | 13 |
| C49-SHUFFLED | 21 | 3 | 4 |
| C49-RANDOM | 20 | 3 | 14 |
| C49-CONSERVATIVE | 21 | 3 | 0 |

The C49-ALIGNED improvement over shuffled is **+2 goals** concentrated on **two positive graph clusters**, with a descriptive graph-resampling 95% interval **[0,+5]**. Its difference versus RESET is **0 goals**, interval **[−3,+3]**. The aligned model still fails the strict no-regression condition gate. It does **not** establish a reliable generalizable policy advantage; the interval against shuffled is exploratory and selected after a long sequence of experiments, not a preregistered confirmatory test.

On the independent blind Wilson/Kruskal corpus, C49-ALIGNED selected RESET 3 times and incurred **−2 actual goals relative to A0** on those selections; shuffled selected RESET zero times. This is a directly negative cross-generator transfer result.

A separate, **not newly trained** comparator emerges as the strongest mid-trajectory performer here: C40-PAIR completed 32 goals versus RESET's 23 (a +9 difference across 24 graph clusters, four favorable and no unfavorable; graph-bootstrap descriptive interval **[+1,+19]**). The advantage occurs on DFS/Prim families; both frozen controllers tie on Wilson and fail on Kruskal. This finding is an exploratory *specific host-state enrichment effect*, not evidence of CyberSand-native or broad model superiority. Repetition with more independent graph seeds and different goal changes is essential before interpreting it as a genuine transfer improvement.

The suffix horizon is locally restarted at zero after the host event (including a fresh deterministic random stream). This is matched counterfactual suffix evaluation, **not uninterrupted continuous replay of the original 78-tick episode**. That limitation matters: a policy can benefit from resetting its stochastic stream and local control clock.

## Safety, personality and reproducibility

All 756 evaluated suffix episodes had **zero synthetic host action rejections**, zero raw illegal top-choice proposals and protected-set hashes preserved by the synthetic executor. All **27 no-guidance suffixes per model** were byte-identical to C18-A0 for both movement-action and painting hashes. Model decisions under strong cues do change active trajectories; these facts do **not** prove whole-trajectory personality preservation. Preferred/opposed painting and bounded C40 personality-shift summaries are retained in the compact results. There is no native CyberSand physical acceptance, no new canonical host ABI, no change to the frozen `d_common` interface, no immutable GitHub Actions checkpoint artifact custody, and no paid GPU/API work.

The source includes a bounded reproducible CPU-run command and an independent replay test. The retained gallery shows **18 synthetic-host executed suffix paths** following legal prefixes; replay reconstructed all 18 recorded source and host action/ink hashes, preserving protected masks. Neither the gallery nor the simulator constitutes an image of native CyberSand.

### Run recipe (from the complete local archive)

```bash
# Requires numpy, CPU torch and safetensors. Unpack frozen C47 prerequisites:
mkdir -p prior
(cd prior && unzip -oq ../frozen/C47-archive.zip)
export PYTHONPATH="$PWD/source:$PWD/prior/source:$PWD/prior/previous/C45-source"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python source/c49_midtrajectory_correction.py prior output --train-worlds 4 --online-worlds 1
python -m unittest source/test_c49_midtrajectory_correction.py -v
python source/c49_trajectory_gallery.py output/C49-curated-paths.json output/C49-gallery.svg
```

Existing checkpoint receipts allow bounded continuation after interruption; for scientific replication start with a new empty output directory and compare the training/checkpoint hash receipts. Frozen upstream checkpoint archives have not been promoted into Git history. See the local manifest-verified archive and committed SHA-256 receipt for provenance. The repository alone cannot reproduce full inference from source without the archived predecessor checkpoints.

**Disposition:** HOLD C49-ALIGNED; no promotion or merge. Prioritize repetition of the frozen C40 mid-trajectory advantage across much larger independent Wilson/Kruskal/Prim cohorts, followed by genuine continuous-episode recovery, active-personality and native CyberSand gates. Do not build another selector on top of this thin result.
