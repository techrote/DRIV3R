# Corpus Factory — counterfactual fork recovery v7

**16 November 2026 · Independent synthetic positive-control experiment · DRIV3R PR #1 Draft/HOLD**

## Research question

Earlier v6 spatial-memory training produced only **1/12 vs 0/12** solved held-out slalom worlds, and had regressions elsewhere. That result did not establish whether action history was *necessary*, versus merely adding parameters. This v7 experiment tests a narrower, deliberately controlled question: **can a real CPU-trained action selector exploit executed-route memory to disambiguate two otherwise identical recovery decisions?**

The generator makes matched pairs of 32×24 corridor worlds. Both have the same 5×5 local material grid, relative goal, heading, and branch legality at the fork. Beyond the visible radius, the goal-connected arm and distant dead-end arm are interchanged. A synthetic host executes a legal, protected-safe excursion down the dead end and returns to the fork. Executed route counts then differ across the pair. The opposite branch is a hard negative.

The dataset has seeded local stone/sand/metal motifs and protected surfaces, with a unique model-visible local state *per pair*. Within each pair, the identical state with different history is **intentional counterfactual supervision**, not an accidental duplicated row. The scenario generator verifies zero duplicate world-geometry hashes, disjoint paired-group seeds across splits, paired-observation equality, and both arm continuations by post-decision graph reachability.

## Fitted models and independently grouped results

Two genuine scikit-learn SGD log-loss classifiers, **24 CPU epochs each**, use identical **129-feature** vectors and identical 512 TRAIN examples. In one arm the two variable history slots are the host-observed visits to LEFT/RIGHT next cells; the capacity-matched sham gets seeded nuisance values identical across the two members of each pair.

| Split and metric | Sham-history arm | Real-history arm | Real-history with inference-time swap |
| --- | ---: | ---: | ---: |
| DEV: branch accuracy, 64 pairs / 128 decisions | 64/128 (50%) | 128/128 (100%) | Not evaluated |
| Held-out: branch accuracy, 256 pairs / 512 decisions | 256/512 (50%) | 512/512 (100%) | 0/512 (0%) |
| Held-out: both members of a pair correct | 0/256 | 256/256 | 0/256 |
| Held-out: chosen branch leads to a goal without revisiting the hub | 256/512 | 512/512 | 0/512 |

There are **256 TRAIN, 64 DEV and 256 held-out independent paired seed groups**, or 1,152 correlated decisions in 576 world pairs. The special 50% sham ceiling is mathematically forced by the deliberately conflicting labels on identical sham-visible paired states. The 100% result is a successful **information-availability positive control**, not evidence that general navigation is solved. The 0% memory-swapped negative control confirms that the actual learned policy relies on the directional memory signal.

The branch decision is applied to the synthetic corridor and reachability of the chosen route is verified with the hub disallowed. The *subsequent route is not autonomously driven by the trained model*. No official DEV, native execution, material simulation, or artistic assessment is claimed.

## Reproduction and provenance

Run from the owned directory:

~~~bash
python -m unittest -v test_counterfactual_fork_v7.py
python counterfactual_fork_v7.py --out /tmp/corpus_fork_v7
~~~

Requires NumPy and scikit-learn. There is no paid API or GPU dependency. The output path is immutable: the producer refuses to overwrite an existing directory. Seven local safety/partition/semantics/reproduction tests passed. Two independent complete executions gave byte-identical JSON and TRAIN-only JSONL outputs.

- Source Git blob: e3fed311a768b5878d31eb8d0f6e124682eba010.
- Test Git blob: ca65a0025ad9759d2ad605d38bb171d987e352ad.
- Result JSON Git blob: e1e540fe3bea091390edd5ce56801dfa2b15dd2c.
- Full JSON report SHA-256: 9f200b13b4338abf78fa9c3276d3537179473562ccf57fbc2dc47cc8900a7407.
- Regenerated TRAIN-only JSONL SHA-256: 311bf2a65e11e3ef1331f1abafbc513d08a40af3a3f6e6e21a29bc7cfcce9828.
- SGD coefficient/intercept SHA-256: memory 5fb38bdb3d7b5504fbab05cc9a9417b4a8c04d979598b83fb6c5062b9af8bc23; sham 705fd5dc34a0a93b773d53b42e9427a657161a0de0f0ae48b5b3c9a63e2b9ad9.
- Each TRAIN line includes a provenance identity, observed-prelude trace hash, full source world hash, synthetic-only evidence designation, and legal opposite-branch hard negative. DEV and held-out messages are not exported to training.

**Restrictions and limitations:** This is a simplified branch-choice subsystem: host menus include RETRACE/STOP, but the benchmark selects only LEFT/RIGHT. Its full executed prelude is scripted, its two-sided label construction is intentionally idealized, and its held-out seeds use the same generator grammar. The distant protected-material field diversifies geometry but is not a navigation challenge. Do not mix these paired artificial labels into general-purpose gameplay training without a designated curriculum gate and downstream independent-world tests. The earlier v6 negative/mixed results remain valid. The generated corpus is locally hash-verified, **not yet an immutable Actions artifact**. No model binary, game file, or production setting is committed.

**Follow-on:** Replace the scripted fork with adversarially sampled host-applied recovery episodes and a *distinct held-out grammar*; evaluate complete model-controlled trajectories, protected-cell legality and geometry-generalization separately. Any canonical host-ABI implications belong in a read-only CyberSand interface proposal, not a game edit.
