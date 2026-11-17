# GAL-017 — Pretrained SLM action-language transfer: executable gate, blocked optimizer

**Research-only, 17 November 2026.** Ownership: DRIV3R Draft/HOLD PR #1; this entire increment stays under `experiments/driv3r/training/action_language/**`. No CyberSand/Neurounner writes, no game/native acceptance, no new workflow, no external paid compute.

## Why this is not another surrogate increment

GAL-015's deterministic route scaffold (144/144 synthetic completions) and GAL-016's fixed-memory/dynamic-world ablation (81-cell memory: 25/64 static, 24/64 pulse; unlimited: 64/64 static, 57/64 pulse) did **not** establish a trainable SLM policy. The algorithmic route oracle is useful as a teacher/control, not a learned answer.

GAL-017 therefore adds a runner targeting **actual** `EleutherAI/pythia-70m-deduped`, exact immutable revision `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c` (the model already studied in the independent Model Zoo lane). The runner uses genuine PEFT rank-4 QKV LoRA training with AdamW, token-level host-menu action loss, fresh adapters for one-cell and 1–4-cell swept vocabularies, and paired **stock-versus-adapted** closed-loop rollouts on held-out tree/comb/deep-tree/path worlds with host-controlled legal moves and transient obstructions. A bounded ledger supplies 81/161 material cells and current radius-4 crop. Neither the trainer nor the model accesses hidden world geometry for action choice. All scoring IDs come from the current host offer; stale leases and unoffered actions remain rejected by the host.

**Actual model training did NOT execute in this environment:** `transformers` and `peft` are unavailable, pretrained model weights are not mounted, and the container cannot reach GitHub/Hugging Face directly. The intentional `--out /mnt/data/GAL017-offline-blocked --updates 1 --max-decisions 12` attempt produced a `BLOCKED_OR_FAILED` report before loading a model or performing an optimizer update. This is the precise resource blocker, **not** a claimed trained checkpoint. An opt-in remote CPU model runner and immutable artifacts are needed. This lane is not authorised to edit `.github/workflows/**`, nor is a workflow-dispatch action exposed in the connected GitHub interface. Do not silently substitute a random small network or claim the oracle is an SLM.

## Offline qualification, not neural results

- Seven focused source tests passed, including host legal menus, stale/illegal rejection, protected material invariants, hidden distant-material invariance of observation prompts, disjoint train/dev/final seeds, and failure-closed unavailability of the pretrained runtime.
- `python gal017_pythia_adapter.py --self-test` passed for one-cell and 1–4-cell swept actions under both static and pulsed obstacles; **0 model loads and 0 optimizer updates**.
- Preflight-only observation-limited **deterministic oracle** on two DEV seeds × four topologies × two dynamics × two vocabularies (32 synthetic episodes, 48-decision cap): one-cell completed **2/16**, mean target coverage **0.6680**; swept completed **4/16**, mean coverage **0.8099**. These are *not* SLM performance metrics, nor independently tested artistic intent. The 48-decision cap disadvantages one-cell progress on 48/64-cell tasks; follow-up must compare equal decisions and equal ink budgets.
- Original per-world oracle metrics, hashes, source fixture pins and explicit negative-state classification are in `GAL017_PREFLIGHT.json`. Existing GAL-016 results remain unchanged and are historical controls only.

## Execute the real test, under a pre-existing opt-in CPU runner

From repository root, using a **pinned pretrained model download/cache** and an independently approved bounded CPU environment with `torch`, `transformers`, `peft` and `huggingface_hub`:

```bash
python experiments/driv3r/training/action_language/gal017_pythia_adapter.py --self-test
python experiments/driv3r/training/action_language/gal017_pythia_adapter.py \
    --out /tmp/gal017-original-pythia70m --updates 24 --max-decisions 48 --final-seeds 1
```

The script verifies model revision and tensor SHA-256 before training, verifies each of 36 offered-code tokens is atomic, rejects truncated action catalogs, applies host-only actions with unchanged lease tokens, captures per-decision inference latency, RSS, raw unmasked preference errors, executed-illegal rate, source/fixture/checkpoint hashes, trained adapter weights and held-out original trajectories. It refuses to overwrite an existing output directory and caps optimizer updates (1–64), decisions (8–80), learning rate and held-out seed count (1–2). **All these execution claims describe checks in executable source, not yet executed pretrained-model results**; any run-time incompatibility is a negative finding to preserve.

Store `/tmp/gal017-original-pythia70m/**` as **one immutable GitHub Actions artifact**, with the Actions run ID, exact PR head and file hashes entered in the lane-local handoff. No model files in Git. Once the first held-out evaluation is complete, reproduce on `--final-seeds 2` only as a distinct, explicitly identified follow-on; do not rebrand prior inspected seeds as blind. Record CPU p50/p95 latency, peak RSS, legal and raw-unmasked preferences, art quality caveats, completion, actual calls and ink/decision separately. Compare trained SLM to its own frozen stock baseline, as well as the observation-limited deterministic oracle. No native integration is implied.

## Interface handoff (proposal only)

For future CyberSand game-side review, keep *host source of truth* for `{world_epoch, lease, cursor, heading, local_material_crop, protected_crop, task_start_policy, target_ink_budget, legal_action_catalog, action_paths, budget}`. Model receives a **compact projection** plus bounded episodic memory and selects **one opaque offered action ID with the unchanged offer token**. Host revalidates token and every swept cell before executing, returns receipts with changed material and revised menu/epoch; observations invalidate remembered blocked cells. STOP is always offered; no native camera, stroke implementation, physics or provider change is requested. Native limits must be checked separately against CyberSand's current canonical ABI, **not** inferred from the frozen DRIV3R host fixture.

**Research status:** Source-ready / neural training blocked by pretrained runtime and opt-in execution path. Draft/HOLD, do not merge or promote.

## GAL-017a loader correction (17 November 2026)

A deterministic blocker was found in the published neural runner: `huggingface_hub.cached_file` is **not a public Hugging Face Hub API** and is absent from the installed Hub version. Even with Transformers, PEFT and weights provisioned, this would have raised an ImportError, incorrectly resembling unavailable dependencies. Source now uses `huggingface_hub.hf_hub_download(repo_id=..., filename='model.safetensors', revision=MODEL_REVISION, local_files_only=True)` for byte-level pinned snapshot hashing. A dependency-free fake-snapshot regression test exercises the full import-and-load control flow, verifies pinned revision and cache-only access, and rejects the former import error. Fake tensor bytes are *only* a loader contract fixture; no SLM or adapter was trained or evaluated.

The external resource gate remains: pretrained tensor snapshot plus Transformers and PEFT in a bounded CPU runner, and a permitted manual opt-in workflow path. No new workflow was created. The preceding GAL-017 oracle preflight is unchanged and remains non-neural evidence.
