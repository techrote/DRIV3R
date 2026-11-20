# POSTMERGE-REPLAY-001 — merged-source science qualification

**Scope:** CPU-only replay of archived MATERIAL-EPOCH-001 evidence after the owner merged DRIV3R PR #1. New code and evidence are restricted to the long-horizon laboratory. This is not new model training or native gameplay.

- Merged main source: 85df84666a8bad18f0092b930c08557b8010687f.
- Original external ZIP SHA-256: e6514b747af8c271f110d2da50ff3570731cedf44d4f1f5f46ab922b64bd5fa0.
- Reconstructed dependencies were independently checked against their merged-main Git blob identities before execution; replay's original source and checkpoint SHA-256 assertions passed.
- Full original evidence: **320/320 synthetic DEVELOPMENT episodes**, **16,140 host-legal action receipts**, **8,542 neural probability vectors** independently re-executed from restored original checkpoints.
- Unchanged original goal completion: neural 32/64; fixed24 47/64; fixed24_epoch 47/64; event_only 39/64; each_stage_epoch 48/64.
- Replay took 5.532 seconds on the first local four-worker run and 6.197 seconds using the new wrapper; wrapper wall time 6.237 seconds and peak process RSS 134,844 KiB.
- New tests: **6/6 PASS** with original external ZIP injected. When the archive is absent, source-only tests pass and the complete replay explicitly SKIPS.
- Model-specific latency was not separately re-benchmarked. Training updates: zero. Native CyberSand receipts: zero. Held-out worlds: sealed.

## Reproduce in a DRIV3R checkout

Set DRIV3R_MATERIAL_EPOCH_ARCHIVE to the absolute path of the historical evidence ZIP. Execute:

    python -m unittest experiments.driv3r.training.long_horizon.test_postmerge_archive_replay -v
    python -m experiments.driv3r.training.long_horizon.postmerge_archive_replay --archive /path/to/DRIV3R_MATERIAL_EPOCH_20261118_QUALIFIED.zip --workers 4

The archive is external and contains model binaries; no checkpoints, paid model calls, workflows or game code are committed here. The wrapper checks the local merged research code and original source identity, archive safety, checkpoint provenance, compressed receipt hashes, model distributions, host-legal actions and final material states.

## Scientific limitations

This qualifies one previously retained synthetic DEVELOPMENT campaign against source identical to the merged main snapshot, **not** the entire historical import or any native CyberSand behaviour. Scripted material changes did not actually invalidate the imminent cached action. Original negative conclusion remains intact: unconditional material-epoch invalidation performed extra searches without improving completion in those fixtures.

Next experiment: directly obstruct future cached routes across substantially different topologies, with lazy legality, full-footprint and unconditional baselines. Do not open structural HELDOUT until separately authorized.
