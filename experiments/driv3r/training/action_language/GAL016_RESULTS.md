# GAL-016 — Finite material memory and transient host obstructions

Date: 2026-11-16. Research owner: Game Action Language & Embodiment Lab.
Scope: techrote/DRIV3R Draft/HOLD PR #1, experiments/driv3r/training/action_language/** only.

## Frozen-weight experiment

GAL-015's route-to-frontier scaffold was evaluated with finite material-cell LRU memory and transient external-host obstructions. Frozen GAL-011 memory and GAL-015 trained paint scorers were reused. **No optimizer updates were performed in GAL-016**: this is a matched failure-mode evaluation, not a trained new checkpoint.

Four memory capacities (81, 161, 321, unlimited cells) crossed with static/pulsed host material, four topology families (path, tree, comb, deep_tree), 48/64-cell targets and eight unseen seeds (2640000–2640007) give **512 held-out episodes**, 64 per condition/capacity panel. Only radius-four material crops and the current host-leased legal action IDs reach the model. The synthetic host can temporarily block an unpainted, nonprotected corridor cell after decisions 8 and 26, releasing at 14 and 32. It does not reveal hidden geometry or event locations to inference.

## Exact target completions

| Host material | 81 cells | 161 cells | 321 cells | Unlimited |
|---|---:|---:|---:|---:|
| Static | 25/64 | 35/64 | 58/64 | **64/64** |
| Transient obstruction | 24/64 | 33/64 | 51/64 | **57/64** |

Unrestricted versus 81-cell capacity: static **+60.94 percentage points** (seed-cluster exploratory 95% bootstrap interval +56.25 to +65.63); pulsed **+51.56 points** (+43.75 to +59.38). The host-controlled pulse reduces unlimited-memory completion by seven episodes. Most failures occur in long-limbed comb/deep-tree cases, not straight paths.

Lower model-call proxy counts with small memories reflect **premature stopping**, not genuine inference savings. Neither native latency nor artistic quality was measured.

## Source, receipts, and qualification

Five dedicated tests passed, covering protected-cell safety, stale leases, bounded observations, invalidation and reproducibility. **512/512** saved episode rows matched a fresh execution; **512/512** action streams were independently replayed through a fresh external synthetic host with matching exogenous event logs, material hashes and receipts.

- Frozen GAL-011 checkpoint SHA-256: 97f88f0349de134c3c430997f998a8534d1a2b62e372ba81970f85e12baf61d1
- Trained GAL-015 paint checkpoint SHA-256: 5a158ec70292813b3b3d881f544c5bf7135f89825f7d695e17eeacabe699b94d
- Original 512-episode JSON SHA-256: af989e150d6e15dd4fd9e6b902f47d502c75b2466504c85db62deb6be6bc51dd
- Original 16-gallery-trace JSON SHA-256: 75385a6affcbd4eba655544eb34fae93d2d1b6df4981c287c1cc4d7bdd1d5d6e
- Original metrics JSON SHA-256: aec0bb16169cd9aaab6a78b792f055450e93f4c7e24a589c24ef3eb6b35b36ba
- Original SVG trajectory gallery SHA-256: 3c0aa24a984e78861284e983d93e5e4159464ccb0a9abadfb7880860b506d7da
- Original paired-bootstrap JSON SHA-256: 78ca147bb4c33b4b7a60c8e89010d2531b99b2b980c6667b839bb8d735142a20
- Verified dependency Git blob IDs: lab.py 24ad6cb0; frontier_memory.py f37f45ec; material_corridors.py 974048b3; route_scaffold.py 0b61cdd9. Newly committed GAL-016 Python sources match executed local blobs.

The large original receipt/gallery payloads are preserved in the experiment evidence archive, but **have not been published as durable GitHub Actions artifacts**. SHA-256 entries above are provenance pointers, not proof of accessible artifact hosting. No checkpoint binaries are in Git. All findings are from a synthetic external host, not CyberSand Workbench, W0, resident SLM inference, native acceptance or externally assessed aesthetics. Fixed intervention timing and only eight seed clusters limit generality.

## Next research decision

Retain the GAL-015 scaffold as a research baseline; reject 81/161-cell raw LRU caches for long-horizon branching. Compare a compressed versioned topological frontier graph against raw material LRU at the **same byte budget** under newly unseen topology and changing materials, without modifying the game ABI.

PR #1 remains Draft/HOLD and unmerged. CyberSand and Neurounner remain read-only.
