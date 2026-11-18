# C65–C66 original evidence-custody restoration (18 November 2026)

**Scope:** Restore exact, previously omitted synthetic-host evidence to the existing Draft/HOLD guidance-recovery laboratory. No new model promotion, no native CyberSand test and no game-side changes.

## Source and archive integrity

The original owner-evaluation C65 and C66 ZIPs supplied in this conversation were unpacked locally. Every restored evidence file matched its original `MANIFEST.sha256.json` entry **before** upload. For both studies, the compressed JSON ledger was decoded and its embedded full `summary` was compared for exact structural equality with the separately archived summary. SVG galleries were checked as rendered-source SVG text. The already committed `c65_paired_delayed_credit.py`, `test_c65_paired_delayed_credit.py`, `c66_holdout.py` and `test_c66_holdout.py` all independently matched the original archive Git blob SHA-1 values, so the existing source files were not changed.

| Previously missing evidence | Original file SHA-256 |
|---|---|
| C65-summary.json | `fcaaf5581428ad09423184af65d11459af073d65ab456e8a3f8097bbb5d87106` |
| C65-executed-ledger.zlib.b64 | `7f899475cfa1f317468855ee9cbeacd3df664fd7ccbd0a8d9b380285473eee8e` |
| C65-gallery.svg | `cab9b9b4d0ab709ad6c75a61d37e27959d5deb3dc7daac8477a31bc4f0257a42` |
| C66-summary.json | `9da6e83551c9bd2f8472576a6c391bd13effb862eea6a50bb53c119b029f6df0` |
| C66-executed-ledger.zlib.b64 | `fbaee7a28a2dd6f8e51ed41acf665f40f4b4efba1386c2cbd42d5093b80802d6` |
| C66-gallery.svg | `5b96e66b1f9c946fca1948ae33e87ba76abc6a1e5fa2c34d9099b465de265188` |

Original substantive negative outcomes remain unchanged:

- **C65:** 4,184 action-matched training samples per head, 48 AdamW updates per head. ALIGN and SHIFT both 10/24 goals on each memory arm, versus non-neural FRONTIER and frozen C60 at 12/24. The C65 ledger includes 72 original condition records.
- **C66:** independently fresh 24-graph transfer cohort, 720 executed synthetic-host suffixes. C65 ALIGN 27/48 versus wrong-credit SHIFT 29/48 per memory condition. The C66 ledger contains all 120 matched condition records and 36 labeled full paths.
- No protected-cell mutation or host-action rejection was reported in the original executed records. No new optimizer iterations were performed by this *custody* restoration.

To decode a ledger: `json.loads(zlib.decompress(base64.b64decode(Path(...).read_text())))`. The C67 cue-strength transfer experiment now has its exact C66 training ledger retained alongside its own source and evidence.

**Limits:** Source and evidence transfer is not full scientific reproduction of the original C65 or C66 training. This repair does not provide a self-contained checkout (frozen predecessors and original C65 checkpoint binaries remain in the owner archive), native CyberSand validation, or immutable Actions custody of the two original checkpoint binaries. All checkpoint binaries remain outside Git history. No promotion, merge or Ready transition is authorized.
