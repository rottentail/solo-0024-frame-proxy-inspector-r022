# Random-access frame fetch experiments

These experiments decide **what the random-frame-fetch promise should be**
for compressed material with reordered packets and variable display
intervals. They do not implement a format adapter and never import
`src/`. The environment has no ffmpeg and the service allows no
third-party packages, so a small, fully inspectable laboratory codec
(`labcodec.py`) models the exact stream properties that make random
access hard:

- `I` packets are independently decodable random access points (RAPs);
  `P`/`B` packets carry explicit reference lists and cannot be decoded
  without their references;
- packets are stored in **decode order**, which differs from **display
  order** whenever B packets are present;
- presentation timestamps (PTS) are independent of packet order, so they
  can be non-uniform (VFR) and can repeat for two distinct pictures.

Every ground-truth picture has a content hash, so a fetch is proven
correct by comparing the returned hash to the manifest instead of by
visual inspection.

## Run

```bash
python3 -I experiments/random_access/run_all.py
```

Each script can also be run on its own, e.g.
`python3 -I experiments/random_access/exp2_shared_pts_identity.py`.
Reports land in `generated/*.json`; the `.lab` files are the generated
short fixtures and are safe to delete and regenerate.

## Experiments

| Script | Question answered | Key evidence |
| --- | --- | --- |
| `exp1_reorder_provenance.py` | With decode order != display order, does each strategy return the exact requested original, and what does it cost? | RAP chain is exact for every requested ordinal; one full pass decodes 24 pictures for the batch while the grouped RAP chain decodes 10; per-cell head replay decodes 71 across the same six cells. |
| `exp2_shared_pts_identity.py` | When two display pictures share a PTS and intervals vary, can the original still be identified? | Ordinal-keyed RAP fetch returns both distinct originals (different hashes) at the shared PTS; nearest-PTS returns ordinal 8 when ordinal 9 was requested. |
| `exp3_rap_coverage_gate.py` | What happens when usable RAPs disappear or a reference breaks? | A structural gate admits bounded RAP coverage, rejects a 24-long chain (`chain_length_exceeds_bound`), and rejects a broken chain (`pictures_without_reachable_rap`); on head-only material "random access" costs exactly one full replay. |
| `exp4_contact_sheet_flows.py` | How do the three contact-sheet flows compare end to end per scene? | Admitted fixtures keep every cell exact with worst-cell decode inside the bound; head replay is strictly more expensive; nearest-PTS is inexact on the shared-PTS fixture. |

## Strategies under test

- `sequential_full` — one decode pass from packet zero, caching requested
  pictures. Source-accurate; bounded by a single full-file pass.
- `rap_chain` — group requests by the RAP their reference chain starts
  at, seek once per RAP region, decode only the union of needed chains.
  Source-accurate and bounded per cell **when the admission gate passes**.
- `head_replay` — restart the decode for every cell. Source-accurate but
  quadratic; the pattern the design forbids for contact sheets.
- `nearest_pts` — answer by closest timestamp instead of the requested
  original. Cheap but provably wrong on repeated/gapped PTS.

The admission decision lives in `admission.py` and is purely structural:
it walks declared references without decoding, so it can run before any
expensive media work, matching the service's "prove coverage before
publication" stance.
