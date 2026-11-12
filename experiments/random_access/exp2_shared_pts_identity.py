"""Experiment 2 - original picture identity under shared and gapped PTS.

Fixture properties:
* two distinct display pictures share one PTS (a repeated display
  timestamp), so PTS alone cannot identify an original picture;
* one PTS gap models variable display intervals;
* packets are still stored in decode order with B reorder.

The experiment proves:
1. RAP-chain fetch keyed by the *display ordinal* returns each distinct
   original picture even when two of them share a PTS;
2. a nearest-PTS heuristic returns the wrong picture for the repeated
   timestamp, which the content hash exposes;
3. decode order differs from display order throughout.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fetchers import (  # noqa: E402
    ground_truth_hash,
    nearest_pts,
    rap_chain,
    sequential_full,
)
from labcodec import StreamConfig, build_manifest, load_container, write_container  # noqa: E402
from reportutil import banner, check, summarize_costs, write_report  # noqa: E402

GENERATED = Path(__file__).resolve().parent / "generated"


def main() -> int:
    banner("experiment 2: shared PTS and variable display intervals")
    # Ordinal 8 and 9 share a PTS (duplicate at 9 shifts it back); ordinal 14
    # gets an extra step (a gap) through vfr_extra at 14.
    config = StreamConfig(
        scene_lengths=(6, 6, 6, 6),
        gop=6,
        b_count=1,
        open_gop=False,
        anchor_is_rap=True,
        vfr_extra=(14,),
        duplicate_at=(9,),
    )
    manifest = build_manifest(config)
    fixture = GENERATED / "exp2_shared_pts.lab"
    write_container(manifest, fixture)
    container = load_container(fixture)

    pts_by_ord = {entry.display_ord: entry.pts for entry in container.display}
    hash_by_ord = {entry.display_ord: entry.hash16 for entry in container.display}
    print("PTS map (ordinal -> pts):")
    print({ordinal: pts_by_ord[ordinal] for ordinal in sorted(pts_by_ord)})

    # 1. A shared PTS must map to two *distinct* original pictures.
    shared = {
        pts: ordinals
        for pts, ordinals in container.pts_to_ordinals.items()
        if len(ordinals) > 1
    }
    check(shared, "fixture must contain a repeated PTS")
    repeated_pts, repeated_ordinals = next(iter(shared.items()))
    check(
        len(repeated_ordinals) == 2
        and hash_by_ord[repeated_ordinals[0]]
        != hash_by_ord[repeated_ordinals[1]],
        "pictures sharing a PTS must still be distinct originals",
    )
    print(
        f"repeated PTS {repeated_pts} -> ordinals {list(repeated_ordinals)} "
        f"with distinct hashes"
    )

    # 2. VFR gap must be visible in the PTS deltas.
    deltas = [
        pts_by_ord[index] - pts_by_ord[index - 1]
        for index in range(1, len(container.display))
    ]
    check(max(deltas) != min(deltas), "fixture must contain variable intervals")
    print(f"PTS deltas: {sorted(set(deltas))}")

    # 3. Ordinal-keyed exact fetch of both pictures that share a PTS.
    requested = (repeated_ordinals[0], repeated_ordinals[1], 14, 20)
    rap = rap_chain(container, requested)
    seq = sequential_full(container, requested)
    for batch in (rap, seq):
        for outcome in batch.outcomes:
            check(
                outcome.returned_hash
                == ground_truth_hash(container, outcome.requested_ord),
                "ordinal-keyed fetch must return the exact requested original",
            )
    check(
        rap.outcomes[0].returned_hash != rap.outcomes[1].returned_hash,
        "the two shared-PTS cells must show different originals",
    )

    # 4. Nearest-PTS heuristic cannot disambiguate the repeated timestamp.
    intended = requested
    pts_requested = tuple(pts_by_ord[ordinal] for ordinal in intended)
    nearest = nearest_pts(container, pts_requested, intended)
    mismatches = [
        {
            "requested_ord": outcome.requested_ord,
            "returned_ord": outcome.returned_ord,
        }
        for outcome in nearest.outcomes
        if not outcome.exact_match
    ]
    check(mismatches, "nearest PTS must fail at least one disambiguation")
    print(f"nearest-PTS impersonations: {mismatches}")

    costs = {
        "rap_chain": summarize_costs(rap),
        "sequential_full": summarize_costs(seq),
        "nearest_pts": summarize_costs(nearest),
    }
    print("costs:")
    for name, summary in costs.items():
        print(f"  {name:15s} {summary}")

    report = {
        "fixture": str(fixture.relative_to(fixture.parents[1])),
        "pts_by_ordinal": pts_by_ord,
        "repeated_pts": repeated_pts,
        "repeated_pts_ordinals": list(repeated_ordinals),
        "repeated_pts_distinct_hashes": [
            hash_by_ord[ordinal] for ordinal in repeated_ordinals
        ],
        "pts_deltas_unique": sorted(set(deltas)),
        "requested_display_ordinals": list(requested),
        "rap_chain_exact": rap.all_exact,
        "nearest_pts_mismatches": mismatches,
        "strategies": costs,
    }
    path = write_report("exp2_shared_pts_identity", report)
    print(f"report -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
