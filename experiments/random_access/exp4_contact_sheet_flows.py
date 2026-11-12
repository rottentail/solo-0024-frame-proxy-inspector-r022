"""Experiment 4 - contact-sheet cell fetching on the three flows.

A contact sheet needs one representative display ordinal per scene. This
experiment derives those ordinals from the manifest's scene ranges (the
same half-open ranges the production service stores) and runs all three
fetch flows on each fixture family:

* one full sequential pass (accurate, bounded by one file pass);
* grouped RAP chains (accurate, bounded per cell, the proposed promise);
* per-cell head replay (accurate but quadratic, the forbidden baseline);
* nearest-PTS (cheap but provably wrong when PTS repeat).

It also records the admission verdict per fixture so the design can state
exactly which material is allowed into the random-access flow.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from admission import evaluate_random_access  # noqa: E402
from fetchers import (  # noqa: E402
    head_replay,
    nearest_pts,
    rap_chain,
    sequential_full,
)
from labcodec import StreamConfig, build_manifest, load_container, write_container  # noqa: E402
from reportutil import banner, check, summarize_costs, write_report  # noqa: E402

GENERATED = Path(__file__).resolve().parent / "generated"
CHAIN_BOUND = 6


def representative_ordinals(container) -> tuple[int, ...]:
    """One midpoint ordinal per contiguous scene (half-open ranges)."""
    chosen: list[int] = []
    current_scene = None
    start = 0
    entries = container.display
    for index, entry in enumerate(entries):
        if current_scene is None:
            current_scene = entry.scene_ord
            start = index
        end_scene = index == len(entries) - 1
        if not end_scene and entries[index + 1].scene_ord != current_scene:
            end_scene = True
        if end_scene:
            end = index + 1
            chosen.append(start + (end - start) // 2)
            current_scene = None
    return tuple(chosen)


def build_fixtures() -> dict:
    specs = {
        "reorder_closed_gop": StreamConfig(
            scene_lengths=(6, 6, 6, 6), gop=6, b_count=1, anchor_is_rap=True
        ),
        "shared_pts_vfr": StreamConfig(
            scene_lengths=(6, 6, 6, 6),
            gop=6,
            b_count=1,
            anchor_is_rap=True,
            vfr_extra=(14,),
            duplicate_at=(9,),
        ),
        "head_only_long_p": StreamConfig(
            scene_lengths=(8, 8, 8),
            gop=24,
            b_count=0,
            anchor_is_rap=False,
            inter_refs_previous_picture=True,
        ),
    }
    fixtures = {}
    for name, config in specs.items():
        manifest = build_manifest(config)
        path = GENERATED / f"exp4_{name}.lab"
        write_container(manifest, path)
        fixtures[name] = load_container(path)
    return fixtures


def main() -> int:
    banner("experiment 4: contact-sheet flows over the fixture families")
    fixtures = build_fixtures()
    full_report: dict = {"chain_bound": CHAIN_BOUND, "fixtures": {}}

    for name, container in fixtures.items():
        cells = representative_ordinals(container)
        verdict = evaluate_random_access(container, CHAIN_BOUND)
        seq = sequential_full(container, cells)
        rap = rap_chain(container, cells)
        head = head_replay(container, cells)
        pts_requested = tuple(
            next(
                entry.pts
                for entry in container.display
                if entry.display_ord == ordinal
            )
            for ordinal in cells
        )
        nearest = nearest_pts(container, pts_requested, cells)

        check(seq.all_exact, f"{name}: sequential must be exact")
        check(rap.all_exact, f"{name}: RAP chain must be exact")
        check(head.all_exact, f"{name}: head replay stays exact")

        full_report["fixtures"][name] = {
            "cells_per_scene": list(cells),
            "scene_count": len(cells),
            "admission": {
                "admitted": verdict.admitted,
                "reason": verdict.reason,
                "max_chain_length": verdict.max_chain_length,
            },
            "sequential_full": summarize_costs(seq),
            "rap_chain": summarize_costs(rap),
            "head_replay": summarize_costs(head),
            "nearest_pts": summarize_costs(nearest),
        }
        print(f"\n[{name}] cells={list(cells)} admitted={verdict.admitted}")
        for flow in ("sequential_full", "rap_chain", "head_replay", "nearest_pts"):
            print(f"  {flow:15s} {full_report['fixtures'][name][flow]}")

    # Structural guarantees that turn into design promises.
    admitted = {
        name: data
        for name, data in full_report["fixtures"].items()
        if data["admission"]["admitted"]
    }
    check(len(admitted) >= 2, "the reorder and shared-PTS fixtures must be admissible")
    for name, data in admitted.items():
        check(
            data["rap_chain"]["all_exact"]
            and data["rap_chain"]["worst_single_cell_pictures"] <= CHAIN_BOUND,
            f"{name}: admitted material must keep every cell bounded and exact",
        )
        check(
            data["rap_chain"]["batch_pictures_decoded"]
            <= data["head_replay"]["batch_pictures_decoded"],
            f"{name}: RAP flow must never cost more than head replay",
        )
    rejected = full_report["fixtures"]["head_only_long_p"]
    check(
        not rejected["admission"]["admitted"],
        "the long-P head-only fixture must be rejected from random access",
    )
    check(
        rejected["nearest_pts"]["all_exact"] is False
        or full_report["fixtures"]["shared_pts_vfr"]["nearest_pts"]["all_exact"]
        is False,
        "nearest-PTS must be provably inexact on at least one fixture",
    )

    path = write_report("exp4_contact_sheet_flows", full_report)
    print(f"\nreport -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
