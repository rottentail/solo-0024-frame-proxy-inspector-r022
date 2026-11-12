"""Experiment 1 - provenance under reorder, and fetch cost.

Fixture: a short stream with one B packet per GOP, so packet (decode)
order differs from display order. We request several display ordinals
and compare what each strategy returns against the ground-truth content
hash recorded in the manifest, plus what each strategy costs.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fetchers import (  # noqa: E402
    ground_truth_hash,
    head_replay,
    nearest_pts,
    rap_chain,
    sequential_full,
)
from labcodec import StreamConfig, build_manifest, load_container, write_container  # noqa: E402
from reportutil import banner, check, summarize_costs, write_report  # noqa: E402

GENERATED = Path(__file__).resolve().parent / "generated"


def main() -> int:
    banner("experiment 1: reorder provenance and fetch cost")
    config = StreamConfig(
        scene_lengths=(6, 6, 6, 6),
        gop=6,
        b_count=1,
        open_gop=False,
    )
    manifest = build_manifest(config)
    fixture = GENERATED / "exp1_reordered.lab"
    write_container(manifest, fixture)
    container = load_container(fixture)

    decode_order = [packet.display_ord for packet in container.packets]
    display_order = [entry.display_ord for entry in container.display]
    check(decode_order != display_order, "fixture must reorder packets")
    b_packets = [packet for packet in container.packets if packet.kind == "B"]
    check(b_packets, "fixture must contain B packets")
    print(f"picture_count={len(display_order)} packets={len(decode_order)}")
    print(f"display_order={display_order}")
    print(f"decode_order ={decode_order}")
    print(f"rap_ordinals ={list(container.rap_ordinals)}")

    requested = (1, 5, 8, 11, 17, 23)
    check(all(0 <= ordinal < len(display_order) for ordinal in requested), "bounds")

    seq = sequential_full(container, requested)
    rap = rap_chain(container, requested)
    head = head_replay(container, requested)

    for batch in (seq, rap, head):
        for outcome in batch.outcomes:
            expected = ground_truth_hash(container, outcome.requested_ord)
            check(
                outcome.returned_hash == expected,
                f"strategy returned wrong picture for {outcome.requested_ord}",
            )

    # The timestamp-only heuristic on the same (unique, uniform) PTS.
    pts_requested = tuple(
        next(
            entry.pts
            for entry in container.display
            if entry.display_ord == ordinal
        )
        for ordinal in requested
    )
    nearest = nearest_pts(container, pts_requested, requested)
    nearest_exact_here = nearest.all_exact
    print(
        "nearest_pts on uniform unique PTS is exact only by construction: "
        f"{nearest_exact_here}"
    )

    # Prove the RAP chain never includes packets from the start of the file
    # for a mid-stream target: its earliest packet is a RAP above packet zero.
    from labcodec import chain_for_target

    mid_chain = chain_for_target(container, 17)
    earliest = mid_chain[0]
    check(earliest.kind == "I", "RAP chain must begin at an I packet")
    check(earliest.display_ord == 12, "chain for ordinal 17 must start at RAP 12")
    check(
        all(packet.display_ord >= 12 for packet in mid_chain),
        "RAP chain must not replay packets from before the chosen RAP",
    )

    costs = {
        "sequential_full": summarize_costs(seq),
        "rap_chain": summarize_costs(rap),
        "head_replay": summarize_costs(head),
        "nearest_pts": summarize_costs(nearest),
    }
    print("costs:")
    for name, summary in costs.items():
        print(f"  {name:15s} {summary}")

    # Hard, machine-checked expectations.
    check(costs["sequential_full"]["all_exact"], "sequential must be exact")
    check(costs["rap_chain"]["all_exact"], "rap chain must be exact")
    check(costs["head_replay"]["all_exact"], "head replay is exact but wasteful")
    check(
        costs["rap_chain"]["batch_pictures_decoded"]
        < costs["head_replay"]["batch_pictures_decoded"],
        "RAP chain must decode fewer pictures than per-cell head replay",
    )
    check(
        costs["rap_chain"]["batch_pictures_decoded"]
        <= costs["sequential_full"]["batch_pictures_decoded"],
        "RAP chain must not decode more than one full pass",
    )

    report = {
        "fixture": str(fixture.relative_to(fixture.parents[1])),
        "picture_count": len(display_order),
        "gop": config.gop,
        "b_count": config.b_count,
        "display_order": display_order,
        "decode_order": decode_order,
        "rap_ordinals": list(container.rap_ordinals),
        "requested_display_ordinals": list(requested),
        "chain_for_ord_17": {
            "packets_in_decode_order": [
                packet.display_ord for packet in mid_chain
            ],
            "chain_length": len(mid_chain),
        },
        "strategies": costs,
        "nearest_pts_exact_on_uniform_pts": nearest_exact_here,
    }
    path = write_report("exp1_reorder_provenance", report)
    print(f"report -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
