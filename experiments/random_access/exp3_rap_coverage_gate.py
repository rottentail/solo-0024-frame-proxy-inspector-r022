"""Experiment 3 - bounded playback when usable random access points vanish.

Three short fixtures:

* ``bounded``  : every GOP starts with an I packet (true RAP coverage);
* ``head_only``: only ordinal zero is an I packet, every later anchor is a
  P packet chained from the previous one, so late targets force a chain
  back to the head;
* ``broken_ref``: one referenced packet is removed, leaving targets with
  no reachable RAP at all.

The experiment measures reference-chain lengths (no decoding required),
runs the admission gate against a fixed bound, and then shows on the
head-only fixture that exact fetch is still possible through a full
sequential pass while random access cannot stay bounded.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from admission import evaluate_random_access  # noqa: E402
from fetchers import rap_chain, sequential_full  # noqa: E402
from labcodec import (  # noqa: E402
    MAGIC,
    StreamConfig,
    build_manifest,
    chain_for_target,
    load_container,
    write_container,
)
from reportutil import banner, check, summarize_costs, write_report  # noqa: E402

GENERATED = Path(__file__).resolve().parent / "generated"
# Real services cap decode work per cell. A fixture with true RAPs keeps
# every chain inside a single GOP; the head-only fixture grows a chain
# proportional to the whole file, so a bound between them separates them.
CHAIN_BOUND = 6


def write_broken_container(manifest, path: Path, drop_ord: int) -> None:
    """Copy the container but drop one packet and its index row."""
    good = GENERATED / "exp3_head_only.lab"
    container = load_container(good)
    kept = [p for p in container.packets if p.display_ord != drop_ord]
    with path.open("wb") as handle:
        handle.write(MAGIC)
        handle.write(struct.pack("<I", len(kept)))
        for packet in kept:
            handle.write(packet.payload)
        index_rows = [
            {
                "kind": packet.kind,
                "display_ord": packet.display_ord,
                "decode_index": packet.decode_index,
                "offset": packet.byte_offset,
                "length": packet.byte_length,
                "refs": list(packet.refs),
            }
            for packet in kept
        ]
        display_rows = [
            {
                "display_ord": entry.display_ord,
                "pts": entry.pts,
                "hash16": entry.hash16,
                "scene_ord": entry.scene_ord,
            }
            for entry in container.display
        ]
        index = {
            "gop": manifest.config.gop,
            "b_count": manifest.config.b_count,
            "open_gop": manifest.config.open_gop,
            "rap": [ordinal for ordinal in container.rap_ordinals if ordinal != drop_ord],
            "packets": index_rows,
            "display": display_rows,
        }
        raw_index = json.dumps(index, separators=(",", ":")).encode("utf-8")
        handle.write(struct.pack("<I", len(raw_index)))
        handle.write(raw_index)


def main() -> int:
    banner("experiment 3: random access point coverage and the playback bound")

    # A bounded fixture keeps RAPs every GOP; the head-only fixture spans
    # 24 pictures with only ordinal zero independently decodable.
    bounded_config = StreamConfig(
        scene_lengths=(6, 6, 6, 6), gop=6, b_count=1, anchor_is_rap=True
    )
    # One nominal GOP spanning the whole 24-picture clip: only ordinal zero
    # is an anchor (an I packet), everything else is temporal-predicted P.
    head_only_config = StreamConfig(
        scene_lengths=(8, 8, 8),
        gop=24,
        b_count=0,
        anchor_is_rap=False,
        inter_refs_previous_picture=True,
    )

    bounded_manifest = build_manifest(bounded_config)
    head_manifest = build_manifest(head_only_config)
    bounded_path = GENERATED / "exp3_bounded.lab"
    head_path = GENERATED / "exp3_head_only.lab"
    write_container(bounded_manifest, bounded_path)
    write_container(head_manifest, head_path)
    broken_path = GENERATED / "exp3_broken_ref.lab"
    # Drop an interior P packet: every later picture loses its reference walk.
    write_broken_container(head_manifest, broken_path, drop_ord=12)

    bounded = load_container(bounded_path)
    head_only = load_container(head_path)
    broken = load_container(broken_path)

    verdicts = {
        "bounded": evaluate_random_access(bounded, CHAIN_BOUND),
        "head_only": evaluate_random_access(head_only, CHAIN_BOUND),
        "broken_ref": evaluate_random_access(broken, CHAIN_BOUND),
    }
    for name, verdict in verdicts.items():
        print(
            f"{name:11s} admitted={verdict.admitted} reason={verdict.reason} "
            f"max_chain={verdict.max_chain_length} rap={verdict.rap_count} "
            f"unreachable={len(verdict.unreachable)}"
        )

    check(verdicts["bounded"].admitted, "true RAP coverage must be admitted")
    check(
        not verdicts["head_only"].admitted
        and verdicts["head_only"].reason == "chain_length_exceeds_bound",
        "head-only RAP must be rejected for bounded random access",
    )
    check(
        not verdicts["broken_ref"].admitted
        and verdicts["broken_ref"].reason == "pictures_without_reachable_rap",
        "a broken reference chain must be rejected",
    )

    # Head-only fixture: exact source survives via a full sequential pass,
    # but a target's "random access" chain reaches all the way to ordinal 0.
    last_target = len(head_only.display) - 1
    last_chain = chain_for_target(head_only, last_target)
    check(
        len(last_chain) == len(head_only.display),
        "without RAPs the chain for the last picture must span the whole file",
    )
    requested = (1, 7, 13, last_target)
    seq = sequential_full(head_only, requested)
    rap = rap_chain(head_only, requested)
    check(seq.all_exact and rap.all_exact, "both strategies stay source-accurate")
    seq_cost = summarize_costs(seq)
    rap_cost = summarize_costs(rap)
    print("head-only fetch costs:")
    print(f"  sequential_full {seq_cost}")
    print(f"  rap_chain       {rap_cost}")
    check(
        rap_cost["batch_pictures_decoded"] == seq_cost["batch_pictures_decoded"],
        "with no usable RAP, random access degenerates to a full head replay",
    )
    check(
        rap_cost["worst_single_cell_pictures"] == len(head_only.display),
        "a single late cell can force replaying the entire clip",
    )

    # Bounded fixture keeps every chain inside the bound for comparison.
    bounded_chains = {
        target: len(chain_for_target(bounded, target))
        for target in range(len(bounded.display))
    }
    check(
        max(bounded_chains.values()) <= CHAIN_BOUND,
        "bounded fixture chains must respect the configured bound",
    )

    report = {
        "chain_bound": CHAIN_BOUND,
        "verdicts": {
            name: {
                "admitted": verdict.admitted,
                "reason": verdict.reason,
                "max_chain_length": verdict.max_chain_length,
                "max_chain_target": verdict.max_chain_target,
                "rap_count": verdict.rap_count,
                "unreachable": list(verdict.unreachable),
            }
            for name, verdict in verdicts.items()
        },
        "head_only_last_target_chain_length": len(last_chain),
        "head_only_picture_count": len(head_only.display),
        "head_only_fetch_costs": {
            "sequential_full": seq_cost,
            "rap_chain": rap_cost,
        },
        "bounded_max_chain_length": max(bounded_chains.values()),
    }
    path = write_report("exp3_rap_coverage_gate", report)
    print(f"report -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
