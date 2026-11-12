"""Frame fetch strategies compared by the random-access experiments.

Two strategies model production-worthy behavior:

* ``sequential_full`` decodes the stream from packet zero once and keeps
  every requested picture; this mirrors what a from-scratch sequential
  decoder can guarantee.
* ``rap_chain`` seeks to the nearest preceding random access point and
  decodes only the reference chain of the requested display ordinal,
  grouping requests that share a RAP into one pass.

Two strategies model behavior the design must forbid:

* ``head_replay`` restarts the full decode for every cell, which makes a
  contact sheet's cost grow quadratically with the number of cells;
* ``nearest_pts`` answers a request by timestamp proximity instead of the
  requested original picture, which silently returns the wrong picture
  when timestamps repeat or are non-uniform.
"""

from __future__ import annotations

from dataclasses import dataclass

from labcodec import (
    Decoder,
    LoadedContainer,
    chain_for_target,
    content_hash,
    decode_chain,
)


@dataclass(frozen=True, slots=True)
class FetchOutcome:
    requested_ord: int
    returned_ord: int
    returned_hash: str
    exact_match: bool
    # Cost attributable to this cell if it were fetched on its own.
    cell_pictures_decoded: int
    cell_bytes_fed: int


@dataclass(frozen=True, slots=True)
class BatchFetch:
    outcomes: tuple[FetchOutcome, ...]
    # Cost of the whole batch, including shared work counted once.
    batch_pictures_decoded: int
    batch_bytes_fed: int
    seeks: int

    @property
    def all_exact(self) -> bool:
        return all(item.exact_match for item in self.outcomes)


def sequential_full(
    container: LoadedContainer,
    requested: tuple[int, ...],
) -> BatchFetch:
    """One decode pass from packet zero; requested pictures are cached."""
    wanted = set(requested)
    decoder = Decoder()
    hashes: dict[int, str] = {}
    for packet in container.packets:  # already in decode order
        raster = decoder.feed(packet, container.raw)
        if packet.display_ord in wanted:
            hashes[packet.display_ord] = content_hash(raster)
    outcomes = tuple(
        FetchOutcome(
            requested_ord=ordinal,
            returned_ord=ordinal,
            returned_hash=hashes[ordinal],
            exact_match=True,
            # On its own, each cell still forces a full head-to-target pass.
            cell_pictures_decoded=decoder.pictures_decoded,
            cell_bytes_fed=decoder.bytes_fed,
        )
        for ordinal in requested
    )
    return BatchFetch(
        outcomes=outcomes,
        batch_pictures_decoded=decoder.pictures_decoded,
        batch_bytes_fed=decoder.bytes_fed,
        seeks=0,
    )


def rap_chain(
    container: LoadedContainer,
    requested: tuple[int, ...],
) -> BatchFetch:
    """Seek once per RAP region; decode only the union of needed chains."""
    groups: dict[int, list[int]] = {}
    for ordinal in requested:
        chain = chain_for_target(container, ordinal)
        rap_ord = next(packet.display_ord for packet in chain if packet.kind == "I")
        groups.setdefault(rap_ord, []).append(ordinal)

    total_pictures = 0
    total_bytes = 0
    hashes: dict[int, str] = {}
    cell_cost: dict[int, tuple[int, int]] = {}
    for rap_ord, ordinals in groups.items():
        needed_ordinals: set[int] = set()
        for ordinal in ordinals:
            chain = chain_for_target(container, ordinal)
            cell_cost[ordinal] = (len(chain), sum(p.byte_length for p in chain))
            needed_ordinals.update(packet.display_ord for packet in chain)
        chain = tuple(
            packet
            for packet in sorted(
                container.packets, key=lambda item: item.decode_index
            )
            if packet.display_ord in needed_ordinals
        )
        decoder = Decoder()
        for packet in chain:
            raster = decoder.feed(packet, container.raw)
            if packet.display_ord in ordinals:
                hashes[packet.display_ord] = content_hash(raster)
        total_pictures += decoder.pictures_decoded
        total_bytes += decoder.bytes_fed

    outcomes = tuple(
        FetchOutcome(
            requested_ord=ordinal,
            returned_ord=ordinal,
            returned_hash=hashes[ordinal],
            exact_match=True,
            cell_pictures_decoded=cell_cost[ordinal][0],
            cell_bytes_fed=cell_cost[ordinal][1],
        )
        for ordinal in requested
    )
    return BatchFetch(
        outcomes=outcomes,
        batch_pictures_decoded=total_pictures,
        batch_bytes_fed=total_bytes,
        seeks=len(groups),
    )


def head_replay(
    container: LoadedContainer,
    requested: tuple[int, ...],
) -> BatchFetch:
    """Anti-pattern: restart the whole decode for each requested cell."""
    outcomes: list[FetchOutcome] = []
    total_pictures = 0
    total_bytes = 0
    for ordinal in requested:
        decoder = Decoder()
        raster = b""
        for packet in container.packets:
            raster = decoder.feed(packet, container.raw)
            if packet.display_ord == ordinal:
                break
        total_pictures += decoder.pictures_decoded
        total_bytes += decoder.bytes_fed
        outcomes.append(
            FetchOutcome(
                requested_ord=ordinal,
                returned_ord=ordinal,
                returned_hash=content_hash(raster),
                exact_match=True,
                cell_pictures_decoded=decoder.pictures_decoded,
                cell_bytes_fed=decoder.bytes_fed,
            )
        )
    return BatchFetch(
        outcomes=tuple(outcomes),
        batch_pictures_decoded=total_pictures,
        batch_bytes_fed=total_bytes,
        seeks=len(requested),
    )


def nearest_pts(
    container: LoadedContainer,
    requested_pts: tuple[int, ...],
    intended_ordinals: tuple[int, ...],
) -> BatchFetch:
    """Anti-pattern: answer a PTS request with the closest stored PTS.

    ``intended_ordinals`` records which original picture each PTS request
    was *meant* to identify. On unique uniform PTS the closest PTS happens
    to be that picture; repeated or gapped PTS expose the impersonation.
    """
    entries = sorted(container.display, key=lambda entry: entry.pts)
    outcomes: list[FetchOutcome] = []
    total_pictures = 0
    total_bytes = 0
    for requested_ord, pts in zip(intended_ordinals, requested_pts):
        chosen = min(entries, key=lambda entry: abs(entry.pts - pts))
        chain = chain_for_target(container, chosen.display_ord)
        _raster, pictures, fed_bytes = decode_chain(container, chain)
        total_pictures += pictures
        total_bytes += fed_bytes
        outcomes.append(
            FetchOutcome(
                requested_ord=requested_ord,
                returned_ord=chosen.display_ord,
                returned_hash=chosen.hash16,
                exact_match=chosen.display_ord == requested_ord,
                cell_pictures_decoded=pictures,
                cell_bytes_fed=fed_bytes,
            )
        )
    return BatchFetch(
        outcomes=tuple(outcomes),
        batch_pictures_decoded=total_pictures,
        batch_bytes_fed=total_bytes,
        seeks=len(requested_pts),
    )


def ground_truth_hash(container: LoadedContainer, ordinal: int) -> str:
    return next(
        entry.hash16 for entry in container.display if entry.display_ord == ordinal
    )
