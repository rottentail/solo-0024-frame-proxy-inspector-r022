"""Admission gate for bounded random access.

A clip is admissible for random-access contact-sheet fetching only when
*every* display picture can be reached from a random access point through
a reference chain no longer than a configured bound. The evaluation is
purely structural: it walks declared references, decodes nothing, and
therefore runs before any expensive media work.

This mirrors the existing service's "prove complete coverage before
publication" stance rather than accepting a stream and discovering the
cost problem while composing a sheet.
"""

from __future__ import annotations

from dataclasses import dataclass

from labcodec import LoadedContainer, chain_for_target


@dataclass(frozen=True, slots=True)
class GateVerdict:
    admitted: bool
    reason: str
    max_chain_length: int
    max_chain_target: int
    bound: int
    rap_count: int
    unreachable: tuple[int, ...]
    per_target_chain_length: dict[int, int]


def evaluate_random_access(
    container: LoadedContainer,
    chain_bound: int,
) -> GateVerdict:
    per_target: dict[int, int] = {}
    unreachable: list[int] = []
    max_length = 0
    max_target = -1
    for entry in container.display:
        target = entry.display_ord
        try:
            chain = chain_for_target(container, target)
        except KeyError:
            unreachable.append(target)
            continue
        if not any(packet.kind == "I" for packet in chain):
            unreachable.append(target)
            continue
        per_target[target] = len(chain)
        if len(chain) > max_length:
            max_length = len(chain)
            max_target = target

    if unreachable:
        return GateVerdict(
            admitted=False,
            reason="pictures_without_reachable_rap",
            max_chain_length=max_length,
            max_chain_target=max_target,
            bound=chain_bound,
            rap_count=len(container.rap_ordinals),
            unreachable=tuple(unreachable),
            per_target_chain_length=per_target,
        )
    if max_length > chain_bound:
        return GateVerdict(
            admitted=False,
            reason="chain_length_exceeds_bound",
            max_chain_length=max_length,
            max_chain_target=max_target,
            bound=chain_bound,
            rap_count=len(container.rap_ordinals),
            unreachable=(),
            per_target_chain_length=per_target,
        )
    return GateVerdict(
        admitted=True,
        reason="bounded_rap_coverage",
        max_chain_length=max_length,
        max_chain_target=max_target,
        bound=chain_bound,
        rap_count=len(container.rap_ordinals),
        unreachable=(),
        per_target_chain_length=per_target,
    )
