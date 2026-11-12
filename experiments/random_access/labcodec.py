"""Deterministic laboratory model of a compressed stream.

This module is *not* a production codec and is not imported by ``src/``.
It exists so the random-access design can be tested without a real
decoder (the environment ships no ffmpeg and the service allows no
third-party packages).  It deliberately models the three properties that
make random frame access hard in real compressed material:

1. anchor packets (``I``) are independently decodable; dependent packets
   (``P``/``B``) are stored as byte deltas against explicit reference
   pictures and cannot be decoded without them;
2. packets are written in decode order, which differs from display order
   whenever B packets are present;
3. presentation timestamps are independent of the packet order, so they
   can be non-uniform (VFR) and can repeat for more than one distinct
   display picture.

Every ground-truth picture has a stable content hash, which lets the
experiments prove *which original picture* a fetch returned instead of
relying on visual inspection.
"""

from __future__ import annotations

import hashlib
import json
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Ground-truth pictures
# ---------------------------------------------------------------------------

SCENE_COLORS = (
    (214, 66, 48),
    (44, 151, 168),
    (222, 183, 53),
    (71, 166, 91),
    (96, 92, 200),
)
PICTURE_WIDTH = 24
PICTURE_HEIGHT = 12


def picture_raster(display_ord: int, scene_lengths: tuple[int, ...]) -> bytes:
    """Return the deterministic RGB raster of one display picture."""
    scene_index = 0
    boundary = scene_lengths[0]
    while display_ord >= boundary and scene_index < len(scene_lengths) - 1:
        scene_index += 1
        boundary += scene_lengths[scene_index]
    red, green, blue = SCENE_COLORS[scene_index % len(SCENE_COLORS)]
    rows = bytearray()
    for y in range(PICTURE_HEIGHT):
        for x in range(PICTURE_WIDTH):
            drift = ((x * 3 + y * 5 + display_ord * 11) % 19) - 9
            rows.append(max(0, min(255, red + drift)))
            rows.append(max(0, min(255, green + drift // 2)))
            rows.append(max(0, min(255, blue + drift)))
    return bytes(rows)


def content_hash(raster: bytes) -> str:
    return hashlib.sha256(raster).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Stream configuration and manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StreamConfig:
    scene_lengths: tuple[int, ...]
    gop: int = 6
    b_count: int = 1
    open_gop: bool = False
    # When True every GOP anchor is an independently decodable I packet.
    # When False only ordinal zero is I; later anchors are P packets chained
    # from the previous anchor, so nothing past zero is a random access point.
    anchor_is_rap: bool = True
    # When True non-anchor packets reference the immediately preceding
    # display picture instead of the GOP anchor, so inter chains grow one
    # step per picture. Models long P-only material with no usable restart.
    inter_refs_previous_picture: bool = False
    # PTS delta applied *after* the display picture at this ordinal.
    # PTS starts at 0; ordinal n gets pts(n) = base_step*n + extra[n].
    vfr_extra: tuple[int, ...] = ()
    duplicate_at: tuple[int, ...] = ()
    base_step: int = 1000

    @property
    def picture_count(self) -> int:
        return sum(self.scene_lengths)

    def pts_for(self, display_ord: int) -> int:
        pts = self.base_step * display_ord
        for ordinal in self.vfr_extra:
            if ordinal <= display_ord:
                pts += self.base_step
        if display_ord in self.duplicate_at:
            pts -= self.base_step
        return pts


@dataclass(frozen=True, slots=True)
class DisplayEntry:
    display_ord: int
    pts: int
    hash16: str
    scene_ord: int


@dataclass(frozen=True, slots=True)
class Packet:
    kind: str  # "I", "P", "B"
    display_ord: int
    decode_index: int
    payload: bytes
    byte_offset: int
    byte_length: int
    refs: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Manifest:
    config: StreamConfig
    display: tuple[DisplayEntry, ...]
    packets: tuple[Packet, ...]
    rap_ordinals: tuple[int, ...]
    pts_to_ordinals: dict[int, tuple[int, ...]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Encoder: build packet order and delta payloads
# ---------------------------------------------------------------------------

_HEADER = struct.Struct("<BBiII")  # kind, refs_count, display_ord, raw_len, z_len


def _encode_packet(
    kind: str,
    display_ord: int,
    decode_index: int,
    raster: bytes,
    references: dict[int, bytes],
    refs: tuple[int, ...],
    offset: int,
) -> Packet:
    if kind == "I" or not refs:
        delta = raster
    else:
        base = references[refs[0]]
        delta = bytes(a ^ b for a, b in zip(raster, base))
    compressed = zlib.compress(delta, 9)
    header = _HEADER.pack(
        ord(kind),
        len(refs),
        display_ord,
        len(raster),
        len(compressed),
    )
    payload = header + compressed
    packet = Packet(
        kind=kind,
        display_ord=display_ord,
        decode_index=decode_index,
        payload=payload,
        byte_offset=offset,
        byte_length=len(payload),
        refs=refs,
    )
    return packet


def _is_anchor(display_ord: int, gop: int) -> bool:
    return display_ord % gop == 0


def _decode_order(count: int, gop: int, b_count: int) -> list[int]:
    """Display ordinals in the order packets must be decoded.

    Within each GOP the anchor packet is decoded before the B packets that
    precede it in display order, which is the classic reorder pattern.
    """
    order: list[int] = []
    anchor = 0
    while anchor < count:
        next_anchor = min(anchor + gop, count)
        order.append(anchor)
        cursor = anchor + 1
        while cursor < next_anchor:
            p_position = min(cursor + b_count, next_anchor - 1)
            if p_position > cursor:
                order.append(p_position)
                order.extend(range(cursor, p_position))
            else:
                order.append(cursor)
            cursor = p_position + 1
        anchor = next_anchor
    return order


def _refs_for(
    display_ord: int,
    config: StreamConfig,
    anchors: tuple[int, ...],
) -> tuple[int, ...]:
    if _is_anchor(display_ord, config.gop):
        if config.anchor_is_rap:
            return ()  # anchor is an independently decodable I packet (RAP)
        if display_ord == 0:
            return ()
        return (display_ord - config.gop,)  # P anchor chained from the prior one
    gop_start = (display_ord // config.gop) * config.gop
    future_anchor = min(gop_start + config.gop, config.picture_count - 1)
    if not config.open_gop:
        if config.inter_refs_previous_picture:
            # Strictly temporal prediction: every inter picture depends on
            # the picture shown immediately before it.
            return (display_ord - 1,)
        # Closed GOP: a B packet reaches its own anchor. When that anchor is
        # a chained P packet, the walk continues through earlier anchors.
        refs = [gop_start]
        return tuple(refs)
    previous_anchor = gop_start - config.gop
    refs: list[int] = []
    if previous_anchor >= 0:
        refs.append(previous_anchor)
    refs.append(future_anchor)
    return tuple(refs)


def build_manifest(config: StreamConfig) -> Manifest:
    count = config.picture_count
    rasters = {
        ordinal: picture_raster(ordinal, config.scene_lengths)
        for ordinal in range(count)
    }
    if config.anchor_is_rap:
        anchors = tuple(
            ordinal
            for ordinal in range(count)
            if _is_anchor(ordinal, config.gop)
        )
    else:
        # Only ordinal zero is independently decodable; all later packets
        # have a non-empty reference chain.
        anchors = (0,)
    order = _decode_order(count, config.gop, config.b_count)

    packets: list[Packet] = []
    decoded: dict[int, bytes] = {}
    offset = 0
    for decode_index, display_ord in enumerate(order):
        if _is_anchor(display_ord, config.gop):
            kind = "I" if config.anchor_is_rap or display_ord == 0 else "P"
        else:
            kind = "B"
        refs = () if kind == "I" else _refs_for(display_ord, config, anchors)
        packet = _encode_packet(
            kind,
            display_ord,
            decode_index,
            rasters[display_ord],
            rasters,
            refs,
            offset,
        )
        packets.append(packet)
        offset += packet.byte_length
        decoded[display_ord] = rasters[display_ord]

    scene_ord = 0
    boundary = config.scene_lengths[0]
    entries: list[DisplayEntry] = []
    pts_map: dict[int, list[int]] = {}
    for ordinal in range(count):
        while ordinal >= boundary and scene_ord < len(config.scene_lengths) - 1:
            scene_ord += 1
            boundary += config.scene_lengths[scene_ord]
        pts = config.pts_for(ordinal)
        entries.append(
            DisplayEntry(
                display_ord=ordinal,
                pts=pts,
                hash16=content_hash(rasters[ordinal]),
                scene_ord=scene_ord,
            )
        )
        pts_map.setdefault(pts, []).append(ordinal)
    return Manifest(
        config=config,
        display=tuple(entries),
        packets=tuple(packets),
        rap_ordinals=anchors,
        pts_to_ordinals={key: tuple(value) for key, value in pts_map.items()},
    )


# ---------------------------------------------------------------------------
# On-disk container (laboratory format only)
# ---------------------------------------------------------------------------

MAGIC = b"LABCODEC1"


def write_container(manifest: Manifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(MAGIC)
        handle.write(struct.pack("<I", len(manifest.packets)))
        for packet in manifest.packets:
            handle.write(packet.payload)
        index_rows = []
        for packet in manifest.packets:
            index_rows.append(
                {
                    "kind": packet.kind,
                    "display_ord": packet.display_ord,
                    "decode_index": packet.decode_index,
                    "offset": packet.byte_offset,
                    "length": packet.byte_length,
                    "refs": list(packet.refs),
                }
            )
        display_rows = [
            {
                "display_ord": entry.display_ord,
                "pts": entry.pts,
                "hash16": entry.hash16,
                "scene_ord": entry.scene_ord,
            }
            for entry in manifest.display
        ]
        index = {
            "gop": manifest.config.gop,
            "b_count": manifest.config.b_count,
            "open_gop": manifest.config.open_gop,
            "rap": list(manifest.rap_ordinals),
            "packets": index_rows,
            "display": display_rows,
        }
        raw_index = json.dumps(index, separators=(",", ":")).encode("utf-8")
        handle.write(struct.pack("<I", len(raw_index)))
        handle.write(raw_index)


@dataclass(frozen=True, slots=True)
class LoadedContainer:
    path: Path
    raw: bytes
    packets: tuple[Packet, ...]
    display: tuple[DisplayEntry, ...]
    rap_ordinals: tuple[int, ...]
    pts_to_ordinals: dict[int, tuple[int, ...]]
    data_end: int


def load_container(path: Path) -> LoadedContainer:
    raw = path.read_bytes()
    if raw[: len(MAGIC)] != MAGIC:
        raise ValueError("bad labcodec magic")
    packet_count = struct.unpack_from("<I", raw, len(MAGIC))[0]
    cursor = len(MAGIC) + 4
    packets: list[Packet] = []
    for _ in range(packet_count):
        kind_code, ref_count, display_ord, _raw_len, _z_len = _HEADER.unpack_from(
            raw, cursor
        )
        header_end = cursor + _HEADER.size
        (z_len,) = struct.unpack_from("<I", raw, header_end - 4)
        length = _HEADER.size + z_len
        payload = raw[cursor : cursor + length]
        # Reference lists live in the JSON side index, not the packet header.
        packets.append(
            Packet(
                kind=chr(kind_code),
                display_ord=display_ord,
                decode_index=len(packets),
                payload=payload,
                byte_offset=cursor,
                byte_length=length,
                refs=(),
            )
        )
        cursor += length
    data_end = cursor
    (index_len,) = struct.unpack_from("<I", raw, cursor)
    index = json.loads(raw[cursor + 4 : cursor + 4 + index_len].decode("utf-8"))
    rebuilt: list[Packet] = []
    by_ord = {packet.display_ord: packet for packet in packets}
    for row in index["packets"]:
        packet = by_ord[row["display_ord"]]
        rebuilt.append(
            Packet(
                kind=row["kind"],
                display_ord=row["display_ord"],
                decode_index=row["decode_index"],
                payload=packet.payload,
                byte_offset=row["offset"],
                byte_length=row["length"],
                refs=tuple(row["refs"]),
            )
        )
    display = tuple(
        DisplayEntry(
            display_ord=row["display_ord"],
            pts=row["pts"],
            hash16=row["hash16"],
            scene_ord=row["scene_ord"],
        )
        for row in index["display"]
    )
    pts_map: dict[int, tuple[int, ...]] = {}
    for entry in display:
        pts_map[entry.pts] = (*pts_map.get(entry.pts, ()), entry.display_ord)
    return LoadedContainer(
        path=path,
        raw=raw,
        packets=tuple(rebuilt),
        display=display,
        rap_ordinals=tuple(index["rap"]),
        pts_to_ordinals=pts_map,
        data_end=data_end,
    )


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Decoder:
    """Stateful decoder; packets must be fed in decode order."""

    dpb: dict[int, bytes] = field(default_factory=dict)
    pictures_decoded: int = 0
    bytes_fed: int = 0

    def feed(self, packet: Packet, raw: bytes) -> bytes:
        kind_code, _ref_count, display_ord, raw_len, _z_len = _HEADER.unpack_from(
            packet.payload, 0
        )
        body = packet.payload[_HEADER.size :]
        delta = zlib.decompress(body)
        if packet.kind == "I" or not packet.refs:
            raster = delta
        else:
            base = self.dpb[packet.refs[0]]
            raster = bytes(a ^ b for a, b in zip(delta, base))
        if len(raster) != raw_len:
            raise ValueError("decoded raster has the wrong length")
        self.dpb[display_ord] = raster
        self.pictures_decoded += 1
        self.bytes_fed += packet.byte_length
        return raster


def decode_chain(
    container: LoadedContainer,
    chain: tuple[Packet, ...],
) -> tuple[bytes, int, int]:
    decoder = Decoder()
    raster = b""
    for packet in chain:
        raster = decoder.feed(packet, container.raw)
    return raster, decoder.pictures_decoded, decoder.bytes_fed


# ---------------------------------------------------------------------------
# Reference-chain math (shared by fetchers and the admission gate)
# ---------------------------------------------------------------------------


def chain_for_target(
    container: LoadedContainer,
    target_ord: int,
) -> tuple[Packet, ...]:
    """Packets, in decode order, required to make the target available.

    The walk is purely data-driven from declared references, so it handles
    both open and closed GOP layouts without special cases.
    """
    by_ord = {packet.display_ord: packet for packet in container.packets}
    if target_ord not in by_ord:
        raise KeyError(target_ord)
    needed: set[int] = set()
    stack = [target_ord]
    while stack:
        ordinal = stack.pop()
        if ordinal in needed:
            continue
        needed.add(ordinal)
        packet = by_ord[ordinal]
        for ref in packet.refs:
            if ref not in needed:
                stack.append(ref)
    chain = [
        packet
        for packet in sorted(container.packets, key=lambda item: item.decode_index)
        if packet.display_ord in needed
    ]
    return tuple(chain)


def rap_start_for_target(
    container: LoadedContainer,
    target_ord: int,
) -> int:
    """Ordinal of the earliest RAP whose chain reaches the target."""
    chain = chain_for_target(container, target_ord)
    for packet in chain:
        if packet.kind == "I":
            return packet.display_ord
    raise ValueError("target has no reachable random access point")
