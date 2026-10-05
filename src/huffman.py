"""Canonical Huffman: build lengths, assign codes, (de)serialize the table, decode.

Only (symbol, code_length) pairs are stored. Encoder and decoder both derive the
exact same codes by sorting on (length, symbol), so no tree needs to be shipped.
"""
import heapq
from itertools import count

import bitio


def code_lengths(freq: dict) -> dict:
    if not freq:
        return {}
    if len(freq) == 1:
        return {s: 1 for s in freq}
    uid = count()
    heap = [(f, next(uid), [s]) for s, f in freq.items()]
    heapq.heapify(heap)
    depth = {s: 0 for s in freq}
    while len(heap) > 1:
        f1, _, s1 = heapq.heappop(heap)
        f2, _, s2 = heapq.heappop(heap)
        for s in s1 + s2:
            depth[s] += 1
        heapq.heappush(heap, (f1 + f2, next(uid), s1 + s2))
    return depth


def canonical_codes(lengths: dict) -> dict:
    """symbol -> (code, length)"""
    items = sorted(lengths.items(), key=lambda kv: (kv[1], kv[0]))
    out, code, prev = {}, 0, (items[0][1] if items else 0)
    for sym, l in items:
        code <<= l - prev
        prev = l
        out[sym] = (code, l)
        code += 1
    return out


def pack_table(lengths: dict) -> bytes:
    buf = bytearray()
    items = sorted(lengths.items(), key=lambda kv: (kv[1], kv[0]))
    bitio.put_varint(buf, len(items))
    for sym, l in items:
        bitio.put_varint(buf, sym)
        buf.append(l)
    return bytes(buf)


def unpack_table(data: bytes, pos: int):
    n, pos = bitio.get_varint(data, pos)
    lengths = {}
    for _ in range(n):
        sym, pos = bitio.get_varint(data, pos)
        lengths[sym] = data[pos]
        pos += 1
    return lengths, pos


def decode_table(lengths: dict) -> dict:
    return {(l, c): s for s, (c, l) in canonical_codes(lengths).items()}


def read_symbol(reader: bitio.BitReader, table: dict) -> int:
    code = l = 0
    while True:
        code = (code << 1) | reader.bit()
        l += 1
        s = table.get((l, code))
        if s is not None:
            return s
        if l > 64:
            raise ValueError("corrupt stream")
