"""Huffman-based size estimation + a lossless baseline.

We don't write a real bitstream container here; we build actual Huffman
code lengths and sum bits, plus a rough per-symbol cost for storing the table.
That's close enough to compare methods. Swap in a real writer later.

This one is actually tufff :P 
"""
import heapq
import math
from collections import Counter
from itertools import count

import numpy as np

TABLE_BYTES_PER_SYMBOL = 3  # rough cost of shipping the code table


def huffman_lengths(freq: dict) -> dict:
    """symbol -> code length in bits."""
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


def coded_size_bytes(freq: dict) -> int:
    if not freq:
        return 0
    lengths = huffman_lengths(freq)
    bits = sum(lengths[s] * c for s, c in freq.items())
    return math.ceil(bits / 8) + TABLE_BYTES_PER_SYMBOL * len(freq)


def lossless_size_bytes(rgb: np.ndarray) -> int:
    """Left-neighbour prediction per channel, then Huffman over the residuals.
    Fully reversible, so this is a true lossless size estimate."""
    total = 0
    for c in range(3):
        ch = rgb[..., c].astype(np.int16)
        res = ch.copy()
        res[:, 1:] = ch[:, 1:] - ch[:, :-1]
        res = (res % 256).astype(np.uint8).ravel()
        counts = np.bincount(res, minlength=256)
        freq = {i: int(n) for i, n in enumerate(counts) if n}
        total += coded_size_bytes(freq)
    return total


def rle_ac(row: np.ndarray):
    """Zigzagged 64-coef row -> [(zero_run, value), ..., EOB] for the AC part."""
    out, prev = [], -1
    for i in np.nonzero(row[1:])[0]:
        out.append((int(i - prev - 1), int(row[1 + i])))
        prev = i
    out.append(("EOB",))
    return out


def blocks_size_bytes(zz: np.ndarray) -> int:
    """zz: (n_blocks, 64) int array in zigzag order. DC is delta-coded,
    AC is (run, value) + EOB. DC and AC get separate Huffman tables like JPEG."""
    dc = np.diff(zz[:, 0], prepend=0)
    dc_freq = Counter(int(d) for d in dc)
    ac_freq = Counter()
    for row in zz:
        ac_freq.update(rle_ac(row))
    return coded_size_bytes(dc_freq) + coded_size_bytes(ac_freq)
