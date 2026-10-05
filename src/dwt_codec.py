"""Multi-level 2D Haar wavelet codec with a real bitstream.

YCbCr -> 4:2:0 -> Haar DWT (L levels) -> deadzone quantizer (one step size) ->
LL band: left-neighbour deltas | detail bands: zero-run/size tokens -> Huffman.
One Huffman table for LL and one per decomposition level for the details.

The Haar transform here is orthonormal, so a uniform step size spreads error
evenly across bands, and quality maps to a single step: doubles every 12 points.
"""
import numpy as np

import container
import rle
from colorspace import plane_shapes, split_planes, merge_planes

ROUND = 0.4  # <0.5 => slightly wider zero bin (deadzone), helps rate at same step


def step_for(quality: int) -> float:
    return 2.0 ** ((100 - int(np.clip(quality, 1, 100))) / 12.0)


def haar_step(x):
    a, b = x[0::2, 0::2], x[0::2, 1::2]
    c, d = x[1::2, 0::2], x[1::2, 1::2]
    return ((a + b + c + d) / 2,   # LL: local average
            (a - b + c - d) / 2,   # LH: change across columns
            (a + b - c - d) / 2,   # HL: change across rows
            (a - b - c + d) / 2)   # HH: diagonal


def ihaar_step(ll, lh, hl, hh):
    out = np.empty((ll.shape[0] * 2, ll.shape[1] * 2))
    out[0::2, 0::2] = (ll + lh + hl + hh) / 2
    out[0::2, 1::2] = (ll - lh + hl - hh) / 2
    out[1::2, 0::2] = (ll + lh - hl - hh) / 2
    out[1::2, 1::2] = (ll - lh - hl + hh) / 2
    return out


def dwt2(x: np.ndarray, levels: int) -> np.ndarray:
    """x dims must be divisible by 2**levels. Mallat layout: LL top-left."""
    out = x.astype(np.float64).copy()
    H, W = out.shape
    for k in range(1, levels + 1):
        h, w = H >> (k - 1), W >> (k - 1)
        h2, w2 = h // 2, w // 2
        ll, lh, hl, hh = haar_step(out[:h, :w].copy())
        out[:h2, :w2], out[:h2, w2:w] = ll, lh
        out[h2:h, :w2], out[h2:h, w2:w] = hl, hh
    return out


def idwt2(c: np.ndarray, levels: int) -> np.ndarray:
    out = c.copy()
    H, W = out.shape
    for k in range(levels, 0, -1):
        h, w = H >> (k - 1), W >> (k - 1)
        h2, w2 = h // 2, w // 2
        out[:h, :w] = ihaar_step(out[:h2, :w2], out[:h2, w2:w], out[h2:h, :w2], out[h2:h, w2:w])
    return out


def bands(H: int, W: int, levels: int):
    """(table_index, (row_slice, col_slice)) coarse -> fine. table 0 = LL."""
    yield 0, (slice(0, H >> levels), slice(0, W >> levels))
    for k in range(levels, 0, -1):
        h, w, h2, w2 = H >> (k - 1), W >> (k - 1), H >> k, W >> k
        yield k, (slice(0, h2), slice(w2, w))   # LH
        yield k, (slice(h2, h), slice(0, w2))   # HL
        yield k, (slice(h2, h), slice(w2, w))   # HH


def quantize(c: np.ndarray, step: float) -> np.ndarray:
    return (np.sign(c) * np.floor(np.abs(c) / step + ROUND)).astype(np.int32)


def _pad(shape, levels):
    m = 1 << levels
    return shape[0] + (-shape[0]) % m, shape[1] + (-shape[1]) % m


def _write_channel(q: np.ndarray, levels: int) -> bytes:
    tokens = []
    for t, sl in bands(*q.shape, levels):
        band = q[sl].ravel()
        if t == 0:
            for v in np.diff(band, prepend=0).tolist():
                rle.dc_token(v, 0, tokens)
        else:
            rle.vector_tokens(band, t, tokens)
    return container.encode_tokens(tokens, levels + 1)


def _read_channel(payload: bytes, padded_shape, levels: int) -> np.ndarray:
    cr = container.ChannelReader(payload, levels + 1)
    q = np.zeros(padded_shape, dtype=np.int32)
    for t, sl in bands(*padded_shape, levels):
        hh, ww = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if t == 0:
            q[sl] = np.cumsum([rle.read_dc(cr, 0) for _ in range(hh * ww)]).reshape(hh, ww)
        else:
            q[sl] = rle.read_vector(cr, hh * ww, t).reshape(hh, ww)
    return q


def _reconstruct(qs, quality, levels, H, W, sub):
    step = step_for(quality)
    planes = [idwt2(q * step, levels)[: s[0], : s[1]] + 128.0
              for q, s in zip(qs, plane_shapes(H, W, sub))]
    return merge_planes(planes, H, W, sub)


def encode(rgb: np.ndarray, quality: int = 50, levels: int = 3, subsample_chroma: bool = True):
    H, W = rgb.shape[:2]
    planes = split_planes(rgb, subsample_chroma)
    smallest = min(min(p.shape) for p in planes)
    levels = int(max(0, min(levels, np.floor(np.log2(smallest)))))
    step = step_for(quality)
    qs = []
    for p in planes:
        m = 1 << levels
        padded = np.pad(p - 128.0, ((0, (-p.shape[0]) % m), (0, (-p.shape[1]) % m)), mode="edge")
        qs.append(quantize(dwt2(padded, levels), step))
    body = container.frame([_write_channel(q, levels) for q in qs])
    return quality, levels, int(subsample_chroma), body, _reconstruct(qs, quality, levels, H, W, subsample_chroma)


def decode(W, H, p1, p2, flags, body) -> np.ndarray:
    sub, levels = bool(flags & 1), p2
    shapes = plane_shapes(H, W, sub)
    qs = [_read_channel(sec, _pad(s, levels), levels)
          for sec, s in zip(container.unframe(body, 3), shapes)]
    return _reconstruct(qs, p1, levels, H, W, sub)
