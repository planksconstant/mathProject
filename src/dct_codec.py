"""JPEG-style codec with a real bitstream.
YCbCr -> 4:2:0 -> 8x8 DCT -> quantize -> zigzag -> DC delta / AC run-size -> Huffman."""
import numpy as np

import container
import rle
from colorspace import plane_shapes, split_planes, merge_planes

N = 8

# Standard JPEG (Annex K) base tables
Q_LUM = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]], dtype=np.float64)

Q_CHROM = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]], dtype=np.float64)


def _dct_matrix() -> np.ndarray:
    k = np.arange(N)[:, None]
    n = np.arange(N)[None, :]
    C = np.sqrt(2 / N) * np.cos((2 * n + 1) * k * np.pi / (2 * N))
    C[0, :] = np.sqrt(1 / N)
    return C


C = _dct_matrix()  # orthonormal: C @ C.T == I


def _zigzag():
    idx = sorted(((i, j) for i in range(N) for j in range(N)),
                 key=lambda p: (p[0] + p[1], p[1] if (p[0] + p[1]) % 2 == 0 else p[0]))
    return np.array([p[0] for p in idx]), np.array([p[1] for p in idx])


ZI, ZJ = _zigzag()


def quant_table(base: np.ndarray, quality: int) -> np.ndarray:
    quality = int(np.clip(quality, 1, 100))
    scale = 5000 / quality if quality < 50 else 200 - 2 * quality
    return np.clip(np.floor((base * scale + 50) / 100), 1, 255)


def to_blocks(ch: np.ndarray) -> np.ndarray:
    h, w = ch.shape
    ch = np.pad(ch, ((0, (-h) % N), (0, (-w) % N)), mode="edge")
    H, W = ch.shape
    return ch.reshape(H // N, N, W // N, N).swapaxes(1, 2)  # (bh, bw, 8, 8)


def from_blocks(b: np.ndarray, shape) -> np.ndarray:
    bh, bw = b.shape[:2]
    full = b.swapaxes(1, 2).reshape(bh * N, bw * N)
    return full[: shape[0], : shape[1]]


def encode_channel(ch: np.ndarray, qt: np.ndarray) -> np.ndarray:
    blocks = to_blocks(ch - 128.0)
    coef = C @ blocks @ C.T                      # 2D DCT on every block
    return np.rint(coef / qt).astype(np.int32)   # <- the lossy step


def decode_channel(q: np.ndarray, qt: np.ndarray, shape) -> np.ndarray:
    blocks = C.T @ (q * qt) @ C                  # inverse DCT
    return from_blocks(blocks, shape) + 128.0


# ---- bitstream layer ----

def _tables(quality):
    return [quant_table(Q_LUM, quality)] + [quant_table(Q_CHROM, quality)] * 2


def _write_channel(q: np.ndarray) -> bytes:
    zz = q[..., ZI, ZJ].reshape(-1, N * N)       # zigzag every block
    dc = np.diff(zz[:, 0], prepend=0)            # DC is delta-coded across blocks
    tokens = []
    for b in range(zz.shape[0]):
        rle.dc_token(int(dc[b]), 0, tokens)
        rle.vector_tokens(zz[b, 1:], 1, tokens)
    return container.encode_tokens(tokens, 2)


def _read_channel(payload: bytes, shape) -> np.ndarray:
    bh, bw = -(-shape[0] // N), -(-shape[1] // N)
    cr = container.ChannelReader(payload, 2)
    zz = np.zeros((bh * bw, N * N), dtype=np.int32)
    dc = 0
    for b in range(bh * bw):
        dc += rle.read_dc(cr, 0)
        zz[b, 0] = dc
        zz[b, 1:] = rle.read_vector(cr, N * N - 1, 1)
    q = np.zeros((bh, bw, N, N), dtype=np.int32)
    q[..., ZI, ZJ] = zz.reshape(bh, bw, N * N)
    return q


def _reconstruct(qs, quality, H, W, sub):
    shapes = plane_shapes(H, W, sub)
    planes = [decode_channel(q, qt, s) for q, qt, s in zip(qs, _tables(quality), shapes)]
    return merge_planes(planes, H, W, sub)


def encode(rgb: np.ndarray, quality: int = 50, subsample_chroma: bool = True):
    """-> (p1, p2, flags, body, encoder_side_reconstruction)"""
    H, W = rgb.shape[:2]
    planes = split_planes(rgb, subsample_chroma)
    qs = [encode_channel(p, qt) for p, qt in zip(planes, _tables(quality))]
    body = container.frame([_write_channel(q) for q in qs])
    return quality, 0, int(subsample_chroma), body, _reconstruct(qs, quality, H, W, subsample_chroma)


def decode(W, H, p1, p2, flags, body) -> np.ndarray:
    sub = bool(flags & 1)
    shapes = plane_shapes(H, W, sub)
    qs = [_read_channel(sec, s) for sec, s in zip(container.unframe(body, 3), shapes)]
    return _reconstruct(qs, p1, H, W, sub)
