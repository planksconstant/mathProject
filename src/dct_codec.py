import numpy as np

import entropy
from colorspace import rgb_to_ycbcr, ycbcr_to_rgb, subsample, upsample

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
    """libjpeg quality scaling. quality in 1..100."""
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


def run(rgb: np.ndarray, quality: int = 50, chroma_subsample: bool = True):
    """Returns (reconstructed_rgb, compressed_size_bytes)."""
    H, W = rgb.shape[:2]
    ycc = rgb_to_ycbcr(rgb)
    chans = [ycc[..., 0], ycc[..., 1], ycc[..., 2]]
    if chroma_subsample:
        chans[1], chans[2] = subsample(chans[1]), subsample(chans[2])

    qts = [quant_table(Q_LUM, quality)] + [quant_table(Q_CHROM, quality)] * 2
    size, recon = 0, []
    for ch, qt in zip(chans, qts):
        q = encode_channel(ch, qt)
        zz = q[..., ZI, ZJ].reshape(-1, N * N)   # zigzag every block
        size += entropy.blocks_size_bytes(zz)
        recon.append(decode_channel(q, qt, ch.shape))

    if chroma_subsample:
        recon[1], recon[2] = upsample(recon[1], (H, W)), upsample(recon[2], (H, W))
    return ycbcr_to_rgb(np.stack(recon, axis=-1)), size
