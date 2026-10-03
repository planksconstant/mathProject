import numpy as np


def rgb_to_ycbcr(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float64)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return np.stack([y, cb, cr], axis=-1)


def ycbcr_to_rgb(ycc: np.ndarray) -> np.ndarray:
    y, cb, cr = ycc[..., 0], ycc[..., 1] - 128, ycc[..., 2] - 128
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    out = np.stack([r, g, b], axis=-1)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def subsample(ch: np.ndarray) -> np.ndarray:
    """4:2:0 -> average each 2x2 block. Odd dims are edge-padded."""
    h, w = ch.shape
    ch = np.pad(ch, ((0, h % 2), (0, w % 2)), mode="edge")
    H, W = ch.shape
    return ch.reshape(H // 2, 2, W // 2, 2).mean(axis=(1, 3))


def upsample(ch: np.ndarray, shape) -> np.ndarray:
    """Nearest-neighbour 2x upsample, cropped back to `shape`."""
    up = np.repeat(np.repeat(ch, 2, axis=0), 2, axis=1)
    return up[: shape[0], : shape[1]]
