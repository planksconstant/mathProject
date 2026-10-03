"""Low-rank approximation per RGB channel: A ~= U_k S_k V_k^T."""
import numpy as np


def run(rgb: np.ndarray, k: int = 20):
    """Returns (reconstructed_rgb, size_bytes). Factors are stored as float32."""
    out = np.empty(rgb.shape, dtype=np.float64)
    stored = 0
    for c in range(3):
        U, S, Vt = np.linalg.svd(rgb[..., c].astype(np.float64), full_matrices=False)
        kk = min(k, len(S))
        Uk = U[:, :kk].astype(np.float32)
        Sk = S[:kk].astype(np.float32)
        Vk = Vt[:kk].astype(np.float32)
        out[..., c] = (Uk * Sk) @ Vk
        stored += Uk.size + Sk.size + Vk.size
    return np.clip(np.rint(out), 0, 255).astype(np.uint8), stored * 4


def break_even_rank(h: int, w: int) -> int:
    """Rank per channel above which float32 SVD factors outgrow the raw uint8
    channel: 4*k*(h+w+1) >= h*w."""
    return (h * w) // (4 * (h + w + 1))
