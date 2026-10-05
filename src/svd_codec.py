"""Low-rank approximation per RGB channel: A ~= U_k S_k V_k^T.
Stored as float16 U and V^T, float32 singular values (raw, no entropy coding)."""
import numpy as np


def _reconstruct(factors) -> np.ndarray:
    chans = []
    for U, S, Vt in factors:
        chans.append((U.astype(np.float32) * S.astype(np.float32)) @ Vt.astype(np.float32))
    return np.clip(np.rint(np.stack(chans, axis=-1)), 0, 255).astype(np.uint8)


def break_even_rank(h: int, w: int) -> int:
    """Rank where the stored factors outgrow one raw uint8 channel."""
    return (h * w) // (2 * (h + w + 2))


def encode(rgb: np.ndarray, rank: int = 20):
    H, W = rgb.shape[:2]
    k = max(1, min(rank, H, W))
    body, factors = b"", []
    for c in range(3):
        U, S, Vt = np.linalg.svd(rgb[..., c].astype(np.float64), full_matrices=False)
        Uk = U[:, :k].astype("<f2")
        Sk = S[:k].astype("<f4")
        Vk = Vt[:k].astype("<f2")
        body += Uk.tobytes() + Sk.tobytes() + Vk.tobytes()
        factors.append((Uk, Sk, Vk))
    return k, 0, 0, body, _reconstruct(factors)


def decode(W, H, p1, p2, flags, body) -> np.ndarray:
    k, pos, factors = p1, 0, []
    for _ in range(3):
        U = np.frombuffer(body, "<f2", H * k, pos).reshape(H, k); pos += H * k * 2
        S = np.frombuffer(body, "<f4", k, pos);                  pos += k * 4
        V = np.frombuffer(body, "<f2", k * W, pos).reshape(k, W); pos += k * W * 2
        factors.append((U, S, V))
    return _reconstruct(factors)
