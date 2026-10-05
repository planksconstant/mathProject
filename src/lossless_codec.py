"""Lossless baseline: left-neighbour prediction per channel + Huffman on the residuals."""
import numpy as np

import container


def encode(rgb: np.ndarray):
    sections = []
    for c in range(3):
        ch = rgb[..., c].astype(np.int16)
        res = ch.copy()
        res[:, 1:] = ch[:, 1:] - ch[:, :-1]
        res = (res % 256).astype(np.uint8).ravel()
        sections.append(container.encode_tokens([(0, v, 0, 0) for v in res.tolist()], 1))
    return 0, 0, 0, container.frame(sections), rgb


def decode(W, H, p1, p2, flags, body) -> np.ndarray:
    out = np.empty((H, W, 3), dtype=np.uint8)
    for c, payload in enumerate(container.unframe(body, 3)):
        cr = container.ChannelReader(payload, 1)
        res = np.array([cr.sym(0) for _ in range(H * W)], dtype=np.int64).reshape(H, W)
        out[..., c] = np.cumsum(res, axis=1) % 256
    return out
