"""Top level: compress(rgb, method, ...) -> .imgc bytes, decompress(bytes) -> rgb."""
import numpy as np

import container
import dct_codec
import dwt_codec
import lossless_codec
import svd_codec

METHODS = {"lossless": container.LOSSLESS, "dct": container.DCT,
           "dwt": container.DWT, "svd": container.SVD}
_BY_ID = {container.LOSSLESS: lossless_codec, container.DCT: dct_codec,
          container.DWT: dwt_codec, container.SVD: svd_codec}


def compress(rgb: np.ndarray, method: str, quality=50, levels=3, rank=20, subsample=True):
    """Returns (file_bytes, encoder_side_reconstruction)."""
    H, W = rgb.shape[:2]
    mid = METHODS[method]
    if mid == container.LOSSLESS:
        p1, p2, fl, body, recon = lossless_codec.encode(rgb)
    elif mid == container.DCT:
        p1, p2, fl, body, recon = dct_codec.encode(rgb, quality, subsample)
    elif mid == container.DWT:
        p1, p2, fl, body, recon = dwt_codec.encode(rgb, quality, levels, subsample)
    else:
        p1, p2, fl, body, recon = svd_codec.encode(rgb, rank)
    return container.pack_header(mid, W, H, p1, p2, fl) + body, recon


def decompress(data: bytes) -> np.ndarray:
    method, W, H, p1, p2, fl, off = container.unpack_header(data)
    return _BY_ID[method].decode(W, H, p1, p2, fl, data[off:])
