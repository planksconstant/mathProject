"""python src/test_roundtrip.py
For every method / size / setting: file bytes -> decode must equal the encoder's reconstruction,
lossless must equal the input exactly, and lossy PSNR must be sane."""
import numpy as np

import codec
import metrics

rng = np.random.default_rng(0)


def make(h, w):
    y, x = np.mgrid[0:h, 0:w]
    img = np.stack([x * 255 / max(w, 1), y * 255 / max(h, 1), (x + y) * 3 % 256], -1)
    img[h // 4: h // 2, w // 4: w // 2] = [200, 30, 30]
    return np.clip(img + rng.integers(0, 6, img.shape), 0, 255).astype(np.uint8)


cases = [(64, 48), (37, 53), (130, 97), (8, 8), (5, 7), (2, 2), (1, 1)]
settings = [("lossless", {}), ("dct", {"quality": 90}), ("dct", {"quality": 10}),
            ("dct", {"quality": 50, "subsample": False}),
            ("dwt", {"quality": 90, "levels": 3}), ("dwt", {"quality": 30, "levels": 5}),
            ("dwt", {"quality": 60, "levels": 1, "subsample": False}),
            ("svd", {"rank": 10})]

n = 0
for h, w in cases:
    img = make(h, w)
    for method, kw in settings:
        data, enc = codec.compress(img, method, **kw)
        dec = codec.decompress(data)
        assert dec.shape == img.shape, (method, kw, h, w)
        assert np.array_equal(dec, enc), f"bitstream mismatch {method} {kw} {h}x{w}"
        if method == "lossless":
            assert np.array_equal(dec, img), f"lossless not exact {h}x{w}"
        elif min(h, w) >= 37 and kw.get("quality", 0) >= 60:
            assert metrics.psnr(img, dec) > 28, (method, kw, h, w, metrics.psnr(img, dec))
        n += 1
print(f"all {n} round-trip checks passed")
