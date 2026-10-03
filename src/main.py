"""Usage:
    python src/main.py photo.png                       # run everything
    python src/main.py photo.png -m dct -q 30
    python src/main.py photo.png -m svd -k 40
"""
import argparse
import os

import numpy as np
from PIL import Image

import dct_codec
import entropy
import metrics
import svd_codec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("-m", "--method", choices=["dct", "svd", "lossless", "all"], default="all")
    ap.add_argument("-q", "--quality", type=int, default=50, help="DCT quality 1-100")
    ap.add_argument("-k", "--rank", type=int, default=20, help="SVD rank per channel")
    ap.add_argument("--no-subsample", action="store_true", help="disable 4:2:0 chroma subsampling")
    ap.add_argument("-o", "--out", default="out")
    a = ap.parse_args()

    rgb = np.array(Image.open(a.image).convert("RGB"))
    raw = rgb.size  # H*W*3 bytes
    os.makedirs(a.out, exist_ok=True)
    print(f"{a.image}: {rgb.shape[1]}x{rgb.shape[0]}, raw {raw} bytes")
    print(f"{'method':<12}{'bytes':>10}{'ratio':>9}{'PSNR dB':>10}{'MSE':>10}")

    def report(name, size, recon):
        p = metrics.psnr(rgb, recon)
        print(f"{name:<12}{size:>10}{metrics.compression_ratio(raw, size):>9.2f}{p:>10.2f}{metrics.mse(rgb, recon):>10.2f}")
        Image.fromarray(recon).save(os.path.join(a.out, f"{name}.png"))

    if a.method in ("lossless", "all"):
        report("lossless", entropy.lossless_size_bytes(rgb), rgb)
    if a.method in ("dct", "all"):
        recon, size = dct_codec.run(rgb, a.quality, not a.no_subsample)
        report(f"dct_q{a.quality}", size, recon)
    if a.method in ("svd", "all"):
        h, w = rgb.shape[:2]
        recon, size = svd_codec.run(rgb, a.rank)
        report(f"svd_k{a.rank}", size, recon)
        print(f"(svd only saves space below rank ~{svd_codec.break_even_rank(h, w)})")


if __name__ == "__main__":
    main()
