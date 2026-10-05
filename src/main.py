"""Usage:
    python src/main.py photo.png                         # all methods
    python src/main.py photo.png -m dwt -q 60 -l 4
    python src/main.py photo.png -m dct -q 30
    python src/main.py photo.png -m svd -k 40
Writes <out>/<name>.imgc (the real compressed file) and <out>/<name>.png (decoded, for viewing).
Every file is decoded back from disk bytes and checked against the encoder's own reconstruction.
"""
import argparse
import os

import numpy as np
from PIL import Image

import codec
import metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("-m", "--method", choices=["lossless", "dct", "dwt", "svd", "all"], default="all")
    ap.add_argument("-q", "--quality", type=int, default=50, help="dct/dwt quality 1-100")
    ap.add_argument("-l", "--levels", type=int, default=3, help="dwt decomposition levels")
    ap.add_argument("-k", "--rank", type=int, default=20, help="svd rank per channel")
    ap.add_argument("--no-subsample", action="store_true", help="disable 4:2:0 chroma subsampling")
    ap.add_argument("-o", "--out", default="out")
    a = ap.parse_args()

    rgb = np.array(Image.open(a.image).convert("RGB"))
    raw = rgb.size
    os.makedirs(a.out, exist_ok=True)
    print(f"{a.image}: {rgb.shape[1]}x{rgb.shape[0]}, raw {raw} B, input file {os.path.getsize(a.image)} B")
    print(f"{'file':<14}{'bytes':>9}{'ratio':>8}{'PSNR dB':>9}{'MSE':>9}  roundtrip")

    names = {"lossless": "lossless", "dct": f"dct_q{a.quality}",
             "dwt": f"dwt_q{a.quality}_l{a.levels}", "svd": f"svd_k{a.rank}"}
    todo = list(codec.METHODS) if a.method == "all" else [a.method]
    for m in todo:
        data, enc_recon = codec.compress(rgb, m, a.quality, a.levels, a.rank, not a.no_subsample)
        path = os.path.join(a.out, names[m] + ".imgc")
        with open(path, "wb") as f:
            f.write(data)
        with open(path, "rb") as f:
            dec = codec.decompress(f.read())
        ok = np.array_equal(dec, enc_recon) and (m != "lossless" or np.array_equal(dec, rgb))
        Image.fromarray(dec).save(os.path.join(a.out, names[m] + ".png"))
        p = metrics.psnr(rgb, dec)
        print(f"{names[m]:<14}{len(data):>9}{raw / len(data):>8.2f}{p:>9.2f}{metrics.mse(rgb, dec):>9.2f}  {'OK' if ok else 'MISMATCH'}")


if __name__ == "__main__":
    main()
