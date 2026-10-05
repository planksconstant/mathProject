"""python src/decode.py file.imgc out.png"""
import sys

import numpy as np
from PIL import Image

import codec

if len(sys.argv) != 3:
    sys.exit("usage: decode.py file.imgc out.png")
with open(sys.argv[1], "rb") as f:
    Image.fromarray(codec.decompress(f.read())).save(sys.argv[2])
