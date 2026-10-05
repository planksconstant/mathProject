"""JPEG-style run/size tokens.

A token is (table, symbol, extra_value, extra_nbits). Magnitudes are not part of
the Huffman alphabet: the symbol only carries the *bit length* (category) of the
value, and the value's bits are appended raw. That keeps the tables tiny.

  DC-style (deltas): symbol = category                  (0 -> value 0)
  vector (AC / wavelet detail): symbol = run*32 + cat
      0              -> EOB (rest of vector is zero)
      run=255, cat=0 -> ZRL (skip 256 zeros, no value)
"""
import numpy as np

EOB = 0
ZRL = 255 * 32


def amp_bits(v: int, s: int) -> int:
    return v if v > 0 else v + (1 << s) - 1  # one's complement for negatives


def amp_value(bits: int, s: int) -> int:
    return bits if bits >= (1 << (s - 1)) else bits - (1 << s) + 1


def dc_token(v: int, tbl: int, out: list):
    s = abs(v).bit_length()
    out.append((tbl, s, amp_bits(v, s) if s else 0, s))


def vector_tokens(vec, tbl: int, out: list):
    n = len(vec)
    prev = -1
    for i in np.flatnonzero(vec).tolist():
        run = i - prev - 1
        while run > 255:
            out.append((tbl, ZRL, 0, 0))
            run -= 256
        v = int(vec[i])
        s = abs(v).bit_length()
        out.append((tbl, run * 32 + s, amp_bits(v, s), s))
        prev = i
    if prev < n - 1:
        out.append((tbl, EOB, 0, 0))


def read_dc(cr, tbl: int) -> int:
    s = cr.sym(tbl)
    return 0 if s == 0 else amp_value(cr.bits(s), s)


def read_vector(cr, n: int, tbl: int) -> np.ndarray:
    vec = np.zeros(n, dtype=np.int32)
    pos = 0
    while pos < n:
        sym = cr.sym(tbl)
        if sym == EOB:
            break
        run, cat = divmod(sym, 32)
        if cat == 0:  # ZRL
            pos += 256
            continue
        pos += run
        vec[pos] = amp_value(cr.bits(cat), cat)
        pos += 1
    return vec
