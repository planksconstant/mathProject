"""File layout (.imgc), all integers big-endian:

  header (17 B): magic 'IMGC' | method u8 | width u32 | height u32 | p1 u16 | p2 u8 | flags u8
      method: 0 lossless, 1 dct, 2 dwt, 3 svd
      dct:  p1=quality           flags bit0 = 4:2:0 chroma
      dwt:  p1=quality p2=levels flags bit0 = 4:2:0 chroma
      svd:  p1=rank
  body: 3 sections (Y, Cb, Cr / R, G, B), each  u32 length | payload
  entropy-coded payload: Huffman tables (varint count, then varint sym + u8 len each)
                         followed by the packed bitstream.
"""
import struct
from collections import Counter

import huffman
from bitio import BitWriter, BitReader

MAGIC = b"IMGC"
HDR = struct.Struct(">4sBIIHBB")
LOSSLESS, DCT, DWT, SVD = range(4)


def pack_header(method, w, h, p1, p2, flags) -> bytes:
    return HDR.pack(MAGIC, method, w, h, p1, p2, flags)


def unpack_header(data: bytes):
    magic, method, w, h, p1, p2, flags = HDR.unpack_from(data, 0)
    if magic != MAGIC:
        raise ValueError("not an IMGC file")
    return method, w, h, p1, p2, flags, HDR.size


def frame(sections) -> bytes:
    return b"".join(struct.pack(">I", len(s)) + s for s in sections)


def unframe(body: bytes, n: int):
    out, pos = [], 0
    for _ in range(n):
        (length,) = struct.unpack_from(">I", body, pos)
        pos += 4
        out.append(body[pos:pos + length])
        pos += length
    return out


def encode_tokens(tokens, ntables: int) -> bytes:
    """tokens: list of (table, symbol, extra_value, extra_nbits)."""
    freqs = [Counter() for _ in range(ntables)]
    for t, s, _, _ in tokens:
        freqs[t][s] += 1
    lengths = [huffman.code_lengths(f) for f in freqs]
    codes = [huffman.canonical_codes(l) for l in lengths]
    head = b"".join(huffman.pack_table(l) for l in lengths)
    w = BitWriter()
    for t, s, extra, ne in tokens:
        c, l = codes[t][s]
        w.write(c, l)
        w.write(extra, ne)
    return head + w.getvalue()


class ChannelReader:
    def __init__(self, payload: bytes, ntables: int):
        pos, self.tables = 0, []
        for _ in range(ntables):
            lengths, pos = huffman.unpack_table(payload, pos)
            self.tables.append(huffman.decode_table(lengths))
        self.r = BitReader(payload, pos)

    def sym(self, t: int) -> int:
        return huffman.read_symbol(self.r, self.tables[t])

    def bits(self, n: int) -> int:
        return self.r.read(n)
