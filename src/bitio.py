"""Bit-level writer/reader (MSB first) + LEB128 varints."""


class BitWriter:
    def __init__(self):
        self.buf = bytearray()
        self.acc = 0
        self.n = 0

    def write(self, value: int, nbits: int):
        if not nbits:
            return
        self.acc = (self.acc << nbits) | (value & ((1 << nbits) - 1))
        self.n += nbits
        while self.n >= 8:
            self.n -= 8
            self.buf.append((self.acc >> self.n) & 0xFF)
        self.acc &= (1 << self.n) - 1

    def getvalue(self) -> bytes:
        if self.n:  # pad last byte with zeros
            self.buf.append((self.acc << (8 - self.n)) & 0xFF)
            self.acc = 0
            self.n = 0
        return bytes(self.buf)


class BitReader:
    def __init__(self, data: bytes, start: int = 0):
        self.data = data
        self.pos = start * 8  # bit position

    def bit(self) -> int:
        p = self.pos
        self.pos = p + 1
        return (self.data[p >> 3] >> (7 - (p & 7))) & 1

    def read(self, n: int) -> int:
        v = 0
        for _ in range(n):
            v = (v << 1) | self.bit()
        return v


def put_varint(buf: bytearray, n: int):
    while n >= 0x80:
        buf.append((n & 0x7F) | 0x80)
        n >>= 7
    buf.append(n)


def get_varint(data: bytes, pos: int):
    n = shift = 0
    while True:
        b = data[pos]
        pos += 1
        n |= (b & 0x7F) << shift
        if b < 0x80:
            return n, pos
        shift += 7
