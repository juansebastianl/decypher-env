"""Concrete AES-256 and AES-XTS helper primitives.

These functions are intentionally small and dependency-free so they can be used
as a reference evaluator for the constraint model and tests.
"""

from __future__ import annotations

AES_BLOCK_BYTES = 16
AES_256_KEY_BYTES = 32
XTS_KEY_BYTES = 64

SBOX = (
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
)

RCON = (0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36)


def _validate_bytes(name: str, value: bytes, expected_length: int) -> None:
    if len(value) != expected_length:
        raise ValueError(f"{name} must be {expected_length} bytes")


def gf_mul(a: int, b: int) -> int:
    result = 0
    for _ in range(8):
        if b & 1:
            result ^= a
        high_bit = a & 0x80
        a = (a << 1) & 0xFF
        if high_bit:
            a ^= 0x1B
        b >>= 1
    return result


def _sub_word(word: int) -> int:
    return (
        (SBOX[(word >> 24) & 0xFF] << 24)
        | (SBOX[(word >> 16) & 0xFF] << 16)
        | (SBOX[(word >> 8) & 0xFF] << 8)
        | SBOX[word & 0xFF]
    )


def _rot_word(word: int) -> int:
    return ((word << 8) & 0xFFFFFFFF) | (word >> 24)


def expand_key_256(key: bytes) -> list[bytes]:
    """Return 15 AES-256 round keys, each 16 bytes."""

    _validate_bytes("AES-256 key", key, AES_256_KEY_BYTES)
    words = [int.from_bytes(key[i : i + 4], "big") for i in range(0, len(key), 4)]
    for i in range(8, 60):
        temp = words[i - 1]
        if i % 8 == 0:
            temp = _sub_word(_rot_word(temp)) ^ (RCON[(i // 8) - 1] << 24)
        elif i % 8 == 4:
            temp = _sub_word(temp)
        words.append(words[i - 8] ^ temp)

    return [
        b"".join(word.to_bytes(4, "big") for word in words[i : i + 4])
        for i in range(0, 60, 4)
    ]


def _add_round_key(state: list[int], round_key: bytes) -> list[int]:
    return [value ^ round_key[index] for index, value in enumerate(state)]


def _sub_bytes(state: list[int]) -> list[int]:
    return [SBOX[value] for value in state]


def _shift_rows(state: list[int]) -> list[int]:
    out = [0] * AES_BLOCK_BYTES
    for row in range(4):
        for col in range(4):
            out[row + 4 * col] = state[row + 4 * ((col + row) % 4)]
    return out


def _mix_columns(state: list[int]) -> list[int]:
    out = state[:]
    for col in range(4):
        i = 4 * col
        a0, a1, a2, a3 = state[i : i + 4]
        out[i] = gf_mul(a0, 2) ^ gf_mul(a1, 3) ^ a2 ^ a3
        out[i + 1] = a0 ^ gf_mul(a1, 2) ^ gf_mul(a2, 3) ^ a3
        out[i + 2] = a0 ^ a1 ^ gf_mul(a2, 2) ^ gf_mul(a3, 3)
        out[i + 3] = gf_mul(a0, 3) ^ a1 ^ a2 ^ gf_mul(a3, 2)
    return out


def aes256_encrypt_block(key: bytes, block: bytes, *, rounds: int = 14) -> bytes:
    """Encrypt one 16-byte block with AES-256."""

    _validate_bytes("AES-256 key", key, AES_256_KEY_BYTES)
    _validate_bytes("AES block", block, AES_BLOCK_BYTES)
    if rounds < 1 or rounds > 14:
        raise ValueError("AES rounds must be between 1 and 14")
    round_keys = expand_key_256(key)
    state = _add_round_key(list(block), round_keys[0])

    for round_index in range(1, rounds):
        state = _sub_bytes(state)
        state = _shift_rows(state)
        state = _mix_columns(state)
        state = _add_round_key(state, round_keys[round_index])

    state = _sub_bytes(state)
    state = _shift_rows(state)
    state = _add_round_key(state, round_keys[rounds])
    return bytes(state)


def xor_bytes(left: bytes, right: bytes) -> bytes:
    if len(left) != len(right):
        raise ValueError("inputs must have the same length")
    return bytes(a ^ b for a, b in zip(left, right))


def sector_tweak_input(sector_number: int) -> bytes:
    return sector_number.to_bytes(AES_BLOCK_BYTES, byteorder="little", signed=False)


def xts_mul_x(tweak: bytes) -> bytes:
    """Multiply a 128-bit little-endian XTS tweak by x in GF(2^128)."""

    _validate_bytes("XTS tweak", tweak, AES_BLOCK_BYTES)
    value = int.from_bytes(tweak, "little")
    carry = value >> 127
    value = ((value << 1) & ((1 << 128) - 1)) ^ (0x87 if carry else 0)
    return value.to_bytes(AES_BLOCK_BYTES, "little")


def xts_block_tweak(key2: bytes, sector_number: int, block_index: int) -> bytes:
    tweak = aes256_encrypt_block(key2, sector_tweak_input(sector_number))
    for _ in range(block_index):
        tweak = xts_mul_x(tweak)
    return tweak


def aes_xts_encrypt_block(
    key1: bytes,
    key2: bytes,
    plaintext_block: bytes,
    sector_number: int,
    block_index: int,
    *,
    rounds: int = 14,
) -> bytes:
    """Encrypt one complete AES-XTS block for a known sector and block index."""

    _validate_bytes("AES-XTS key1", key1, AES_256_KEY_BYTES)
    _validate_bytes("AES-XTS key2", key2, AES_256_KEY_BYTES)
    _validate_bytes("plaintext block", plaintext_block, AES_BLOCK_BYTES)
    tweak = aes256_encrypt_block(key2, sector_tweak_input(sector_number), rounds=rounds)
    for _ in range(block_index):
        tweak = xts_mul_x(tweak)
    encrypted = aes256_encrypt_block(key1, xor_bytes(plaintext_block, tweak), rounds=rounds)
    return xor_bytes(encrypted, tweak)


def split_xts_key(xts_key: bytes) -> tuple[bytes, bytes]:
    _validate_bytes("AES-XTS key", xts_key, XTS_KEY_BYTES)
    return xts_key[:AES_256_KEY_BYTES], xts_key[AES_256_KEY_BYTES:]
