"""Generate a deterministic, high-entropy AES-XTS test fixture.

The output is intentionally raw fixture data:
  - lorem_ipsum_plaintext.txt: block-aligned printable plaintext
  - lorem_ipsum_aes_xts.bin: raw AES-XTS ciphertext
  - lorem_ipsum_aes_xts.json: metadata needed to reproduce the encryption

The plaintext is drawn uniformly (from a fixed seed) over exactly the ASCII
alphabet the decoder treats as feasible plaintext -- horizontal tab, line feed,
carriage return, and the printable range 0x20-0x7E. This keeps the true
key/plaintext a valid solution for the ASCII-printable constraint while giving
each plaintext column close to the maximum achievable entropy (~6.6 bits/byte).
High column entropy is what makes the per-block key constraints identifiable, so
the fixture stays a meaningful positive control for key-recovery experiments.

The file names keep the historical ``lorem_ipsum`` prefix so existing fixtures,
tests, and CLI defaults keep working; the contents are no longer lorem ipsum.

AES-XTS is normally used for disk sectors and does not define a container file
format. Keeping the ciphertext raw makes it useful for tests that expect the
same bytes a storage layer would write.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


AES_BLOCK_BYTES = 16
SECTOR_BYTES = 512
MIN_AES_BLOCKS = 1_000
MIN_SECTORS = -(-AES_BLOCK_BYTES * MIN_AES_BLOCKS // SECTOR_BYTES)
TARGET_PLAINTEXT_BYTES = SECTOR_BYTES * MIN_SECTORS

# AES-XTS requires two AES keys concatenated together. This 64-byte value gives
# AES-256-XTS behavior and is fixed so generated fixtures are reproducible.
AES_XTS_KEY = bytes.fromhex(
    "00112233445566778899aabbccddeeff"
    "102132435465768798a9babbdcddfeef"
    "ffeeddccbbaa99887766554433221100"
    "efdfcdcbab9a897867564534231201f0"
)

START_SECTOR_NUMBER = 0

# Fixed seed so the high-entropy plaintext is byte-for-byte reproducible on any
# machine or Python version. The stream is derived from SHA-256 rather than the
# stdlib PRNG, whose output is not contractually stable across versions.
PLAINTEXT_SEED = b"aes-xts-high-entropy-fixture-v1"

# Bytes the decoder's ASCII-printable constraint treats as feasible plaintext:
# horizontal tab, line feed, carriage return, and the printable range 0x20-0x7E.
# Drawing uniformly from exactly this set maximizes entropy without ever making
# the true plaintext violate the constraint.
TEXT_ASCII = bytes(sorted({0x09, 0x0A, 0x0D, *range(0x20, 0x7F)}))


def _printable_stream(length: int, seed: bytes) -> bytes:
    """Return ``length`` bytes drawn uniformly from ``TEXT_ASCII``.

    SHA-256 runs in counter mode to produce a deterministic keystream, and each
    raw byte is rejection-sampled so the reduction onto the alphabet stays
    unbiased (every symbol is equally likely).
    """

    alphabet = TEXT_ASCII
    modulus = len(alphabet)
    # Largest multiple of the alphabet size that fits in a byte. Raw bytes at or
    # above this are rejected so no symbol is favored by the modulo reduction.
    rejection_limit = (256 // modulus) * modulus

    out = bytearray()
    counter = 0
    while len(out) < length:
        keystream = hashlib.sha256(seed + counter.to_bytes(8, "little")).digest()
        counter += 1
        for raw in keystream:
            if raw < rejection_limit:
                out.append(alphabet[raw % modulus])
                if len(out) >= length:
                    break

    return bytes(out)


def build_plaintext() -> bytes:
    """Return sector-aligned, high-entropy printable bytes (>=1,000 AES blocks).

    Every byte is independent and uniform over the feasible ASCII alphabet, so
    each plaintext column carries close to the maximum entropy (~6.6 bits). High
    column entropy is what keeps the per-block key constraints identifiable.
    """

    plaintext = _printable_stream(TARGET_PLAINTEXT_BYTES, PLAINTEXT_SEED)

    if len(plaintext) != TARGET_PLAINTEXT_BYTES:
        raise AssertionError("plaintext generation did not reach target length")
    if len(plaintext) % SECTOR_BYTES:
        raise AssertionError("AES-XTS fixture plaintext must be sector aligned")

    return plaintext


def sector_tweak(sector_number: int) -> bytes:
    """Return the 16-byte XTS tweak for a sector number."""

    return sector_number.to_bytes(AES_BLOCK_BYTES, byteorder="little", signed=False)


def encrypt_aes_xts(plaintext: bytes) -> bytes:
    """Encrypt sector-aligned data with AES-XTS using one tweak per sector."""

    if len(plaintext) < TARGET_PLAINTEXT_BYTES:
        raise ValueError("plaintext must be at least 1,000 AES blocks long")

    if len(plaintext) % SECTOR_BYTES:
        raise ValueError("AES-XTS fixture plaintext must be sector aligned")

    ciphertext = bytearray()
    for sector_offset in range(0, len(plaintext), SECTOR_BYTES):
        sector_number = START_SECTOR_NUMBER + (sector_offset // SECTOR_BYTES)
        sector_plaintext = plaintext[sector_offset : sector_offset + SECTOR_BYTES]
        encryptor = Cipher(
            algorithms.AES(AES_XTS_KEY),
            modes.XTS(sector_tweak(sector_number)),
        ).encryptor()
        ciphertext.extend(encryptor.update(sector_plaintext) + encryptor.finalize())

    return bytes(ciphertext)


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    plaintext_path = output_dir / "lorem_ipsum_plaintext.txt"
    ciphertext_path = output_dir / "lorem_ipsum_aes_xts.bin"
    metadata_path = output_dir / "lorem_ipsum_aes_xts.json"

    plaintext = build_plaintext()
    ciphertext = encrypt_aes_xts(plaintext)

    plaintext_path.write_bytes(plaintext)
    ciphertext_path.write_bytes(ciphertext)
    metadata_path.write_text(
        json.dumps(
            {
                "algorithm": "AES-256-XTS",
                "aes_block_bytes": AES_BLOCK_BYTES,
                "sector_bytes": SECTOR_BYTES,
                "sectors": len(plaintext) // SECTOR_BYTES,
                "plaintext_bytes": len(plaintext),
                "plaintext_blocks": len(plaintext) // AES_BLOCK_BYTES,
                "ciphertext_bytes": len(ciphertext),
                "key_hex": AES_XTS_KEY.hex(),
                "start_sector_number": START_SECTOR_NUMBER,
                "tweak_format": "little-endian 128-bit sector number",
                "plaintext_file": plaintext_path.name,
                "ciphertext_file": ciphertext_path.name,
                "format": "raw concatenated 512-byte AES-XTS sectors",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {plaintext_path}")
    print(f"Wrote {ciphertext_path}")
    print(f"Wrote {metadata_path}")
    print(f"Plaintext blocks: {len(plaintext) // AES_BLOCK_BYTES}")
    print(f"Sectors: {len(plaintext) // SECTOR_BYTES}")


if __name__ == "__main__":
    main()
