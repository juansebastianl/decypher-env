"""ASCII domain helpers for plaintext constraints."""

from __future__ import annotations

from collections.abc import Iterable

PRINTABLE_ASCII = tuple(range(0x20, 0x7F))
TEXT_ASCII = tuple(sorted((*PRINTABLE_ASCII, 0x09, 0x0A, 0x0D)))
_TEXT_ASCII_SET = frozenset(TEXT_ASCII)


def is_text_ascii(value: int | Iterable[int]) -> bool:
    """Return whether a byte is text ASCII.

    Accepts a single byte value or any iterable of byte values (e.g. ``bytes``,
    ``bytearray``); for a sequence the result is ``True`` only when *every* byte
    is text ASCII. Passing a multi-byte ``bytes`` object used to silently return
    ``False`` because the object was compared against a tuple of integers.
    """

    if isinstance(value, int):
        return value in _TEXT_ASCII_SET
    return all(byte in _TEXT_ASCII_SET for byte in value)


def coerce_to_text_ascii(value: int) -> int:
    if is_text_ascii(value):
        return value
    return PRINTABLE_ASCII[value % len(PRINTABLE_ASCII)]
