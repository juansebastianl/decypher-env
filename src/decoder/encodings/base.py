"""Encoding interfaces for AES block models."""

from __future__ import annotations

from enum import IntEnum
from typing import Protocol

from ..circuit import CircuitBuilder


class KeySelector(IntEnum):
    KEY1 = 1
    KEY2 = 2


class AesEncoding(Protocol):
    def encrypt_block(
        self,
        builder: CircuitBuilder,
        block: list[int],
        key_selector: KeySelector,
        label: str,
    ) -> list[int]:
        """Emit an AES-256 encryption circuit and return output byte value IDs."""
