"""Opaque AES block encoding."""

from __future__ import annotations

from ..aes_primitives import AES_BLOCK_BYTES
from ..circuit import CircuitBuilder, OpCode
from .base import KeySelector


class OpaqueAesEncoding:
    def encrypt_block(
        self,
        builder: CircuitBuilder,
        block: list[int],
        key_selector: KeySelector,
        label: str,
    ) -> list[int]:
        output = builder.op(
            OpCode.AES256_BLOCK,
            AES_BLOCK_BYTES * 8,
            block,
            f"{label}:aes256_block",
            int(key_selector),
        )
        bytes_out = []
        for index in range(AES_BLOCK_BYTES):
            byte = builder.op(
                OpCode.EXTRACT_BYTE,
                8,
                [output],
                f"{label}:opaque_byte_{index}",
                index,
            )
            bytes_out.append(byte)
        return bytes_out
