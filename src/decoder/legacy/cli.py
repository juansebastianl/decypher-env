"""Command-line interface for AES-XTS constraint sampling."""

from __future__ import annotations

import argparse
from pathlib import Path

from .backends import available_backends
from .sampler import SamplerConfig, XtsSampler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample AES-XTS constrained candidates.")
    parser.add_argument(
        "metadata",
        type=Path,
        help="Path to lorem_ipsum_aes_xts.json or another compatible fixture metadata file.",
    )
    parser.add_argument("--start-block", type=int, default=0)
    parser.add_argument("--block-count", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--fixed-key",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Seed the initial key from fixture metadata (leaks the true key). Off by default.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--encoding", choices=("opaque", "lowered"), default="opaque")
    parser.add_argument("--relaxation", choices=("discrete", "continuous"), default="discrete")
    parser.add_argument(
        "--backend",
        default="auto",
        choices=("auto", "python", "native", "native-pt", "native-continuous-relaxed", "native-algebraic-relaxed"),
        help=f"Sampler engine backend. Available now: {', '.join(available_backends())}",
    )
    parser.add_argument(
        "--check-fixture",
        action="store_true",
        help="Score the known plaintext/key from fixture metadata instead of sampling.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sampler = XtsSampler(
        SamplerConfig(
            metadata_path=args.metadata,
            start_block=args.start_block,
            block_count=args.block_count,
            fixed_key=args.fixed_key,
            seed=args.seed,
            backend=args.backend,
            encoding=args.encoding,
            relaxation=args.relaxation,
        )
    )

    if args.check_fixture:
        result = sampler.score(sampler.known_fixture_assignment())
        print(f"satisfied={result.satisfied}")
        print(f"constraints={result.constraints}")
        print(f"violations={result.violations}")
        print(f"hamming_score={result.hamming_score}")
        return 0 if result.satisfied else 1

    state = sampler.run(args.iterations)
    print(f"satisfied={state.result.satisfied}")
    print(f"constraints={state.result.constraints}")
    print(f"violations={state.result.violations}")
    print(f"hamming_score={state.result.hamming_score}")
    print(f"plaintext_preview={bytes(state.assignment.plaintext[:64])!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
