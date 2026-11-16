"""Command-line interface: python -m procbrick <command>."""

import argparse
import sys

from . import __version__
from .paths import NAMES, fetch
from .sampler import PARTS, generate


def _cmd_fetch(args):
    try:
        fetch(args.names)
    except ValueError as error:
        sys.exit(str(error))


def _cmd_generate(args):
    def progress(done: int, total: int) -> None:
        print(f"{done}/{total} blocks", flush=True)

    generate(args.out, args.count, args.workers, parts=args.parts, progress=progress)
    print(f"{args.count} trees -> {args.out}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="procbrick", description=__doc__)
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch", help="download released files into the data directory")
    p.add_argument("names", nargs="*", default=["assets"], help=f"{', '.join(NAMES)} (default: assets)")
    p.set_defaults(func=_cmd_fetch)

    p = sub.add_parser("generate", help="sample trees into a new directory; a stopped run resumes")
    p.add_argument("out", help="output directory")
    p.add_argument("--count", type=int, required=True, help="trees to sample")
    p.add_argument("--parts", type=int, default=PARTS, help=f"parts per tree (default: {PARTS})")
    p.add_argument("--workers", type=int, help="worker processes")
    p.set_defaults(func=_cmd_generate)

    args = ap.parse_args(argv)
    try:
        args.func(args)
    except FileNotFoundError as error:
        sys.exit(str(error))


if __name__ == "__main__":
    main()
