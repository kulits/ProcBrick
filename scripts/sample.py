#!/usr/bin/env python3
"""Procedural trees with appearance.

Usage: python scripts/sample.py out --count 4
"""

import argparse
from pathlib import Path

import bricknet
import numpy as np

import procbrick


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path, help="output directory of .ldr files")
    ap.add_argument("--count", type=int, default=4, help="trees to sample")
    ap.add_argument("--parts", type=int, default=100, help="parts per tree")
    args = ap.parse_args(argv)
    sampler = procbrick.Sampler()
    rng = np.random.default_rng()
    args.out.mkdir(parents=True, exist_ok=True)
    for k in range(args.count):
        tree = sampler.sample(parts=args.parts)
        variants, colors = procbrick.sample_appearance([part.part_id for part in tree.parts], rng)
        parts = tuple(bricknet.Part(int(part), int(color)) for part, color in zip(variants, colors))
        graph = bricknet.tree_to_graph(bricknet.Tree(parts, tree.edges))
        path = args.out / f"{k}.ldr"
        path.write_text(bricknet.graph_to_ldr(graph), encoding="utf-8")
        print(f"{len(parts)} parts, {len(set(variants.tolist()))} part types -> {path}")


if __name__ == "__main__":
    main()
