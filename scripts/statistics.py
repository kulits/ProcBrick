#!/usr/bin/env python3
"""Statistics of sampled and human-designed assemblies.

python scripts/statistics.py generate procbrick --count 100000
python scripts/statistics.py generate no-rejection --count 100000 --no-rejection
python scripts/statistics.py measure procbrick -o procbrick.npz
python scripts/statistics.py measure no-rejection -o no-rejection.npz
python scripts/statistics.py measure sft.npz --prefixes none -o human.npz
python scripts/statistics.py plot Human-designed=human.npz ProcBrick=procbrick.npz "No rejection=no-rejection.npz"

`sft.npz` is BrickNet's decompressed SFT data.
"""

import argparse
import json
import multiprocessing
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from bricknet import decode_graph, graph_to_ldr, load_graphs, parse_ldr, tree_to_graph
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from scipy.spatial import ConvexHull, QhullError

import procbrick

NEAR = 0.25  # LDU
STATISTICS = (
    "volume",
    "density",
    "short_long",
    "neighbors",
    "joins_per_neighbor",
    "top_type",
    "near_neighbors",
    "near_fraction",
)
CORNERS = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)])
_WORKER = {}

WIDTH = 5.5  # inches
INK, GRID = "#202124", "#9AA0A6"
COLORS = ("#ff4081", "#40c4ff", "#69f0ae", "#e040fb", "#536dfe", "#ffab40")
STYLE = {
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "axes.grid": True,
    "axes.labelsize": 9,
    "text.color": INK,
    "font.family": "monospace",
    "font.monospace": ["DejaVu Sans Mono", "monospace"],
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.direction": "out",
    "ytick.direction": "inout",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.size": 0,
    "ytick.minor.size": 0,
    "xtick.major.pad": 3,
    "ytick.major.pad": 3,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "grid.linestyle": "--",
    "lines.linewidth": 1.2,
    "lines.dash_capstyle": "butt",
    "legend.frameon": True,
    "legend.fancybox": True,
    "legend.framealpha": 1.0,
    "legend.edgecolor": INK,
    "legend.fontsize": 6,
    "legend.borderpad": 0.4,
    "legend.labelspacing": 0.3,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.4,
    "svg.fonttype": "path",
}
# statistic: axis label, log scale, fixed limits (None: from the data), ticks (None: automatic)
AXES = {
    "volume": ("Volume (LDU$^3$)", True, None, (1e4, 1e5, 1e6, 1e7, 1e8, 1e9)),
    "density": ("Density", False, None, None),
    "short_long": ("Short / long extent", False, (0, 1), (0, 0.5, 1)),
    "neighbors": ("Neighbors per part", False, None, (2, 2.5, 3, 3.5, 4)),
    "joins_per_neighbor": ("Joins per neighbor pair", True, None, (1, 1.5, 2, 3, 4, 6, 8)),
    "top_type": ("Top-type fraction", True, None, (0.05, 0.1, 0.2, 0.5, 1)),
}
PREFIXES = (20, 50, 100)
TAIL = 0.5  # percent of each series left out of the axis range on each side
STEPS = 1000  # most steps of a drawn CDF
SPARSE = 20  # assemblies per part count below which a series is drawn as points and an 11-count moving average


def _shape(boxes: np.ndarray, poses: np.ndarray) -> tuple[float, float, float]:
    """Hull volume, density, and short over long extent of boxes (lo, hi) placed at `poses`."""
    sizes = boxes[:, 1] - boxes[:, 0]
    rotations, translations = poses[:, :3, :3], poses[:, :3, 3]
    corners = boxes[:, None, 0] + CORNERS[None] * sizes[:, None]
    points = (np.einsum("nij,nkj->nki", rotations, corners) + translations[:, None]).reshape(-1, 3)
    centered = points - points.mean(axis=0)
    _, axes = np.linalg.eigh(centered.T @ centered)
    extents = np.sort(np.ptp(centered @ axes, axis=0))
    volume = ConvexHull(points).volume
    return volume, np.prod(sizes, axis=1).sum() / volume, extents[0] / extents[2]


def _statistics(part_ids: np.ndarray, poses: np.ndarray, edges: np.ndarray, prefixes: list) -> dict[str, np.ndarray]:
    """Statistics of an assembly and its prefixes, by size less one (NaN where not measured)."""
    n = len(part_ids)
    out = {name: np.full(n, np.nan) for name in STATISTICS}
    boxes = np.asarray(_WORKER["colliders"].bounds(part_ids.tolist())).reshape(-1, 2, 3)
    for size in sorted({size for size in prefixes if size <= n} | {n}):
        inside = edges[edges.max(axis=1) < size]
        pairs = len(np.unique(np.sort(inside, axis=1), axis=0))
        out["neighbors"][size - 1] = 2 * pairs / size
        out["joins_per_neighbor"][size - 1] = len(inside) / max(1, pairs)
        out["top_type"][size - 1] = np.unique(_WORKER["classes"][part_ids[:size]], return_counts=True)[1].max() / size
        if not np.isnan(boxes).any():
            try:
                values = _shape(boxes[:size], poses[:size])
            except QhullError:  # a flat prefix
                continue
            out["volume"][size - 1], out["density"][size - 1], out["short_long"][size - 1] = values
    if np.isnan(boxes).any():  # a part without a collider
        return out
    joined = set(map(tuple, np.sort(edges, axis=1).tolist()))
    pairs = [(a, b) for b in range(n) for a in range(b) if (a, b) not in joined]
    clearances = np.array(_WORKER["colliders"].clearances(part_ids.tolist(), poses.reshape(-1).tolist(), pairs, NEAR))
    near = np.array(pairs, int).reshape(-1, 2)[clearances <= NEAR]
    first = np.full(n, n)  # per part, the size less one of the first prefix in which it has a near neighbor
    for side in (0, 1):
        np.minimum.at(first, near[:, side], near[:, 1])
    sizes = np.arange(1, n + 1)
    near_neighbors = 2 * np.bincount(near[:, 1], minlength=n).cumsum() / sizes
    near_fraction = np.bincount(first, minlength=n + 1).cumsum()[:n] / sizes
    measured = sizes >= 4 if prefixes else sizes == n
    out["near_neighbors"][measured], out["near_fraction"][measured] = near_neighbors[measured], near_fraction[measured]
    return out


def _start(kind, source):
    _WORKER["colliders"] = procbrick.load_colliders()
    _WORKER["classes"] = procbrick.class_ids()
    if kind == "graphs":
        _WORKER["graphs"] = load_graphs(source)
    _WORKER["source"] = source


def _task(item):
    """Part count and statistics of one assembly; the joins are those BrickNet parses."""
    (kind, item), prefixes = item
    if kind == "trees":
        graph = parse_ldr(graph_to_ldr(tree_to_graph(procbrick.load_trees(_WORKER["source"], [item])[0])))
    else:
        graph = parse_ldr(Path(item).read_text(encoding="utf-8")) if kind == "ldraw" else _WORKER["graphs"][item]
    part_ids = np.array(graph.part_ids)
    poses = np.asarray(decode_graph(graph) if graph.transforms is None else graph.transforms)
    edges = np.column_stack((graph.edges["a"], graph.edges["b"]))
    return len(part_ids), _statistics(part_ids, poses, edges, prefixes)


def _generate(args) -> None:
    policy = procbrick.load_sampling_policy()
    if args.no_rejection:  # no packing or connection-count acceptance; collision checks are kept
        policy["parameters"].update(mass_growth_penalty=0, extra_connection_bias=0)
        policy["continuous_controls"]["bounds"]["envelope_penalty"] = [0, 0]
    procbrick.generate(args.out, args.count, args.workers, policy=policy)
    print(f"{args.count:,} trees -> {args.out}")


def _measure(args) -> None:
    prefixes = [] if args.prefixes == "none" else [int(size) for size in args.prefixes.split(",")]
    source = args.source[0]
    if source.endswith(".ldr"):
        kind, items = "ldraw", args.source
    elif source.endswith(".npz"):
        with np.load(source) as graphs:
            kind, items = "graphs", range(args.count or len(graphs["node_ptr"]) - 1)
    else:
        count = json.loads((procbrick.paths.find(source) / "manifest.json").read_text(encoding="utf-8"))["count"]
        kind, items = "trees", range(args.count or count)
    with multiprocessing.Pool(args.workers, _start, (kind, source)) as pool:
        rows = list(pool.imap(_task, [((kind, item), prefixes) for item in items], chunksize=16))
    table = {name: np.full((len(rows), max(parts for parts, _ in rows)), np.nan) for name in STATISTICS}
    for k, (parts, values) in enumerate(rows):
        for name in STATISTICS:
            table[name][k, :parts] = values[name]
    sources = np.array([str(item) for item in items])
    np.savez_compressed(args.out, parts=np.array([parts for parts, _ in rows]), sources=sources, **table)
    print(f"{len(rows):,} assemblies -> {args.out}")


def _at(series, name: str, n: int) -> np.ndarray:
    """Finite values of a statistic at prefixes of `n` parts."""
    values = series["table"][name][:, n - 1] if n <= series["table"][name].shape[1] else np.empty(0)
    return values[np.isfinite(values)]


def _limits(name: str, values: list) -> tuple[float, float]:
    """Fixed range of a statistic, or the central range of every series (whole decades for volume)."""
    _, logarithmic, fixed, _ = AXES[name]
    if fixed is not None:
        return fixed
    low = min(np.percentile(v, TAIL) for v in values if len(v))
    high = max(np.percentile(v, 100 - TAIL) for v in values if len(v))
    if name == "volume":
        return 10 ** np.floor(np.log10(low)), 10 ** np.ceil(np.log10(high))
    if logarithmic:
        return low / 1.08, high * 1.08
    pad = 0.03 * (high - low)
    return low - pad, high + pad


def _x_axis(axis, name: str, lower: float, upper: float) -> None:
    _, logarithmic, _, ticks = AXES[name]
    if logarithmic:
        axis.set_xscale("log")
        axis.xaxis.set_minor_locator(NullLocator())
    if ticks is not None:
        axis.xaxis.set_major_locator(FixedLocator([t for t in ticks if lower <= t <= upper]))
        if name != "volume":
            axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axis.set_xlim(lower, upper)


def _legend(figure, series: list) -> None:
    handles = [Line2D([0], [0], color=s["color"]) for s in series]
    labels = [s["label"] for s in series]
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=len(series))


def _cdfs(series: list, cells: list, rows: int, figsize):
    """CDFs of (statistic, prefix size) `cells` in `rows` rows, one x range per statistic."""
    figure, axes = plt.subplots(rows, len(cells) // rows, figsize=figsize, squeeze=False)
    values = {(name, n, s["id"]): _at(s, name, n) for name, n in cells for s in series}
    for axis, (name, n) in zip(axes.flat, cells):
        for s in series:
            v = np.sort(values[name, n, s["id"]])
            k = np.unique(np.linspace(0, len(v) - 1, min(len(v), STEPS)).round().astype(int))
            axis.step(v[k], (k + 1) / len(v), where="post", color=s["color"], gid=f"{name}-{n}-{s['id']}")
        _x_axis(axis, name, *_limits(name, [v for key, v in values.items() if key[0] == name]))
        axis.set(xlabel=AXES[name][0], ylim=(0, 1.02), yticks=(0, 0.5, 1))
    return figure, axes


def _distributions(series: list):
    """CDFs of the six statistics at 100 parts."""
    figure, axes = _cdfs(series, [(name, 100) for name in AXES], 2, (WIDTH, 3.3))
    for axis in axes[:, 0]:
        axis.set_ylabel("CDF")
    figure.tight_layout(w_pad=1.6, h_pad=0.6, pad=0.5)
    _legend(figure, series)
    return figure


def _prefix_distributions(series: list):
    """CDFs of volume and density at 20, 50, and 100 parts."""
    cells = [(name, n) for n in PREFIXES for name in ("volume", "density")]
    figure, axes = _cdfs(series, cells, len(PREFIXES), (WIDTH, 4.2))
    for row, n in zip(axes, PREFIXES):
        row[0].set_ylabel(f"{n} parts")
    for axis in axes[:-1].flat:
        axis.set_xlabel("")
        axis.tick_params(labelbottom=False)
    figure.tight_layout(w_pad=1.6, h_pad=0.4, pad=0.5)
    _legend(figure, series)
    return figure


def _physical(series: list):
    """Near neighbors per part and the fraction of parts with one, by part count."""
    figure, axes = plt.subplots(1, 2, figsize=(WIDTH, 1.9))
    sizes = np.arange(4, 101)
    for axis, name in zip(axes, ("near_neighbors", "near_fraction")):
        for s in series:
            counts = np.array([len(_at(s, name, n)) for n in sizes])
            n = sizes[counts > 0]
            means = np.array([_at(s, name, k).mean() for k in n])
            gid = f"{name}-{s['id']}"
            if np.median(counts[counts > 0]) < SPARSE:
                axis.scatter(n, means, color=s["color"], s=3, alpha=0.35, linewidths=0, gid=f"{gid}-points")
                means = np.array([means[np.abs(n - k) <= 5].mean() for k in n])
            axis.plot(n, means, color=s["color"], gid=gid)
        axis.set(xlabel="Parts", xlim=(4, 100), xticks=(4, 25, 50, 75, 100), ylim=(0, None))
    axes[0].set_ylabel("Near neighbors")
    axes[1].set(ylabel="In near contact", ylim=(0, 1))
    figure.tight_layout(w_pad=1.6, pad=0.5)
    _legend(figure, series)
    return figure


def _plot(args) -> None:
    if len(args.series) > len(COLORS):
        raise SystemExit(f"at most {len(COLORS)} series")
    plt.rcParams.update(STYLE)
    series = []
    for text, color in zip(args.series, COLORS):
        label, path = text.split("=", 1)
        with np.load(path) as data:
            table = {name: data[name] for name in data.files if name not in ("parts", "sources")}
        series.append(dict(label=label, color=color, table=table, id=re.sub(r"[^a-z0-9]+", "-", label.lower())))
    args.out.mkdir(parents=True, exist_ok=True)
    for name, plot in (
        ("distributions", _distributions),
        ("prefix-distributions", _prefix_distributions),
        ("physical", _physical),
    ):
        figure = plot(series)
        figure.savefig(args.out / f"{name}.svg", bbox_inches="tight", pad_inches=0.05)
        plt.close(figure)
        print(f"-> {args.out / name}.svg")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate", help="sample a directory of trees")
    p.add_argument("out", help="output directory")
    p.add_argument("--count", type=int, default=100000)
    p.add_argument("--no-rejection", action="store_true", help="no packing or connection-count acceptance")
    p.add_argument("--workers", type=int, help="worker processes")
    p.set_defaults(func=_generate)

    p = sub.add_parser("measure", help="statistics of a directory of trees, LDraw files, or a BrickNet graph batch")
    p.add_argument("source", nargs="+", help="a directory of trees, .ldr files, or a BrickNet graph batch (.npz)")
    p.add_argument("-o", "--out", type=Path, required=True, help="output statistics (.npz)")
    p.add_argument("--prefixes", default="20,50,100", help="prefix sizes to measure besides whole assemblies, or none")
    p.add_argument("--count", type=int, help="measure the first assemblies or graphs only")
    p.add_argument("--workers", type=int, default=len(procbrick.cpus()), help="worker processes")
    p.set_defaults(func=_measure)

    p = sub.add_parser("plot", help="the figures of statistics files over each other")
    p.add_argument("series", nargs="+", help="LABEL=statistics file (.npz), drawn in order")
    p.add_argument("-o", "--out", type=Path, default=Path("docs/images"), help="output directory")
    p.set_defaults(func=_plot)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
