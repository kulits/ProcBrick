"""Compressed BrickNet trees."""

import functools
import json
import os
from pathlib import Path

import numpy as np
from bricknet import AxleEdge, BallEdge, FixedEdge, HingeEdge, Part, StudEdge, Tree, load_connector_list
from bricknet.core import AxleSub, Edge, StudSub

from . import _core
from .catalog import representatives
from .paths import find, sha256

_GROUP = 64
_STUD, _AXLE, _BALL, _HINGE, _FIXED = range(5)  # family codes of the atlas
_CODES = {StudEdge: _STUD, AxleEdge: _AXLE, BallEdge: _BALL, HingeEdge: _HINGE, FixedEdge: _FIXED}
_COORDINATES = (("yaw",), ("flip", "yaw", "slide"), ("rx", "ry", "rz"), ("flip", "yaw"), ())


@functools.lru_cache(maxsize=4)
def _coding(assets: str) -> _core.Coding:
    return _core.Coding(assets, representatives().tolist())


@functools.lru_cache(maxsize=8)
def _reader(path: Path) -> _core.Records:
    return _core.Records(path)


def _edge(row: list[int], child: int, part_ids: list[int]) -> Edge:
    """BrickNet tree edge of a sampler row: parent, its connector, the child's connector, family code, and the joint
    coordinates from the parent."""
    parent, a, b, family, q0, q1, q2 = row
    connectors = load_connector_list()
    pc, cc = connectors[part_ids[parent]][a], connectors[part_ids[child]][b]
    if family in (_STUD, _AXLE):
        kind = StudSub if family == _STUD else AxleSub
        subtypes = kind[pc[1]], kind[cc[1]]
    else:  # BrickNet codes subtype * 2 + polarity, the child's polarity the opposite of the parent's
        subtypes = 2 * int(pc[1]) + (pc[2] == "on"), 2 * int(pc[1]) + (pc[2] == "in")
    base = (parent, child, *subtypes, pc[3], cc[3])
    if family == _STUD:
        return StudEdge(*base, yaw=q0)
    if family == _AXLE:
        return AxleEdge(*base, flip=bool(q0), yaw=q1, slide=q2)
    if family == _BALL:
        return BallEdge(*base, rx=q0, ry=q1, rz=q2)
    if family == _HINGE:
        return HingeEdge(*base, flip=bool(q0), yaw=q1)
    return FixedEdge(*base)


def _row(edge: Edge, part_ids: tuple[int, ...]) -> list[int]:
    """Sampler row of a BrickNet tree edge (the inverse of `_edge`)."""
    family = _CODES[type(edge)]
    if family in (_STUD, _AXLE):
        kind = StudSub if family == _STUD else AxleSub
        keys = (kind(edge.parent_sub).name, None), (kind(edge.child_sub).name, None)
    else:
        polarity = "in" if edge.parent_sub % 2 == 0 else "on"
        keys = (str(edge.parent_sub // 2), polarity), (str(edge.child_sub // 2), "on" if polarity == "in" else "in")
    flat = []
    for part, (subtype, polarity), index in zip((edge.parent, edge.child), keys, (edge.parent_conn, edge.child_conn)):
        connectors = load_connector_list()[part_ids[part]]
        found = [k for k, c in enumerate(connectors) if (c[1], c[2], c[3]) == (subtype, polarity, index)]
        if not found:
            raise ValueError(f"part {part_ids[part]} has no connector {subtype} {polarity} {index}")
        flat.append(found[0])
    q = [getattr(edge, name) for name in _COORDINATES[family]]
    q.extend([0] * (3 - len(q)))
    return [edge.parent, *flat, family, *q]


def _trees(sizes: list[int], part_ids, rows, colors=None) -> list[Tree]:
    """Trees of `sizes[k]` parts from the concatenated part ids and sampler rows of their edges (edge k of a tree
    attaches part k + 1)."""
    part_ids = np.asarray(part_ids, np.int64).tolist()
    rows = np.asarray(rows, np.int64).reshape(-1, 7).tolist()
    colors = [16] * len(part_ids) if colors is None else np.asarray(colors, np.int64).tolist()
    trees, start, edge = [], 0, 0
    for n in sizes:
        ids = part_ids[start : start + n]
        parts = tuple(Part(part, color) for part, color in zip(ids, colors[start : start + n]))
        edges = tuple(_edge(rows[edge + k], k + 1, ids) for k in range(n - 1))
        trees.append(Tree(parts, edges))
        start, edge = start + n, edge + n - 1
    return trees


def _write(path: Path, data: bytes) -> None:
    """Write a file through a temporary file synced to disk, then renamed."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}")
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _index(offsets: np.ndarray) -> bytes:
    """Per group of 64 records, the offset of its first, then each record's from it modulo 2^16."""
    count = len(offsets) - 1
    groups = -(-count // _GROUP)
    starts = np.zeros(groups * _GROUP, "<u8")
    starts[:count] = offsets[:-1]
    base = starts[::_GROUP].copy()
    within = np.where(np.arange(groups * _GROUP) < count, starts - np.repeat(base, _GROUP), 0) % (1 << 16)
    index = np.zeros(groups, [("base", "<u8"), ("offsets", "<u2", _GROUP)])
    index["base"], index["offsets"] = base, within.reshape(groups, -1)
    return index.tobytes()


def _seal(out: Path, offsets: np.ndarray) -> None:
    """Write the index and manifest of records `out/data` at `offsets` (one more than the records)."""
    _write(out / "index", _index(offsets))
    files = {name: dict(bytes=(out / name).stat().st_size, sha256=sha256(out / name)) for name in ("data", "index")}
    _write(out / "manifest.json", json.dumps(dict(count=len(offsets) - 1, files=files), indent=2).encode())


def _save_records(path: Path, records: list[bytes]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _write(path / "data", b"".join(records))
    _seal(path, np.cumsum([0] + [len(record) for record in records], dtype="<u8"))


def load_trees(
    path: str | Path,
    indices: list[int] | np.ndarray | None = None,
    parts: int | None = None,
    assets: str | Path = "assets",
) -> list[Tree]:
    """Trees `indices` (default: all) of a directory of trees, each cut to its first `parts` parts (default: all)."""
    reader = _reader(find(path).resolve())
    indices = np.arange(reader.count) if indices is None else np.asarray(indices)
    if indices.ndim != 1 or (indices.size and indices.dtype.kind not in "iu"):
        raise TypeError("indices must be a 1-D sequence of integers")
    indices = indices.astype(np.int64).tolist()
    sizes, part_ids, _, rows = _coding(str(find(assets))).read(
        reader, indices, None if parts is None else [parts] * len(indices)
    )
    return _trees(sizes, np.frombuffer(part_ids, "<u4"), np.frombuffer(rows, "<i8"))


def save_trees(trees: list[Tree], path: str | Path, assets: str | Path = "assets") -> None:
    """Write trees of class representatives colored 16 to a new directory of trees."""
    path = Path(path)
    if (path / "manifest.json").exists():
        raise FileExistsError(f"{path} holds trees already")
    coding = _coding(str(find(assets)))
    records = []
    for tree in trees:
        part_ids = tuple(part.part_id for part in tree.parts)
        if {part.color for part in tree.parts} != {16}:
            raise ValueError("a directory of trees holds no colors: color every part 16")
        if [edge.child for edge in tree.edges] != list(range(1, len(part_ids))):
            raise ValueError("edge k of a tree must attach part k + 1")
        records.append(coding.encode(list(part_ids), [_row(edge, part_ids) for edge in tree.edges], None))
    if not records:
        raise ValueError("no trees to write")
    _save_records(path, records)
