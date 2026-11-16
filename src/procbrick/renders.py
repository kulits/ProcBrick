"""Rendered trees, cameras, and images."""

import functools
import io
import operator
import os
import tarfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
from bricknet import Part, Tree
from PIL import Image

from . import _core
from .appearance import load_appearance_policy
from .catalog import _checked, canonical_id, class_ids
from .trees import _coding as _tree_coding
from .trees import _reader, _save_records, _trees, save_trees
from .paths import find

_PAIR = 102  # parts of two partner renders
_OPEN_SHARDS = 256
_CAMERA = ("camera_world", "projection", "ldraw_to_world")
_FRAME = np.dtype([("offset", "<u8"), ("bytes", "<u4")] + [(name, "<f8", (4, 4)) for name in _CAMERA])


@functools.lru_cache(maxsize=4)
def _coding(assets: str) -> _core.Appearance:
    return _core.Appearance(assets, class_ids().tolist())


class Renders:
    """Rendered tree prefixes, indexed images, and cameras in a directory or a released split."""

    def __init__(self, root: str | Path, assets: str | Path = "assets") -> None:
        self.root = find(root)
        self._assets = assets
        self.sizes = np.load(self.root / "sizes.npy", mmap_mode="r")
        self.frames = np.load(self.root / "frames.npy", mmap_mode="r")
        self.shard_ids = np.load(self.root / "shard_ids.npy", mmap_mode="r")
        self._shards = OrderedDict()  # open shards of process `_pid`, least recently read first
        self._pid = os.getpid()

    @functools.cached_property
    def _readers(self) -> tuple[_core.Records, _core.Records, np.ndarray]:
        assets = str(find(self._assets))
        trees = _reader(self.root / "assemblies")
        appearance = _reader(self.root / "appearance")
        return trees, appearance, load_appearance_policy(assets)["colors"]

    def __len__(self) -> int:
        return len(self.sizes)

    def __getitem__(self, index: int) -> Tree:
        return self.read([index])[0]

    def __reduce__(self):
        return Renders, (self.root, self._assets)

    def read(self, indices: list[int] | np.ndarray) -> list[Tree]:
        """Trees of renders `indices`: their parts with print variants and LDraw colors, and their edges."""
        trees, _ = self._decode(indices)
        return trees

    def _decode(self, indices):
        indices = [self._index(i) for i in indices]
        trees, appearance, colors = self._readers
        assets = str(find(self._assets))
        sizes, part_ids, poses, rows = _tree_coding(assets).read(trees, indices, self.sizes[indices].tolist())
        starts = np.r_[0, np.cumsum(sizes)].tolist()
        variants, materials = _coding(assets).read(appearance, indices, np.frombuffer(part_ids, "<u4").tolist(), starts)
        colors = colors[np.frombuffer(materials, "<u2")]
        trees = _trees(sizes, np.frombuffer(variants, "<u4"), np.frombuffer(rows, "<i8"), colors)
        return trees, np.frombuffer(poses, "<f8").reshape(-1, 4, 4)

    def poses(self, index: int) -> np.ndarray:
        """Poses (n, 4, 4) of the parts of render `index`, float64 as the sampler placed them (LDU)."""
        index = self._index(index)
        _, _, poses, _ = _tree_coding(str(find(self._assets))).read(self._readers[0], [index], [int(self.sizes[index])])
        return np.frombuffer(poses, "<f8").reshape(-1, 4, 4).copy()

    def camera(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """`camera_world`, `projection`, and `ldraw_to_world` of render `index`: a point p (LDU) is at clip coordinates
        `projection @ inverse(camera_world) @ ldraw_to_world @ p`."""
        frame = self.frames[self._index(index)]
        return tuple(np.array(frame[name]) for name in _CAMERA)

    def paired(self) -> bool:
        """Whether every render has its partner."""
        sizes = np.asarray(self.sizes, np.int64)
        return len(sizes) % 2 == 0 and bool((sizes[0::2] + sizes[1::2] == _PAIR).all())

    def paired_index(self, index: int) -> int:
        """Partner of render `index`."""
        index = self._index(index)
        partner = index ^ 1
        if partner >= len(self) or int(self.sizes[index]) + int(self.sizes[partner]) != _PAIR:
            raise ValueError(f"render {index} has no partner")
        return partner

    def image_bytes(self, index: int) -> bytes:
        """AVIF image (RGBA) of render `index`."""
        index = self._index(index)
        frame = self.frames[index]
        shard = self.root / f"shards/{int(self.sizes[index]):03d}/{int(self.shard_ids[index]):05d}.tar"
        offset, size = int(frame["offset"]), int(frame["bytes"])
        if hasattr(os, "pread"):
            raw = os.pread(self._shard(shard).fileno(), size, offset)
        else:
            with open(shard, "rb", buffering=0) as stream:
                stream.seek(offset)
                raw = stream.read(size)
        if len(raw) != size:
            raise ValueError(f"{shard} is truncated")
        return raw

    def image(self, index: int) -> np.ndarray:
        """Image of render `index`, (H, W, 4) uint8 RGBA."""
        with Image.open(io.BytesIO(self.image_bytes(index))) as image:
            return np.asarray(image if image.mode == "RGBA" else image.convert("RGBA"))

    def _shard(self, path: Path):
        if self._pid != os.getpid():  # a child process opens its own
            self._shards = OrderedDict()
            self._pid = os.getpid()
        stream = self._shards.pop(path, None)
        if stream is None:
            stream = open(path, "rb", buffering=0)
            if len(self._shards) >= _OPEN_SHARDS:
                self._shards.popitem(last=False)
        self._shards[path] = stream
        return stream

    def _index(self, index: int) -> int:
        index = operator.index(index)
        if not -len(self) <= index < len(self):
            raise IndexError(f"render {index} is out of range ({len(self)})")
        return index % len(self)


def load_renders(path: str | Path, assets: str | Path = "assets") -> Renders:
    """Renders in a directory or a released name (`val/procbrick`)."""
    return Renders(path, assets)


def save_renders(
    trees: list[Tree],
    cameras: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    images: list[bytes],
    path: str | Path,
    assets: str | Path = "assets",
    shard_size: int = 1024,
) -> None:
    """Write renders to a new directory: render i shows `trees[i]` (parts with print variants and the colors of
    `load_appearance_policy()["colors"]`) through `cameras[i]` (`Renders.camera`) in the AVIF image `images[i]`."""
    out = Path(path)
    if not len(trees) == len(cameras) == len(images) > 0:
        raise ValueError("trees, cameras, and images must be as many as the renders")
    sizes = np.array([len(tree.parts) for tree in trees], np.int64)
    if sizes.min() < 2 or sizes.max() >= 1 << 16:
        raise ValueError("renders show 2 to 65,535 parts")
    assets = str(find(assets))
    material = {int(color): m for m, color in enumerate(load_appearance_policy(assets)["colors"])}
    coder = _coding(assets)
    records, shapes = [], []
    for i, tree in enumerate(trees):
        if not {part.color for part in tree.parts} <= material.keys():
            raise ValueError(f"render {i} has colors outside the appearance policy's")
        variants = _checked([part.part_id for part in tree.parts])
        records.append(coder.encode(variants.tolist(), [material[part.color] for part in tree.parts]))
        parts = tuple(Part(part, 16) for part in canonical_id(variants).tolist())
        shapes.append(Tree(parts, tree.edges))
    save_trees(shapes, out / "assemblies", assets)
    _save_records(out / "appearance", records)
    frames = np.zeros(len(sizes), _FRAME)
    shard_ids = np.zeros(len(sizes), "<u4")
    for size in np.unique(sizes):
        renders = np.flatnonzero(sizes == size)
        for shard_id, start in enumerate(range(0, len(renders), shard_size)):
            shard = renders[start : start + shard_size]
            path = out / f"shards/{size:03d}/{shard_id:05d}.tar"
            path.parent.mkdir(parents=True, exist_ok=True)
            with tarfile.open(path, "w") as archive:
                for i in shard:
                    member = tarfile.TarInfo(f"images/{size:03d}/{i:07d}.avif")
                    member.size = len(images[i])
                    archive.addfile(member, io.BytesIO(images[i]))
            with tarfile.open(path) as archive:
                members = archive.getmembers()
            frames["offset"][shard] = [member.offset_data for member in members]
            frames["bytes"][shard] = [member.size for member in members]
            shard_ids[shard] = shard_id
    for k, name in enumerate(_CAMERA):
        frames[name] = [camera[k] for camera in cameras]
    np.save(out / "sizes.npy", sizes.astype(np.uint8 if sizes.max() < 256 else np.uint16))
    np.save(out / "shard_ids.npy", shard_ids)
    np.save(out / "frames.npy", frames)
