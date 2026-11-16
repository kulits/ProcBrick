"""Procedural tree sampling."""

import contextlib
import hashlib
import json
import math
import multiprocessing
import os
import random
import secrets
import shutil
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from bricknet import Tree, load_catalog
from bricknet.collision import data_dir as bricknet_data_dir

from . import _core
from .trees import _coding, _seal, _trees, _write
from .paths import data_dir, find, sha256

PARTS = 100
_BLOCK = 256
_ASSETS = ("charts", "mates", "parts.json", "pose_dictionary", "assembly_model")
_WORKER = {}


def cpus() -> list[int]:
    """Processor affinity of this process."""
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    return list(range(os.cpu_count() or 1))


def load_sampling_policy(assets: str | Path = "assets") -> dict:
    """Part weights, axle seats, sampler options, and the distribution of per-assembly controls."""
    return json.loads((find(assets) / "sampling_policy.json").read_text(encoding="utf-8"))


def _controls(policy: dict, seed: int) -> dict:
    """Policy options and a draw of the assembly controls."""
    rng = random.Random(seed ^ 0x9E3779B9)
    distribution = policy["continuous_controls"]
    a, b, c = (rng.gauss(0, 1) for _ in range(3))
    rho = distribution["correlation"]
    latent = (a, b, rho * a + math.sqrt(1 - rho * rho) * c)
    packing, repetition, growth = (
        0.5 + 0.5 * math.tanh((math.log(m / (1 - m)) + s * z) / 2)
        for m, s, z in zip(distribution["median"], distribution["spread"], latent)
    )
    bounds = distribution["bounds"]

    def blend(key: str, value: float) -> float:
        low, high = bounds[key]
        return (1 - value) * low + value * high

    low, high = bounds["reuse_concentration"]
    return dict(
        policy["parameters"],
        envelope_penalty=blend("envelope_penalty", packing),
        envelope_ramp=blend("envelope_ramp", packing),
        reuse_concentration=math.exp((1 - repetition) * math.log(low) + repetition * math.log(high)),
        instance_reuse_probability=blend("instance_reuse_probability", repetition),
        extra_connection_ramp=blend("extra_connection_ramp", growth),
        connection_targets=(blend("connection_joins", growth), blend("connection_neighbors", growth)),
    )


def _cache(assets: str | Path) -> str:
    """Collider cache of `assets` in the data directory: BrickNet's colliders of its parts, parsed once from their OBJ
    files (each checked against its SHA-256 in parts.json) and memory-mapped by every sampler; built at first use."""
    parts = (find(assets) / "parts.json").read_bytes()
    path = data_dir() / "colliders.cache"
    if path.exists():
        with path.open("rb") as stream:
            if stream.read(40)[8:] == hashlib.sha256(parts).digest():
                return str(path)
        path.unlink()  # built for other parts
    root, stems = bricknet_data_dir() / "cvx", load_catalog().id_to_stem
    path.parent.mkdir(parents=True, exist_ok=True)
    packer = _core.GeometryPacker(str(path))
    for mesh, digest in json.loads(parts)["geometry"]:
        collider = root / f"{stems[mesh]}.cvx.obj"
        if not collider.exists():
            raise FileNotFoundError(f"{collider} is not installed; run `python -m procbrick fetch`")
        raw = collider.read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"{collider} is not the collider the atlas was built with")
        packer.add(mesh, raw)
    packer.finish(parts)
    return str(path)


def load_colliders(assets: str | Path = "assets", cache_bytes: int = 1 << 30) -> _core.Colliders:
    """BrickNet's colliders by part id: `bounds(part_ids)` and `clearances(part_ids, poses, pairs, radius)` of placed
    parts (LDU)."""
    return _core.Colliders(str(find(assets)), _cache(assets), cache_bytes)


class Sampler:
    """Tree sampler; one per process. `cache_bytes` and `collider_cache_bytes` bound its caches."""

    def __init__(
        self,
        assets: str | Path = "assets",
        policy: dict | None = None,
        cache_bytes: int = 512 << 20,
        collider_cache_bytes: int = 1 << 30,
    ) -> None:
        self.policy = load_sampling_policy(assets) if policy is None else policy
        self._sampler = _core.Sampler(
            str(find(assets)),
            _cache(assets),
            cache_bytes,
            collider_cache_bytes,
            self.policy["part_weights"],
            self.policy["root_part_weights"],
            [tuple(row) for row in self.policy["axle_seats"]],
        )

    def sample(self, seed: int | None = None, parts: int = PARTS, options: dict | None = None) -> Tree:
        """Tree of `parts` parts colored 16; edge k attaches part k + 1. `options` override the sampler options drawn
        for the tree."""
        part_ids, _, rows = self._sample(secrets.randbits(64) if seed is None else seed, parts, options)
        return _trees([len(part_ids)], part_ids, rows)[0]

    def _sample(self, seed: int, parts: int = PARTS, options: dict | None = None) -> tuple[list, list, list]:
        """Part ids, row-major poses, and edge rows."""
        if not 0 <= seed < 2**64:
            raise ValueError("seed must fit in uint64")
        drawn = _controls(self.policy, seed)
        unknown = set(options or {}) - set(drawn)
        if unknown:
            raise ValueError(f"unknown sampler options {', '.join(sorted(unknown))}")
        return self._sampler.sample(parts, seed, dict(drawn, **(options or {})))


def _finished(stage: Path, k: int, size: int) -> bool:
    """Whether block k of `size` trees is whole: size + 1 offsets, the last its data's size."""
    index, data = stage / f"{k:06d}.index", stage / f"{k:06d}.data"
    if not (index.exists() and data.exists()) or index.stat().st_size != 8 * (size + 1):
        return False
    return int(np.fromfile(index, "<u8")[-1]) == data.stat().st_size


def _start(assets: str, policy: dict, stage: Path) -> None:
    plan = json.loads((stage / "plan.json").read_text(encoding="utf-8"))
    _WORKER.update(
        sampler=Sampler(assets, policy),
        coding=_coding(assets),
        stage=stage,
        block=plan["block"],
        parts=plan["parts"],
        seeds=np.fromfile(stage / "seeds", "<u8"),
    )


def _write_block(k: int) -> int:
    """Sample a block and write its offsets last."""
    block = _WORKER["block"]
    records = []
    for seed in _WORKER["seeds"][k * block : (k + 1) * block]:
        part_ids, poses, edges = _WORKER["sampler"]._sample(int(seed), _WORKER["parts"])
        records.append(_WORKER["coding"].encode(part_ids, edges, np.array(poses, "<f8").tobytes()))
    offsets = np.cumsum([0] + [len(record) for record in records], dtype="<u8")
    _write(_WORKER["stage"] / f"{k:06d}.data", b"".join(records))
    _write(_WORKER["stage"] / f"{k:06d}.index", offsets.tobytes())
    return k


def generate(
    out: str | Path,
    count: int,
    workers: int | None = None,
    *,
    parts: int = PARTS,
    seed: int | None = None,
    assets: str | Path = "assets",
    policy: dict | None = None,
    block: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> None:
    """Sample trees into a new directory, resuming interrupted calls. Call under `if __name__ == "__main__":`."""
    if count < 1 or parts < 2:
        raise ValueError("count must be positive and parts at least 2")
    workers = len(cpus()) if workers is None else workers
    out = Path(out)
    if (out / "manifest.json").exists():
        raise FileExistsError(f"{out} holds trees already")
    stage = out / ".blocks"
    saved = None
    if (stage / "plan.json").exists():
        saved = json.loads((stage / "plan.json").read_text(encoding="utf-8"))
    if block is None:
        block = saved["block"] if saved else min(_BLOCK, -(-count // (4 * workers)))
    policy = load_sampling_policy(assets) if policy is None else policy
    assets = str(find(assets))
    _cache(assets)  # built once, before the workers start
    plan = dict(
        count=count,
        parts=parts,
        block=block,
        seed=seed,
        policy=hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest(),
        assets={name: sha256(Path(assets, name)) for name in _ASSETS},
        version=_core.__version__,
    )
    if saved is None:
        stage.mkdir(parents=True, exist_ok=True)
        seeds = np.random.default_rng(seed).integers(2**64, size=count, dtype=np.uint64)
        _write(stage / "seeds", seeds.astype("<u8").tobytes())
        _write(stage / "plan.json", json.dumps(plan).encode())
    elif saved != plan:
        raise ValueError(f"{stage} holds the blocks of another call; remove it to start over")
    total = -(-count // block)
    sizes = [min(block, count - k * block) for k in range(total)]
    pending = [k for k in range(total) if not _finished(stage, k, sizes[k])]
    done = total - len(pending)
    with contextlib.ExitStack() as stack:
        if workers > 1 and pending:
            context = multiprocessing.get_context("spawn")
            pool = stack.enter_context(
                ProcessPoolExecutor(
                    min(workers, len(pending)),
                    mp_context=context,
                    initializer=_start,
                    initargs=(assets, policy, stage.resolve()),
                )
            )
            finished = pool.map(_write_block, pending)
        else:
            stack.callback(_WORKER.clear)
            if pending:
                _start(assets, policy, stage.resolve())
            finished = map(_write_block, pending)
        for _ in finished:
            done += 1
            if progress:
                progress(done, total)
    damaged = [k for k in range(total) if not _finished(stage, k, sizes[k])]
    if damaged:
        raise RuntimeError(f"blocks {damaged} of {stage} are incomplete; run again to sample them")
    offsets = [np.zeros(1, "<u8")]
    with (out / "data").open("wb") as data:
        for k in range(total):
            offsets.append(np.fromfile(stage / f"{k:06d}.index", "<u8")[1:] + offsets[-1][-1])
            with (stage / f"{k:06d}.data").open("rb") as records:
                shutil.copyfileobj(records, data)
        data.flush()
        os.fsync(data.fileno())
    _seal(out, np.concatenate(offsets))
    shutil.rmtree(stage)
