"""BrickNet part ids and their physical classes."""

import json
import lzma
from functools import lru_cache
from importlib.resources import files

import numpy as np


@lru_cache(maxsize=1)
def _classes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = json.loads(lzma.decompress(files("procbrick").joinpath("_data/classes.json.xz").read_bytes()))
    class_ids = np.asarray(raw["class_ids"], np.int64)
    representatives = np.asarray(raw["representatives"], np.int64)
    found = representatives[class_ids]
    canonical = np.where(found >= 0, found, np.arange(len(class_ids)))
    for array in (class_ids, representatives, canonical):
        array.setflags(write=False)
    return class_ids, representatives, canonical


def _checked(ids: list[int] | np.ndarray) -> np.ndarray:
    """`ids` as int64, checked against the catalog."""
    ids = np.asarray(ids, np.int64)
    if ids.size and (ids.min() < 0 or ids.max() >= len(class_ids())):
        raise IndexError(f"part ids must be in the catalog of {len(class_ids())} parts")
    return ids


def class_ids() -> np.ndarray:
    """Physical class of each part id."""
    return _classes()[0]


def representatives() -> np.ndarray:
    """Representative part id of each class; -1 for a class none of whose parts can connect."""
    return _classes()[1]


def canonical_id(ids: list[int] | np.ndarray) -> np.ndarray:
    """Representative of each part id's class (the part id itself if its class has none)."""
    return _classes()[2][_checked(ids)]
