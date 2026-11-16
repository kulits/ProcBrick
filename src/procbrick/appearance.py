"""Print variants and materials of renders."""

from functools import lru_cache
from pathlib import Path

import numpy as np

from .catalog import class_ids
from .paths import find

_SCALES = np.exp(np.linspace(-14, 14, 129))  # reuse scales of `material_inventory_profile`
# palette cells: by distinct materials, then by the most common material's share
_PALETTE_SIZES = [0.5, 1.5, 2.5, 3.5, 5.5, 8.5, 12.5, 20.5, 100.5]
_DOMINANCE = [0, 0.25, 0.5, 0.75, 0.9, 1.01]


@lru_cache(maxsize=4)
def _policy(assets: str) -> dict:
    with np.load(Path(assets) / "appearance_policy.npz") as arrays:
        policy = {name: arrays[name] for name in arrays.files}
    policy["classes"] = class_ids()
    for array in policy.values():
        array.setflags(write=False)
    return policy


def load_appearance_policy(assets: str | Path = "assets") -> dict:
    """Variant and material weights, the materials and their LDraw colors, and the palette distribution (DATA.md)."""
    return dict(_policy(str(find(assets))))


def _choice(weights: np.ndarray, rng: np.random.Generator) -> int:
    """Index drawn with probability proportional to `weights`."""
    total = weights.sum()
    if not np.isfinite(total) or total <= 0 or weights.min() < 0:
        raise ValueError("categorical weights must be finite, nonnegative, and not all zero")
    cdf = (weights / total).cumsum()
    cdf /= cdf[-1]
    return int(cdf.searchsorted(rng.random(), side="right"))


def _latent(rng: np.random.Generator, theta: np.ndarray, size: int) -> np.ndarray:
    """Palette strength and material reuse of an assembly of `size` parts: bounded, correlated log-normals with a trend
    in size."""
    sd = np.exp(theta[2:4])
    covariance = np.diag(sd**2)
    covariance[0, 1] = covariance[1, 0] = theta[4] * np.prod(sd)
    mean = theta[:2] + theta[5:7] * np.log(size / 25)
    while True:
        x = rng.multivariate_normal(mean, covariance)
        if -2 * np.log(2) <= x[0] <= 8 * np.log(2) and -2 * np.log(2) <= x[1] <= 10 * np.log(2):
            return np.exp(x)


def _draw(classes, members, policy, rng, strength, reuse, diversity) -> tuple[np.ndarray, np.ndarray]:
    """One candidate appearance: variants and materials of parts of `classes`, with a palette of `strength`."""
    variant_classes = policy["classes"]
    material_p = policy["material_p"]
    palette = 0.99 * rng.dirichlet(strength * material_p) + 0.01 * material_p
    copies = np.maximum(np.bincount(classes, minlength=int(variant_classes.max()) + 1), 1)
    prior = policy["variant_strength"] * (len(classes) / 25) ** policy["variant_size_power"]
    pairs = copies * (copies - 1) / 2
    capable = np.bincount(variant_classes, minlength=len(copies)) > 1
    # scale the reuse of the classes with and without print variants to their inventory's material diversity
    for kind, target in enumerate(diversity):
        active = (pairs > 0) & (capable == kind)
        if not np.any(active):
            continue
        if not 0 < target < 1 or np.any(reuse[active] <= 0):
            raise ValueError("palette diversity must be in (0, 1), with positive reuse")
        rate, w = reuse[active], pairs[active] / pairs[active].sum()
        odds = 1 / target - 1
        low, high = odds / rate.max(), odds / rate.min()
        for _ in range(32):
            if low == high:
                break
            middle = np.sqrt(low * high)
            if (w / (1 + middle * rate)).sum() > target:
                low = middle
            else:
                high = middle
        reuse = np.where(active, reuse * np.sqrt(low * high), reuse)
    variants = np.empty(len(classes), np.int64)
    materials = np.empty(len(classes), np.int64)
    class_variants = {}
    class_materials = {}
    for i, cid in enumerate(classes):
        if cid not in class_variants:
            weights = prior[cid] * policy["variant_p"][members[cid]]
            class_variants[cid] = (weights, np.zeros(len(members[cid])))
            class_materials[cid] = np.zeros(len(material_p))
        weights, taken = class_variants[cid]
        choice = _choice(weights + taken, rng)
        variants[i] = members[cid][choice]
        taken[choice] += 1
        materials[i] = _choice(palette + reuse[cid] * class_materials[cid], rng)
        class_materials[cid][materials[i]] += 1
    return variants, materials


def sample_appearance(
    part_ids: list[int] | np.ndarray, rng: np.random.Generator, policy: dict | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Part ids with print variants and LDraw colors of an assembly's parts (2 or more)."""
    policy = load_appearance_policy() if policy is None else policy
    part_ids = np.asarray(part_ids, np.int64)
    size = len(part_ids)
    if part_ids.ndim != 1 or size < 2:
        raise ValueError("appearances are drawn for assemblies of 2 or more parts")
    row = min(size, len(policy["palette_weights"]) + 1) - 2
    classes = policy["classes"][part_ids]
    members = {cid: np.flatnonzero(policy["classes"] == cid) for cid in np.unique(classes)}
    draws = []
    for _ in range(int(policy["palette_candidates"])):
        strength, reuse = _latent(rng, policy["latent_parameters"], row + 2)
        diversity = policy["material_inventory_profile"][row] @ (1 / (1 + reuse * _SCALES))
        draws.append(_draw(classes, members, policy, rng, strength, reuse * policy["material_reuse"], diversity))
    counts = [np.unique(materials, return_counts=True)[1] for _, materials in draws]
    palette = np.searchsorted(_PALETTE_SIZES, [len(k) for k in counts], side="right") - 1
    dominance = np.searchsorted(_DOMINANCE, np.array([k.max() for k in counts]) / size, side="right") - 1
    weights = policy["palette_weights"][row][palette * 5 + dominance]
    variants, materials = draws[int(rng.choice(len(draws), p=weights / weights.sum()))]
    return variants, policy["colors"][materials]
