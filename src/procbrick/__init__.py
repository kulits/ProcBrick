"""ProcBrick: Procedural Brick Assemblies."""

from . import _core
from .appearance import load_appearance_policy, sample_appearance
from .catalog import canonical_id, class_ids, representatives
from .paths import fetch
from .renders import Renders, load_renders, save_renders
from .sampler import Sampler, cpus, generate, load_colliders, load_sampling_policy
from .trees import load_trees, save_trees

__version__ = _core.__version__

__all__ = [
    "Renders",
    "Sampler",
    "canonical_id",
    "class_ids",
    "cpus",
    "fetch",
    "generate",
    "load_appearance_policy",
    "load_colliders",
    "load_renders",
    "load_sampling_policy",
    "load_trees",
    "representatives",
    "sample_appearance",
    "save_renders",
    "save_trees",
]
