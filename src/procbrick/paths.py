"""Data directory and the released files."""

import hashlib
from pathlib import Path

from bricknet.__main__ import main as bricknet_main
from bricknet.collision import data_dir as bricknet_data_dir
from huggingface_hub import snapshot_download

REPO = "kulits/BrickFlow"
REVISION = "4db71f2622730f69c409ee39e3a0303060f6d991"
NAMES = ("assets", "train/pt", "train/ic", "val/procbrick")


def data_dir() -> Path:
    """`procbrick/` in BrickNet's data directory."""
    return bricknet_data_dir() / "procbrick"


def find(name: str | Path) -> Path:
    """`name` if it exists, else that file or directory in the data directory."""
    path = Path(name)
    if path.exists():
        return path
    installed = data_dir() / name
    if installed.exists():
        return installed
    for released in NAMES:
        if path.as_posix() == released or path.as_posix().startswith(f"{released}/"):
            raise FileNotFoundError(f"{installed} is not installed; run `python -m procbrick fetch {released}`")
    raise FileNotFoundError(f"{path} does not exist")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(names: str | list[str] = "assets") -> None:
    """Download released files into the data directory; with `assets`, also BrickNet's colliders if missing."""
    names = [names] if isinstance(names, str) else list(names)
    unknown = [name for name in names if name not in NAMES]
    if unknown:
        raise ValueError(f"unknown names {', '.join(unknown)}; the release has {', '.join(NAMES)}")
    patterns = [f"{name}/*" for name in names]
    snapshot_download(REPO, repo_type="dataset", revision=REVISION, allow_patterns=patterns, local_dir=data_dir())
    if "assets" in names and not (bricknet_data_dir() / "cvx").is_dir():
        bricknet_main(["fetch-meshes"])
