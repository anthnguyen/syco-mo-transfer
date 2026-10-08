import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_env(root: Path = REPO_ROOT) -> None:
    """Load KEY=VALUE lines from <repo>/.env into os.environ (existing vars win)."""
    env = root / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def derive_seed(*parts) -> int:
    """Deterministic 31-bit seed from arbitrary parts (stable across processes, unlike hash())."""
    h = hashlib.sha256(json.dumps([str(p) for p in parts]).encode()).hexdigest()
    return int(h[:8], 16) & 0x7FFFFFFF


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)


def rng(*parts) -> np.random.Generator:
    return np.random.default_rng(derive_seed(*parts))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    # write-then-rename: a crash never leaves a truncated file that looks complete
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def write_json(path: Path, obj) -> None:
    _atomic_write_text(Path(path), json.dumps(obj, indent=2, default=_json_default) + "\n")


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def write_jsonl(path: Path, rows: list[dict]) -> None:
    _atomic_write_text(Path(path), "".join(json.dumps(r, default=_json_default) + "\n" for r in rows))


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def save_npz(path: Path, **arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o)}")
