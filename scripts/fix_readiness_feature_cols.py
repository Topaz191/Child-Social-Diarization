#!/usr/bin/env python3
"""把旧 ckpt / summary 里错误的 feature_cols 改成与 X 的 input_dim 一致（通常 18）。"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_train_mod():
    path = ROOT / "scripts" / "train_readiness_lstm.py"
    spec = importlib.util.spec_from_file_location("train_readiness_lstm", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    args = p.parse_args()
    data_dir = args.data_dir

    npz = data_dir / "readiness_samples.npz"
    if not npz.exists():
        raise FileNotFoundError(npz)

    X = np.load(npz)["X"]
    dim = int(X.shape[-1])
    mod = _load_train_mod()
    cols = mod.resolve_feature_cols(dim, None)
    mod.sync_feature_cols_metadata(data_dir, cols, dim)

    ckpt_path = data_dir / "readiness_model.pt"
    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        ckpt["feature_cols"] = cols
        ckpt["input_dim"] = dim
        torch.save(ckpt, ckpt_path)
        print("patched ckpt", ckpt_path, "→", dim, "cols")
    print("done", data_dir, "dim=", dim)


if __name__ == "__main__":
    main()
