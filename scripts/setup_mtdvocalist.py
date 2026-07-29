#!/usr/bin/env python3
"""下载 MTDVocaLiST 推理所需源码与预训练权重。"""

from __future__ import annotations

import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MTD_ROOT = ROOT / "third_party" / "MTDVocaLiST"
WEIGHTS = ROOT / "models" / "mtdvocalist" / "pure_MTDVocaLiST.pth"

FILES = {
    "models/__init__.py": "https://raw.githubusercontent.com/xjchenGit/MTDVocaLiST/main/models/__init__.py",
    "models/conv.py": "https://raw.githubusercontent.com/xjchenGit/MTDVocaLiST/main/models/conv.py",
    "models/student_thin_200_all.py": "https://raw.githubusercontent.com/xjchenGit/MTDVocaLiST/main/models/student_thin_200_all.py",
    "models/transformer_encoder_all.py": "https://raw.githubusercontent.com/xjchenGit/MTDVocaLiST/main/models/transformer_encoder_all.py",
}
WEIGHT_URL = "https://github.com/xjchenGit/MTDVocaLiST/releases/download/v1.0/pure_MTDVocaLiST.pth"


def main() -> None:
    for rel, url in FILES.items():
        dest = MTD_ROOT / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size > 0:
            print("skip", rel)
            continue
        print("download", rel)
        urllib.request.urlretrieve(url, dest)
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    if WEIGHTS.exists() and WEIGHTS.stat().st_size > 1_000_000:
        print("skip weights", WEIGHTS)
    else:
        print("download weights →", WEIGHTS)
        urllib.request.urlretrieve(WEIGHT_URL, WEIGHTS)
    print("done")
    print("  code:", MTD_ROOT)
    print("  weights:", WEIGHTS, WEIGHTS.stat().st_size)


if __name__ == "__main__":
    main()
