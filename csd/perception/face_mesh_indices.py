"""MediaPipe Face Mesh 常用索引子集（可视化 / 特征缓存共用）。"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

# 脸部外轮廓（稀疏点）
FACE_OVAL: Tuple[int, ...] = (
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
)

LEFT_EYE: Tuple[int, ...] = (33, 160, 158, 133, 153, 144)
RIGHT_EYE: Tuple[int, ...] = (362, 385, 387, 263, 373, 380)

# 唇环（与 visualize_visual_cues 一致，约 32 点闭环）
LIPS: Tuple[int, ...] = (
    61,
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    321,
    375,
    291,
    308,
    324,
    318,
    402,
    317,
    14,
    87,
    178,
    88,
    95,
    78,
    191,
    80,
    81,
    82,
    13,
    312,
    311,
    310,
    415,
    308,
)

NOSE: Tuple[int, ...] = (1, 2, 98, 327)

# MAR 四点：上唇内 / 下唇内 / 左嘴角 / 右嘴角
MAR4: Tuple[int, ...] = (13, 14, 61, 291)
MOUTH_TOP, MOUTH_BOTTOM, MOUTH_LEFT, MOUTH_RIGHT = MAR4

# PnP 六点（与 head_pose 一致）
PNP6: Tuple[int, ...] = (1, 152, 33, 263, 61, 291)

SCHEMA_LIP_POINT_IDS: List[str] = [f"lip_{i}" for i in range(len(LIPS))]
SCHEMA_MAR4_IDS: List[str] = ["mar_top", "mar_bottom", "mar_left", "mar_right"]


def index_catalog() -> Dict[str, Sequence[int]]:
    return {
        "face_oval": FACE_OVAL,
        "left_eye": LEFT_EYE,
        "right_eye": RIGHT_EYE,
        "lips": LIPS,
        "nose": NOSE,
        "mar4": MAR4,
        "pnp6": PNP6,
    }
