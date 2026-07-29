"""音–口同步：冻结 MTDVocaLiST 提特征 + 下游匹配头。"""

from csd.avsync.matcher import SyncMatcher
from csd.avsync.mtd_backend import MTDFeatureExtractor

__all__ = ["MTDFeatureExtractor", "SyncMatcher"]
