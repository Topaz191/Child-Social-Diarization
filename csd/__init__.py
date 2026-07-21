"""
CSD — Child Social Diarization
儿童社交态势说话人分离

基于动态可信度融合（MRAF 启发的模态欺骗建模，而非模态缺失）。
"""

__version__ = "0.2.0"

from csd.diarization.situational_diarizer import SituationalDiarizer, SituationalSegment

__all__ = ["SituationalDiarizer", "SituationalSegment", "__version__"]
