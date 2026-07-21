"""基于头部朝向的 pairwise 社交注意力。"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from csd.perception.head_pose import SlotVisualTimeline
from csd.core.utils import time_to_frame


class SocialAttentionComputer:
    """计算「谁在看谁」的软注意力分布。"""

    def __init__(self, temperature: float = 15.0):
        self.temperature = temperature

    @staticmethod
    def _angle_to_target(from_pos: Tuple[float, float], to_pos: Tuple[float, float]) -> float:
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        # 画面坐标：x 向右，y 向下；0° 表示朝画面上方
        return float(np.degrees(np.arctan2(dx, -dy)))

    @staticmethod
    def _wrap_angle(diff: float) -> float:
        return float((diff + 180.0) % 360.0 - 180.0)

    def attention_scores(
        self,
        speakers: List[str],
        positions: Dict[str, Tuple[float, float]],
        yaws: Dict[str, float],
    ) -> Dict[str, Dict[str, float]]:
        """
        返回 attention[from][to]：from 看向 to 的原始得分（未归一化）。
        含 elsewhere 桶在 softmax 前处理。
        """
        raw: Dict[str, Dict[str, float]] = {s: {} for s in speakers}
        for src in speakers:
            pos_i = positions.get(src)
            yaw_i = yaws.get(src, 0.0)
            if pos_i is None:
                continue
            scores = []
            targets = []
            for dst in speakers:
                if dst == src:
                    continue
                pos_j = positions.get(dst)
                if pos_j is None:
                    continue
                angle_ij = self._angle_to_target(pos_i, pos_j)
                diff = self._wrap_angle(angle_ij - yaw_i)
                score = max(0.0, float(np.cos(np.radians(diff))))
                raw[src][dst] = score
                scores.append(score)
                targets.append(dst)
            # elsewhere：未看向任何已知同伴
            if scores:
                max_to_other = max(scores)
                raw[src]["__elsewhere__"] = max(0.05, 1.0 - max_to_other)
            else:
                raw[src]["__elsewhere__"] = 1.0
        return raw

    @staticmethod
    def softmax_attention(raw: Dict[str, Dict[str, float]], temperature: float) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for src, dst_map in raw.items():
            if not dst_map:
                out[src] = {}
                continue
            keys = list(dst_map.keys())
            vals = np.array([dst_map[k] for k in keys], dtype=np.float64)
            logits = vals * temperature
            logits -= logits.max()
            probs = np.exp(logits)
            probs /= probs.sum() + 1e-8
            out[src] = {k: float(v) for k, v in zip(keys, probs)}
        return out

    def segment_attention(
        self,
        timeline: SlotVisualTimeline,
        speakers: List[str],
        start_time: float,
        end_time: float,
        fps: float,
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
        """
        对一个语音段聚合 pairwise 注意力。
        返回 (平均注意力矩阵, 各说话人收到的注意力 attention_received)。
        """
        start_f = time_to_frame(start_time, fps)
        end_f = time_to_frame(end_time, fps)

        accum: Dict[str, Dict[str, float]] = {s: {} for s in speakers}
        attn_recv_sum = {s: 0.0 for s in speakers}
        count = 0

        frame_indices = set()
        for slot_id, speaker in timeline.slot_to_speaker.items():
            for f in timeline.frames.get(slot_id, {}):
                if start_f <= f <= end_f:
                    frame_indices.add(f)

        for f in sorted(frame_indices):
            yaws: Dict[str, float] = {}
            positions: Dict[str, Tuple[float, float]] = {}
            for spk in speakers:
                slot_id = timeline.speaker_to_slot.get(spk)
                if slot_id is None:
                    continue
                pose = timeline.frames.get(slot_id, {}).get(f)
                if pose is None:
                    continue
                yaws[spk] = pose.yaw
                positions[spk] = timeline.slot_positions.get(slot_id, (0.5, 0.5))

            if len(yaws) < 2:
                continue

            raw = self.attention_scores(speakers, positions, yaws)
            soft = self.softmax_attention(raw, self.temperature)
            count += 1
            for src, dst_map in soft.items():
                for dst, val in dst_map.items():
                    if dst == "__elsewhere__":
                        continue
                    accum[src][dst] = accum[src].get(dst, 0.0) + val
                    attn_recv_sum[dst] = attn_recv_sum.get(dst, 0.0) + val

        if count == 0:
            return {s: {} for s in speakers}, {s: 0.0 for s in speakers}

        avg = {
            src: {dst: val / count for dst, val in dst_map.items()}
            for src, dst_map in accum.items()
        }
        attn_recv = {s: attn_recv_sum[s] / max(count * (len(speakers) - 1), 1) for s in speakers}
        return avg, attn_recv
