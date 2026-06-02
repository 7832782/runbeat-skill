#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块5: 首拍识别与对齐管理 (beat_detector.py)

功能:
    1. 自动检测音频文件的第一个重拍（首拍）位置
    2. 持久化管理多首歌曲的首拍时间戳
    3. 为混音模块提供对齐计算（歌曲首拍 → 节拍器重拍）

输入:
    变速后的音频文件路径

输出:
    data/beat_alignments.json — 持久化的首拍时间戳存储

检测算法:
    1. 加载音频前 15 秒（preview_duration，可配置）
    2. 80Hz 低通分离低频（分离底鼓/贝斯的 onset）
    3. 低频 onset（70% 权重）+ 全频 onset（30%）加权合成
    4. RMS 能量包络检测音乐真正开始位置，跳过前奏空白/环境音
    5. 节拍网格追踪 + 前向外推（处理弱起小节）
    6. 四维评分选首拍：低频强度 30% / 节拍网格对齐 35% / 优先选早 20% / 能量起始 15%
    7. 置信度评估: 基于首拍位置的 onset 强度与全局最大强度的比值

类和数据结构:
    BeatDetectionResult      dataclass — 单首歌曲的检测结果
    FirstBeatDetector        首拍检测器 — 分析音频，返回首拍时间
    BeatAlignmentManager     对齐管理器 — JSON 持久化存储，支持读写删

在混音中的作用:
    混音模块通过 BeatAlignmentManager 获取每首歌的首拍时间戳，
    计算歌曲应该在什么时间开始播放，使其首拍精确对齐到节拍器的重拍位置。
"""

import os
import json
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, sosfilt
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class BeatDetectionResult:
    """首拍检测结果"""
    file_path: str
    file_name: str
    first_beat_time: float
    confidence: float
    onset_times: List[float]
    beat_times: List[float]


class FirstBeatDetector:
    """
    首拍检测器 — 多阶段综合评分算法。

    1. 低频 onset 分析 (80Hz 低通，捕捉底鼓/贝斯)
    2. RMS 能量包络检测"音乐真正开始"
    3. 候选池构建 + 三维度评分
    """

    def __init__(self, preview_duration: float = 15.0):
        self.preview_duration = preview_duration

    def detect(self, audio_path: str) -> BeatDetectionResult:
        """
        多阶段首拍检测。

        Stages:
          1. 80Hz 低通分离低频
          2. 低频 onset (70%权重) + 全频 onset (30%) 合成
          3. 节拍网格追踪
          4. 综合评分找首拍
        """
        y, sr = librosa.load(audio_path, sr=None, mono=True,
                             duration=self.preview_duration)

        # 1. 低频分离 (80Hz 低通)
        sos = butter(4, 80.0 / (sr / 2), 'lowpass', output='sos')
        y_low = sosfilt(sos, y)

        # 2. Onset: 低频 + 全频加权合成
        onset_env_low = librosa.onset.onset_strength(y=y_low, sr=sr)
        onset_env_full = librosa.onset.onset_strength(y=y, sr=sr)
        onset_env = onset_env_full * 0.3 + onset_env_low * 0.7

        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=sr,
            wait=3, pre_avg=3, post_avg=3, pre_max=3, post_max=3,
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)

        # 3. 节拍跟踪 (用全频信号，低频只做 onset 权重)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)

        # 4. 能量起始检测 (跳过纯静音)
        rms = librosa.feature.rms(y=y)[0]
        rms_smooth = np.convolve(rms, np.ones(5) / 5, mode='same')
        noise_floor = np.percentile(rms_smooth, 10)
        energy_threshold = max(noise_floor * 1.5, np.max(rms_smooth) * 0.005)
        music_frames = np.where(rms_smooth > energy_threshold)[0]
        energy_start = (librosa.frames_to_time(music_frames[0], sr=sr)
                        if len(music_frames) > 0 else 0.0)

        # 5. 综合评分
        first_beat_time = self._find_first_strong_beat(
            sr, onset_times, beat_times, tempo, onset_env_low, energy_start,
        )

        confidence = self._calculate_confidence(
            first_beat_time, onset_times, onset_env, sr
        )

        return BeatDetectionResult(
            file_path=audio_path,
            file_name=os.path.basename(audio_path),
            first_beat_time=round(first_beat_time, 3),
            confidence=round(confidence, 2),
            onset_times=onset_times.tolist(),
            beat_times=beat_times.tolist(),
        )

    def _find_first_strong_beat(
        self,
        sr: int,
        onset_times: np.ndarray,
        beat_times: np.ndarray,
        tempo: float,
        onset_env_low: np.ndarray,
        energy_start: float,
    ) -> float:
        """
        综合评分找到第一个重拍。

        评分维度:
          - 低频 onset 强度 (40%) — 底鼓/贝斯能量
          - 节拍网格对齐 (40%)   — 与节拍网格偏差
          - 能量起始奖励 (20%)   — 越接近能量起始点越好
        """
        if len(onset_times) == 0:
            return 0.0

        beat_dur = 60.0 / float(tempo) if float(tempo) > 0 else 0.5

        # ---- 候选池: 所有 onset + 节拍网格 ±2 拍 ----
        candidates = set()
        for t in onset_times:
            candidates.add(round(t, 4))
        window_start = energy_start - beat_dur * 2
        window_end = energy_start + beat_dur * 4
        for t in beat_times:
            if window_start <= t <= window_end:
                candidates.add(round(t, 4))
        if len(beat_times) > 0:
            pre = beat_times[0] - beat_dur
            if pre >= 0:
                candidates.add(round(pre, 4))
            pre2 = pre - beat_dur
            if pre2 >= 0:
                candidates.add(round(pre2, 4))
            pre3 = pre2 - beat_dur
            if pre3 >= 0:
                candidates.add(round(pre3, 4))
        candidates = sorted(c for c in candidates if c >= 0)
        if len(candidates) == 0:
            return onset_times[0]

        # ---- 评分 ----
        hop_length = 512
        env_max = max(np.max(onset_env_low), 1e-10)
        best_time = candidates[0]
        best_score = -1.0
        beat_times_arr = np.array(beat_times, dtype=float)
        first_candidate_time = candidates[0]

        for t in candidates:
            frame = int(t * sr / hop_length)
            win = max(1, int(beat_dur * sr / hop_length * 0.2))

            # 低频 onset 强度 (30%)
            lf_score = 0.0
            if frame < len(onset_env_low):
                s = max(0, frame - win)
                e = min(len(onset_env_low), frame + win)
                lf_score = float(np.mean(onset_env_low[s:e]) / env_max)
            lf_score = min(lf_score, 1.0)

            # 节拍网格对齐 (35%)
            grid_score = 0.0
            if len(beat_times_arr) > 0:
                dist = float(np.min(np.abs(beat_times_arr - t)))
                grid_score = max(0.0, 1.0 - dist / (beat_dur * 0.3))

            # 优先选早 (20%): 越接近第一个候选分越高
            pos_in_candidates = (candidates.index(t) + 1) / len(candidates)
            firstness = 1.0 - pos_in_candidates * 0.5  # 0.5-1.0 range
            firstness = max(0.0, firstness)

            # 能量起始奖励 (15%)
            dist_from_start = abs(t - energy_start)
            start_bonus = max(0.0, 1.0 - dist_from_start / (beat_dur * 4)) * 0.15

            score = lf_score * 0.3 + grid_score * 0.35 + firstness * 0.2 + start_bonus
            if score > best_score:
                best_score = score
                best_time = t

        return best_time

    def _calculate_confidence(
        self,
        first_beat_time: float,
        onset_times: np.ndarray,
        onset_env: np.ndarray,
        sr: int,
    ) -> float:
        """基于 onset 强度比计算置信度"""
        if len(onset_times) == 0:
            return 0.0

        frame = librosa.time_to_frames(first_beat_time, sr=sr)
        if frame < len(onset_env):
            strength = onset_env[frame]
            max_s = np.max(onset_env)
            if max_s > 0:
                return min(1.0, strength / max_s * 1.5)
        return 0.5


class BeatAlignmentManager:
    """
    节拍对齐管理器

    管理多首歌曲的首拍时间戳，支持混音对齐
    """

    def __init__(self, json_path: str = None):
        if json_path is None:
            project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            json_path = os.path.join(project_dir, "data", "beat_alignments.json")
        self.json_path = json_path
        self.alignments = {}
        self._load()

    def _load(self):
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, 'r', encoding='utf-8') as f:
                    self.alignments = json.load(f)
            except Exception:
                self.alignments = {}

    def save(self):
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(self.alignments, f, ensure_ascii=False, indent=2)

    def set_first_beat(self, file_path: str, first_beat_time: float):
        self.alignments[file_path] = {
            'first_beat_time': first_beat_time,
            'file_name': os.path.basename(file_path),
        }
        self.save()

    def get_first_beat(self, file_path: str) -> Optional[float]:
        if file_path in self.alignments:
            return self.alignments[file_path]['first_beat_time']
        return None

    def get_metronome_first_beat(self) -> Optional[float]:
        return self.get_first_beat('__metronome__')

    def set_metronome_first_beat(self, first_beat_time: float):
        self.set_first_beat('__metronome__', first_beat_time)

    def calculate_aligned_start_time(
        self, file_path: str, metronome_bpm: int, prev_song_end_beat: int = 0,
    ) -> float:
        first_beat = self.get_first_beat(file_path)
        if first_beat is None:
            first_beat = 0.0
        beat_duration = 60.0 / metronome_bpm
        target = prev_song_end_beat * beat_duration
        return max(0, target - first_beat)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="首拍检测工具")
    parser.add_argument("--input", "-i", required=True, help="音频文件路径")
    args = parser.parse_args()

    detector = FirstBeatDetector()
    result = detector.detect(args.input)
    print(f"文件: {result.file_name}")
    print(f"首拍时间: {result.first_beat_time:.3f}s")
    print(f"置信度: {result.confidence:.2f}")
    print(f"检测到 {len(result.onset_times)} 个 onset")
    print(f"检测到 {len(result.beat_times)} 个 beat")
