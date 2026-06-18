#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块: 首拍识别与对齐管理 (beat_detector.py)

输入:
    变速后的音频文件路径

输出:
    data/beat_alignments.json — 持久化的首拍时间戳存储

检测算法:
    1. 加载音频前 45 秒
    2. 80Hz 低通分离底鼓/贝斯
    3. 低频 onset (60%) + 全频 onset (40%) 混合检测
    4. librosa beat_track 节拍网格追踪
    5. RMS 能量起始检测（跳过前奏静音）
    6. 三维评分：低频强度 55% + 网格对齐 40% + 能量起始 5%
    7. 置信度评估
"""

import os
import json
import numpy as np
import librosa
from scipy.signal import butter, sosfilt
from typing import Optional, List
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
    首拍检测器 — beat_track + 三维评分。

    流程:
      1. 80Hz 低通分离低频
      2. onset 检测（低频 60% + 全频 40%）
      3. beat_track 节拍网格追踪
      4. RMS 能量起始检测
      5. 三维评分找首拍
    """

    PREVIEW_SEC = 45      # 预览时长
    ONSET_MIX = 0.6       # 低频 onset 混音比
    W_LF = 0.55           # 低频评分权重
    W_GRID = 0.40         # 网格对齐评分权重
    W_START = 0.05        # 能量起始权重
    TOL = 0.3             # 网格对齐容差（× beat_dur）

    def detect(self, audio_path: str) -> BeatDetectionResult:
        """完整首拍检测流水线。"""
        y, sr = librosa.load(audio_path, sr=None, mono=True,
                             duration=self.PREVIEW_SEC)

        # ① 低频分离
        sos = butter(4, 80.0 / (sr / 2), 'lowpass', output='sos')
        y_low = sosfilt(sos, y)

        # ② Onset: 低频 + 全频混合
        onset_env_low = librosa.onset.onset_strength(y=y_low, sr=sr)
        onset_env_full = librosa.onset.onset_strength(y=y, sr=sr)
        onset_env = onset_env_full * (1 - self.ONSET_MIX) + onset_env_low * self.ONSET_MIX

        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=sr,
            wait=3, pre_avg=3, post_avg=3, pre_max=3, post_max=3,
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)

        # ③ 节拍追踪
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        beat_dur = 60.0 / float(tempo) if float(tempo) > 0 else 0.5

        # ④ 能量起始（跳过前奏静音）
        rms = librosa.feature.rms(y=y)[0]
        rms_smooth = np.convolve(rms, np.ones(5) / 5, mode='same')
        noise_floor = np.percentile(rms_smooth, 10)
        energy_threshold = max(noise_floor * 1.5, np.max(rms_smooth) * 0.005)
        music_frames = np.where(rms_smooth > energy_threshold)[0]
        energy_start = (librosa.frames_to_time(music_frames[0], sr=sr)
                        if len(music_frames) > 0 else 0.0)

        # ⑤ 三维评分
        first_beat_time = self._find_first_strong_beat(
            sr, onset_times, beat_times, beat_dur, onset_env_low, energy_start,
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
        beat_dur: float,
        onset_env_low: np.ndarray,
        energy_start: float,
    ) -> float:
        """
        三维评分找首拍。

        评分维度:
          - 低频强度 × W_LF (0.55) — 底鼓/贝斯能量
          - 网格对齐 × W_GRID (0.40) — 距节拍竖线的偏差
          - 能量起始 × W_START (0.05) — 距音乐开始位置
        """
        if len(onset_times) == 0:
            return 0.0

        hop_length = 512

        # 候选池：所有 onset + 所有 beat + 前向外推 3 拍
        candidates = set()
        for t in onset_times:
            candidates.add(round(float(t), 4))
        for t in beat_times:
            candidates.add(round(float(t), 4))
        if len(beat_times) > 0:
            for k in range(1, 4):
                pre = float(beat_times[0]) - k * beat_dur
                if pre >= 0:
                    candidates.add(round(pre, 4))
        candidates = sorted(c for c in candidates if c >= 0)
        if len(candidates) == 0:
            return float(onset_times[0])

        # 评分
        env_max = max(float(np.max(onset_env_low)), 1e-10)
        beat_arr = np.array(beat_times, dtype=float)
        best_score = -1.0
        best_time = candidates[0]

        for t in candidates:
            frame = int(t * sr / hop_length)
            win = max(1, int(beat_dur * sr / hop_length * 0.2))

            # 低频强度
            lf_score = 0.0
            if frame < len(onset_env_low):
                s = max(0, frame - win)
                e = min(len(onset_env_low), frame + win)
                lf_score = float(np.mean(onset_env_low[s:e]) / env_max)
            lf_score = min(lf_score, 1.0)

            # 网格对齐
            grid_score = 0.0
            if len(beat_arr) > 0:
                dist = float(np.min(np.abs(beat_arr - t)))
                grid_score = max(0.0, 1.0 - dist / (beat_dur * self.TOL))

            # 能量起始
            start_bonus = max(0.0, 1.0 - abs(t - energy_start) / (beat_dur * 4))

            score = (lf_score * self.W_LF
                     + grid_score * self.W_GRID
                     + start_bonus * self.W_START)
            if score > best_score:
                best_score = score
                best_time = t

        return float(best_time)

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
    def __init__(self, data_dir: str = None, json_path: str = None):
        if json_path:
            self.data_dir = os.path.dirname(json_path)
            self.alignments_path = json_path
        elif data_dir:
            self.data_dir = data_dir
            self.alignments_path = os.path.join(data_dir, 'beat_alignments.json')
        else:
            self.data_dir = '.'
            self.alignments_path = 'beat_alignments.json'
        self.alignments: dict = {}
        self._load()

    def _load(self):
        if os.path.exists(self.alignments_path):
            try:
                with open(self.alignments_path, 'r', encoding='utf-8') as f:
                    self.alignments = json.load(f)
            except Exception:
                self.alignments = {}

    def _save(self):
        os.makedirs(self.data_dir, exist_ok=True)
        with open(self.alignments_path, 'w', encoding='utf-8') as f:
            json.dump(self.alignments, f, ensure_ascii=False, indent=2)

    def set_first_beat(self, song_path: str, first_beat_time: float):
        """保存某首歌的首拍时间"""
        self.alignments[song_path] = first_beat_time
        self._save()

    def get_first_beat(self, song_path: str) -> Optional[float]:
        """获取某首歌的首拍时间"""
        return self.alignments.get(song_path, None)

    def remove(self, song_path: str):
        """删除某首歌的首拍记录"""
        self.alignments.pop(song_path, None)
        self._save()

    def get_all_alignments(self) -> dict:
        """获取所有首拍时间"""
        return self.alignments.copy()

    def clear(self):
        """清空所有首拍记录"""
        self.alignments = {}
        self._save()

    def get_mix_start_times(
        self, song_paths: List[str], target_bpm: float,
        beats_per_measure: int = 4,
    ) -> List[float]:
        """计算每首歌的对齐开始时间"""
        if not song_paths:
            return []
        beat_dur = 60.0 / target_bpm
        measure_dur = beat_dur * beats_per_measure
        start_times = []
        current_time = 0.0
        for path in song_paths:
            fb = self.get_first_beat(path)
            if fb is None:
                start_times.append(current_time)
                continue
            # 对齐到下一个强拍位置
            target_beat_time = np.ceil(current_time / measure_dur) * measure_dur
            song_start = target_beat_time - fb
            if song_start < current_time:
                target_beat_time += measure_dur
                song_start = target_beat_time - fb
            start_times.append(max(song_start, 0.0))
            current_time = song_start + self._get_duration(path)
        return start_times

    @staticmethod
    def _get_duration(path: str) -> float:
        try:
            import soundfile as sf
            return sf.info(path).duration
        except Exception:
            return 30.0
