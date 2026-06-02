#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
节拍器生成 — Audacity Rhythm Track 复刻版节拍器拍音效

基于 Audacity share/nyquist-plug-ins/rhythmtrack.ny 的 metronome-tick
1:1 翻译：白噪声 + 滤波 + 包络 + JC 混响。

输入:
    - 目标 BPM（如 180）
    - 总时长（秒）
    - 拍号（如 4/4 拍 = 每小节 4 拍）
    - 强/弱拍频率、音量

输出:
    audio_output/metronome/metronome_{bpm}.wav

合成算法:
    300 样本 LCG 白噪声 → PWEV 指数包络 → lowpass2(2×freq)
    → highpass8(freq) → 归一化 → JC 混响(叠加式) → PWLV 包络截断

类和数据结构:
    MetronomeConfig      dataclass — 节拍器参数配置
    MetronomeGenerator   节拍器生成器 — 合成、保存、信息查询
"""

import os
import argparse
import math
import numpy as np
import soundfile as sf
import scipy.signal
from typing import Optional, Tuple, List
from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════════════
#  DSP 工具函数
# ═══════════════════════════════════════════════════════════════════

def _biquad(sig, b0, b1, b2, a0, a1, a2):
    """通用双二阶滤波器 (Direct Form I)。"""
    b = np.array([b0 / a0, b1 / a0, b2 / a0])
    a = np.array([1.0, a1 / a0, a2 / a0])
    return scipy.signal.lfilter(b, a, sig)


def _lowpass2(sig, freq, q, sr):
    """RBJ 二极点谐振低通 (等效 Nyquist lowpass2)。"""
    w0 = 2 * math.pi * freq / sr
    w0 = min(w0, 0.95 * math.pi)
    q = max(q, 0.1)
    alpha = math.sin(w0) / (2 * q)
    cos_w0 = math.cos(w0)
    b0 = (1 - cos_w0) / 2
    b1 = 1 - cos_w0
    b2 = (1 - cos_w0) / 2
    a0 = 1 + alpha
    a1 = -2 * cos_w0
    a2 = 1 - alpha
    return _biquad(sig, b0, b1, b2, a0, a1, a2)


def _highpass2(sig, freq, q, sr):
    """RBJ 二极点高通。"""
    w0 = 2 * math.pi * freq / sr
    w0 = min(w0, 0.95 * math.pi)
    q = max(q, 0.1)
    alpha = math.sin(w0) / (2 * q)
    cos_w0 = math.cos(w0)
    b0 = (1 + cos_w0) / 2
    b1 = -(1 + cos_w0)
    b2 = (1 + cos_w0) / 2
    a0 = 1 + alpha
    a1 = -2 * cos_w0
    a2 = 1 - alpha
    return _biquad(sig, b0, b1, b2, a0, a1, a2)


def _highpass8(sig, freq, sr):
    """八阶 Butterworth 高通 (等效 Nyquist highpass8)。"""
    qs = (1.0 / (2 * np.sin(np.pi * (2 * k - 1) / 16))
          for k in (4, 3, 2, 1))
    for q in qs:
        sig = _highpass2(sig, freq, q, sr)
    return sig


def _comb_filter(sig, delay, gain):
    """梳状滤波器: y[n] = x[n] + gain * y[n-delay]"""
    b = np.array([1.0])
    a = np.zeros(delay + 1)
    a[0] = 1.0
    a[delay] = -gain
    return scipy.signal.lfilter(b, a, sig)


def _allpass_filter(sig, delay, gain):
    """全通滤波器: y[n] = -gain*x[n] + x[n-delay] + gain*y[n-delay]"""
    b = np.zeros(delay + 1)
    b[0] = -gain
    b[delay] = 1.0
    a = np.zeros(delay + 1)
    a[0] = 1.0
    a[delay] = -gain
    return scipy.signal.lfilter(b, a, sig)


def _jcrev(sig, wet_mix, sr):
    """
    JC 混响 (Chowning 型)，叠加式 additive 混音。
    6 路并联梳状 (g=0.84) → 3 路串联全通 (g=0.7) → dry + mix × wet
    """
    acc = np.zeros_like(sig)
    for d_ms in [50, 56, 61, 68, 72, 78]:
        d_samp = max(1, int(d_ms * sr / 1000))
        acc += _comb_filter(sig, d_samp, 0.84)
    wet = acc / 6
    for d_ms in [6, 11, 24]:
        d_samp = max(1, int(d_ms * sr / 1000))
        wet = _allpass_filter(wet, d_samp, 0.7)
    return sig + wet_mix * wet


# ─── 包络生成 ─────────────────────────────────────────────

def _pwev_envelope(levels_durs, sr):
    """分段指数包络 (等效 Nyquist pwev)。"""
    args = list(levels_durs)
    n = len(args)
    levels = args[0::2]
    durs = args[1::2]
    total_dur = sum(durs)
    n_samp = max(1, int(total_dur * sr))
    env = np.empty(n_samp)
    cur = 0.0
    for i in range(len(levels) - 1):
        v0, v1 = levels[i], levels[i + 1]
        dt = durs[i]
        s0 = int(cur * sr)
        s1 = int((cur + dt) * sr)
        s1 = min(s1, n_samp)
        seg_len = s1 - s0
        if seg_len > 0:
            t = np.linspace(0, 1, seg_len, endpoint=False)
            if v0 > 0 and v1 > 0:
                env[s0:s1] = np.exp(np.log(v0) + t * (np.log(v1) - np.log(v0)))
            else:
                env[s0:s1] = np.linspace(v0, v1, seg_len)
        cur += dt
    return env


def _pwlv_envelope(levels_durs, sr):
    """分段线性包络 (等效 Nyquist pwlv)。"""
    args = list(levels_durs)
    n = len(args)
    levels = args[0::2]
    durs = args[1::2]
    total_dur = sum(durs)
    n_samp = max(1, int(total_dur * sr))
    env = np.empty(n_samp)
    cur = 0.0
    for i in range(len(levels) - 1):
        v0, v1 = levels[i], levels[i + 1]
        dt = durs[i]
        s0 = int(cur * sr)
        s1 = int((cur + dt) * sr)
        s1 = min(s1, n_samp)
        if s1 > s0:
            env[s0:s1] = np.linspace(v0, v1, s1 - s0)
        cur += dt
    return env


# ─── 节拍器拍合成 ────────────────────────────────────────

def _generate_metronome_click(freq_hz, volume, sample_rate):
    """
    生成单个节拍器拍 — 高保真复刻 Audacity metronome-tick。

    Args:
        freq_hz: 基频 (Hz)，决定 lowpass2/highpass8 的滤波中心
        volume: 音量 (0-1)
        sample_rate: 采样率

    Returns:
        numpy float32 数组
    """
    sr = sample_rate
    hz = freq_hz

    # 1) 300 样本 LCG 白噪声
    ln = 300
    sig = np.zeros(ln)
    x = 1
    for i in range(ln):
        x = (479 * x) % 997
        sig[i] = (x / 500.0) - 1.0

    # 2) 嵌入 0.2s 静音
    total = int(0.2 * sr)
    buf = np.zeros(total)
    buf[:ln] = sig[:min(ln, total)]
    sig = buf

    # 3) PWEV 包络: (10, ln/sr, 2, 1, 0)
    env = _pwev_envelope([10.0, ln / sr, 2.0, 1.0, 0.0], sr)
    env = np.abs(env)
    if len(env) > len(sig):
        env = env[:len(sig)]
    else:
        env = np.pad(env, (0, len(sig) - len(env)), 'constant')
    sig = sig * env

    # 4) lowpass2(2*hz, Q=6) → highpass8(hz)
    sig = _lowpass2(sig, 2.0 * hz, 6.0, sr)
    sig = _highpass8(sig, hz, sr)

    # 5) 归一化
    pk = np.max(np.abs(sig[:min(300, len(sig))]))
    if pk > 0:
        sig = sig * (volume / pk)
    else:
        sig = sig * volume

    # 6) JC 混响 (叠加式 22%)
    sig = _jcrev(sig, 0.22, sr)

    # 7) PWLV 包络: pwlv(1.11, 0.02, 1.11, 0.05, 0)
    env_pwlv = _pwlv_envelope([1.11, 0.02, 1.11, 0.05, 0.0], sr)
    env_pwlv = np.abs(env_pwlv)
    if len(env_pwlv) > len(sig):
        env_pwlv = env_pwlv[:len(sig)]
    else:
        env_pwlv = np.pad(env_pwlv, (0, len(sig) - len(env_pwlv)), 'constant')
    sig = sig * env_pwlv

    return sig.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════
#  配置 & 生成器
# ═══════════════════════════════════════════════════════════════════

@dataclass
class MetronomeConfig:
    """节拍器配置数据类"""
    bpm: int = 180                    # 目标BPM
    duration: float = 30.0            # 总时长(秒)
    beats_per_measure: int = 4        # 每小节拍数
    sample_rate: int = 44100          # 采样率

    # 强拍(第一拍)音色参数
    strong_freq: float = 1000.0       # 强拍基频 (Hz)，决定滤波中心
    strong_volume: float = 0.8        # 强拍音量 (0-1)

    # 弱拍音色参数
    weak_freq: float = 800.0          # 弱拍基频 (Hz)
    weak_volume: float = 0.5          # 弱拍音量 (0-1)


class MetronomeGenerator:
    """
    节拍器生成器

    生成指定 BPM 的节拍器音频，使用 Audacity Rhythm Track 复刻合成引擎。
    """

    def __init__(self, config: Optional[MetronomeConfig] = None):
        self.config = config or MetronomeConfig()

    def _generate_click(
        self,
        frequency: float,
        volume: float,
        sample_rate: int
    ) -> np.ndarray:
        """生成单个节拍器拍 (Audacity 复刻版)。"""
        return _generate_metronome_click(frequency, volume, sample_rate)

    def generate(
        self,
        output_path: Optional[str] = None
    ) -> Tuple[np.ndarray, int]:
        """
        生成节拍器音频

        Args:
            output_path: 输出文件路径，为None时不保存文件

        Returns:
            (音频数组, 采样率)
        """
        config = self.config
        sr = config.sample_rate

        total_samples = int(config.duration * sr)
        audio = np.zeros(total_samples, dtype=np.float32)

        beat_interval = int(60.0 / config.bpm * sr)

        # 预生成强拍和弱拍声音
        strong_click = self._generate_click(
            config.strong_freq, config.strong_volume, sr
        )
        weak_click = self._generate_click(
            config.weak_freq, config.weak_volume, sr
        )

        # 在音频中放置节拍
        beat_count = 0
        for pos in range(0, total_samples, beat_interval):
            is_strong = (beat_count % config.beats_per_measure) == 0
            click = strong_click if is_strong else weak_click

            end_pos = min(pos + len(click), total_samples)
            click_len = end_pos - pos
            audio[pos:end_pos] += click[:click_len]
            beat_count += 1

        # 防削波
        max_amp = np.max(np.abs(audio))
        if max_amp > 1.0:
            audio = audio / max_amp * 0.95

        if output_path:
            sf.write(output_path, audio, sr)
            print(f"[INFO] 节拍器已保存: {output_path}")
            print(f"       BPM: {config.bpm}, 时长: {config.duration}s, "
                  f"拍号: {config.beats_per_measure}/4, 音色: Audacity 复刻版")

        return audio, sr

    def get_info(self) -> dict:
        """获取节拍器信息"""
        return {
            "bpm": self.config.bpm,
            "duration": self.config.duration,
            "beats_per_measure": self.config.beats_per_measure,
            "sample_rate": self.config.sample_rate,
            "total_beats": int(self.config.duration / (60.0 / self.config.bpm)),
            "strong_freq": self.config.strong_freq,
            "weak_freq": self.config.weak_freq,
            "engine": "Audacity Rhythm Track 复刻版",
        }


def main():
    parser = argparse.ArgumentParser(
        description="节拍器生成工具 (Audacity 复刻版)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python metronome_generator.py --bpm 180
  python metronome_generator.py --bpm 180 --duration 60
  python metronome_generator.py --bpm 180 --strong-freq 1200 --weak-freq 800
        """,
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    parser.add_argument("--bpm", type=int, required=True, help="目标BPM")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="节拍器时长(秒)")
    parser.add_argument("--beats-per-measure", type=int, default=4,
                        help="每小节拍数")
    parser.add_argument("--strong-freq", type=float, default=1000.0,
                        help="强拍频率(Hz)")
    parser.add_argument("--weak-freq", type=float, default=800.0,
                        help="弱拍频率(Hz)")
    parser.add_argument("--strong-volume", type=float, default=0.75,
                        help="强拍音量")
    parser.add_argument("--weak-volume", type=float, default=0.5,
                        help="弱拍音量")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="输出文件路径")

    args = parser.parse_args()

    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        output_path = os.path.join(
            project_dir, "audio_output", "metronome",
            f"metronome_{args.bpm}.wav"
        )
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    config = MetronomeConfig(
        bpm=args.bpm,
        duration=args.duration,
        beats_per_measure=args.beats_per_measure,
        strong_freq=args.strong_freq,
        weak_freq=args.weak_freq,
        strong_volume=args.strong_volume,
        weak_volume=args.weak_volume,
    )

    generator = MetronomeGenerator(config)
    print("[RunBeat] 节拍器生成模块 (Audacity 复刻版)")
    print("=" * 60)
    print(f"[CONFIG] BPM: {config.bpm}")
    print(f"[CONFIG] 时长: {config.duration}s")
    print(f"[CONFIG] 拍号: {config.beats_per_measure}/4")
    print(f"[CONFIG] 强拍: {config.strong_freq}Hz, 音量{config.strong_volume}")
    print(f"[CONFIG] 弱拍: {config.weak_freq}Hz, 音量{config.weak_volume}")
    print("-" * 60)

    audio, sr = generator.generate(output_path)
    info = generator.get_info()
    print(f"[INFO] 总拍数: {info['total_beats']}")
    print(f"[INFO] 音频长度: {len(audio)/sr:.2f}s")
    print("=" * 60)
    print(f"[DONE] 节拍器已生成: {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
