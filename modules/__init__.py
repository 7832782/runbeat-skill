"""
RunBeat 核心模块包 (Modules)

本包包含 RunBeat 的 5 个核心处理模块，按工作流顺序排列：

模块1 - batch_bpm_detector.py  批量 BPM 识别
    输入: 音频文件夹（.mp3/.wav/.flac/.m4a/.ogg/.aac）
    输出: data/song_bpm_list.json（每首歌的原始 BPM、置信度等）
    方法: mixxx-analyzer（高精度，推荐）或 librosa（纯 Python 回退）
    特性: 支持断点续传、多位置采样、批量分析

模块2 - metronome_generator.py  节拍器音频生成
    输入: 目标 BPM、时长、拍号、强/弱拍音色参数
    输出: audio_output/metronome/metronome_{bpm}.wav
    算法: 指数衰减正弦波，强拍(1000Hz)与弱拍(800Hz)音色区分

模块3 - tempo_shifter.py  变速不变调处理
    输入: 原始音频 + song_bpm_list.json + 目标 BPM
    输出: audio_output/shifted_songs/{原名}_{目标bpm}bpm.wav
    方法: librosa.effects.time_stretch（相位声码器）或 soundstretch CLI
    特性: 严格模式（精确目标BPM）/ 非严格模式（最接近的因数，减少音质损失）

模块4 - audio_mixer.py  音频拼接与混音
    输入: 变速后的音频文件夹 + 首拍对齐数据 + 节拍器文件
    输出: audio_output/final_mix/running_mix.wav
    策略: 首拍对齐混音（每首歌首拍对齐节拍器重拍）+ 响度归一化 + 节拍器叠加

模块5 - beat_detector.py  首拍检测与对齐管理
    输入: 变速后的音频文件
    输出: data/beat_alignments.json（每首歌的首拍时间戳）
    算法: librosa onset detection + beat tracking 联合判断
    工具: BeatAlignmentManager 持久化存储，支持手动微调
"""

from .batch_bpm_detector import BatchBPMDetector, BPMAnalysisResult
from .metronome_generator import MetronomeGenerator, MetronomeConfig
from .tempo_shifter import TempoShifter, TempoShiftResult
from .audio_mixer import AudioMixer, MixConfig
from .beat_detector import FirstBeatDetector, BeatAlignmentManager, BeatDetectionResult

__all__ = [
    'BatchBPMDetector', 'BPMAnalysisResult',
    'MetronomeGenerator', 'MetronomeConfig',
    'TempoShifter', 'TempoShiftResult',
    'AudioMixer', 'MixConfig',
    'FirstBeatDetector', 'BeatAlignmentManager', 'BeatDetectionResult',
]
