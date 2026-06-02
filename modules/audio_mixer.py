#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块4: 音频拼接与混音 (audio_mixer.py)

将多首变速后的音频按首拍对齐策略拼接为完整跑步音乐，支持响度均衡和节拍器叠加。

输入:
    - 变速后音频文件夹（audio_output/shifted_songs/）
    - 首拍对齐数据（data/beat_alignments.json）
    - 节拍器文件（可选，audio_output/metronome/）

输出:
    audio_output/final_mix/running_mix.wav

两种混音策略:
    1. mix_folder() — 简单顺序拼接
       所有歌曲按文件名排序，顺序拼接（直接 concatenate），每首歌做响度归一化
       适合不需要首拍对齐的简单场景

    2. mix_with_beat_alignment() — 首拍对齐混音（推荐）
       每首歌的首拍与节拍器的重拍（每 4 拍 = 1 小节）对齐:
       - 第 1 首歌: 从 0s 开始，其首拍时间作为节拍器参考起点
       - 后续歌曲: 首拍对齐到下一个空闲的重拍位置
       - 自动冲突检测: 如果下一首歌的开始时间早于当前歌的结束时间，
         自动推迟到下下个重拍
       - 使用 pydub.AudioSegment.overlay() 按时间轴叠加所有歌曲
       - 可选: 叠加节拍器音轨（从第一首歌的首拍时间开始）

响度归一化:
    - 先用 pydub.effects.normalize 将音频峰值拉到 0dBFS
    - 再 apply_gain() 调整到目标响度（默认 -12dBFS）
    - 保证所有歌曲音量一致，不会忽大忽小

类和数据结构:
    MixConfig   dataclass — 混音参数配置
    AudioMixer  混音器 — 包含加载、归一化、对齐、拼接、导出逻辑
"""

import os
import json
import argparse
import warnings
import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

import numpy as np
import soundfile as sf
from pydub import AudioSegment
from pydub.effects import normalize

warnings.filterwarnings("ignore")


def natural_sort_key(s: str) -> List:
    """自然排序键函数 - 正确处理数字顺序"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

# 导入首拍对齐管理器
from .beat_detector import BeatAlignmentManager


@dataclass
class MixConfig:
    """混音配置数据类"""
    target_loudness: float = -12.0   # 目标响度(dBFS)
    output_format: str = "wav"       # 输出格式
    output_quality: str = "high"     # 输出质量
    use_beat_alignment: bool = True  # 是否使用首拍对齐
    metronome_bpm: int = 180         # 节拍器BPM，用于计算对齐
    metronome_path: Optional[str] = None  # 节拍器音频文件路径
    metronome_volume: float = -0  # 节拍器音量(dB)，负值表示比音乐小


class AudioMixer:
    """
    音频混音器
    
    将多个音频文件拼接成一个完整的跑步音乐混音。
    """
    
    # 支持的音频格式
    SUPPORTED_FORMATS = ('.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac')
    
    def __init__(self, config: Optional[MixConfig] = None):
        """
        初始化音频混音器
        
        Args:
            config: 混音配置，使用默认配置如果为None
        """
        self.config = config or MixConfig()
        self.alignment_manager = BeatAlignmentManager() if BeatAlignmentManager else None
    
    def _get_audio_files(self, folder_path: str) -> List[str]:
        """
        获取文件夹中的所有音频文件
        
        Args:
            folder_path: 音频文件夹路径
            
        Returns:
            按文件名排序的音频文件路径列表
        """
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"文件夹不存在: {folder_path}")
        
        audio_files = []
        for ext in self.SUPPORTED_FORMATS:
            audio_files.extend(folder.glob(f"*{ext}"))
            audio_files.extend(folder.glob(f"*{ext.upper()}"))
        
        # 去重并按自然排序（正确处理数字顺序）
        audio_files = sorted(list(set([str(f) for f in audio_files])), key=natural_sort_key)
        return audio_files
    
    def _load_audio(self, file_path: str) -> AudioSegment:
        """
        加载音频文件为 AudioSegment
        
        Args:
            file_path: 音频文件路径
            
        Returns:
            AudioSegment 对象
        """
        try:
            audio = AudioSegment.from_file(file_path)
            return audio
        except Exception as e:
            print(f"[ERROR] 无法加载文件 {file_path}: {e}")
            raise
    
    def _apply_loudness_normalization(self, audio: AudioSegment) -> AudioSegment:
        """
        应用响度归一化
        
        使用 pydub 的 normalize 函数将音频归一化到目标响度。
        
        Args:
            audio: 输入音频
            
        Returns:
            归一化后的音频
        """
        # 计算当前响度 (dBFS)
        current_dbfs = audio.dBFS
        
        # 归一化到目标响度
        # pydub 的 normalize 会将音频峰值提升到 0dBFS，然后我们降低音量
        normalized = normalize(audio)
        
        # 调整音量到目标响度
        gain_change = self.config.target_loudness - normalized.dBFS
        adjusted = normalized.apply_gain(gain_change)
        
        return adjusted
    
    def _concatenate_clips(
        self,
        clips: List[AudioSegment]
    ) -> AudioSegment:
        """
        将多个音频片段进行交叉淡入淡出拼接
        
        Args:
            clips: 音频片段列表
            
        Returns:
            拼接后的音频
        """
        if not clips:
            raise ValueError("音频片段列表为空")
        
        if len(clips) == 1:
            return clips[0]
        
        # 从第一个片段开始
        # 直接拼接，不使用交叉淡入淡出
        result = clips[0]
        for clip in clips[1:]:
            result = result + clip
        return result
    
    def mix_folder(
        self,
        input_folder: str,
        output_path: str,
        file_order: Optional[List[str]] = None
    ) -> bool:
        """
        混音文件夹中的所有音频
        
        Args:
            input_folder: 输入音频文件夹
            output_path: 输出文件路径
            file_order: 自定义文件顺序，为None时按文件名排序
            
        Returns:
            是否成功
        """
        try:
            # 获取音频文件列表
            if file_order:
                audio_files = [
                    os.path.join(input_folder, f) for f in file_order
                ]
            else:
                audio_files = self._get_audio_files(input_folder)
            
            if not audio_files:
                print(f"[ERROR] 在 {input_folder} 中未找到音频文件")
                return False
            
            print(f"[INFO] 找到 {len(audio_files)} 个音频文件")
            print("-" * 60)
            
            # 加载并处理每个音频
            processed_clips = []
            total_duration = 0
            
            for idx, file_path in enumerate(audio_files, 1):
                file_name = os.path.basename(file_path)
                print(f"[{idx}/{len(audio_files)}] 处理: {file_name}")
                
                # 加载音频
                audio = self._load_audio(file_path)
                original_duration = len(audio) / 1000  # 转换为秒
                
                # 响度归一化
                audio = self._apply_loudness_normalization(audio)

                processed_clips.append(audio)
                total_duration += len(audio) / 1000

                print(f"       时长: {original_duration:.1f}s, 响度: {audio.dBFS:.1f}dBFS")

            print("-" * 60)
            print(f"[INFO] 开始拼接 {len(processed_clips)} 个音频片段...")

            # 直接拼接
            mixed = self._concatenate_clips(processed_clips)
            
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # 导出
            print(f"[INFO] 导出中...")
            mixed.export(
                output_path,
                format=self.config.output_format,
                parameters=["-q:a", "0"] if self.config.output_quality == "high" else []
            )
            
            final_duration = len(mixed) / 1000
            print("=" * 60)
            print("[SUMMARY] 混音完成")
            print("=" * 60)
            print(f"输入文件数: {len(audio_files)}")
            print(f"总时长: {final_duration:.1f}s ({final_duration/60:.1f}分钟)")
            print(f"输出文件: {output_path}")
            print(f"输出格式: {self.config.output_format.upper()}")
            print(f"采样率: {mixed.frame_rate}Hz")
            print(f"声道: {mixed.channels}")
            print("=" * 60)
            
            return True
            
        except Exception as e:
            print(f"[ERROR] 混音失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def mix_from_list(
        self,
        file_list: List[str],
        output_path: str
    ) -> bool:
        """
        从文件列表混音
        
        Args:
            file_list: 音频文件路径列表
            output_path: 输出文件路径
            
        Returns:
            是否成功
        """
        if not file_list:
            print("[ERROR] 文件列表为空")
            return False
        
        # 使用第一个文件的目录作为输入目录
        input_folder = os.path.dirname(file_list[0])
        file_names = [os.path.basename(f) for f in file_list]
        
        return self.mix_folder(input_folder, output_path, file_names)

    def _overlay_metronome(self, mixed_audio: AudioSegment, target_sample_rate: int, start_offset_ms: float = 0) -> AudioSegment:
        """
        将节拍器叠加到混音中

        策略：
        1. 加载节拍器音频（已根据歌曲总时长生成，无需循环）
        2. 调整音量和采样率
        3. 从指定偏移位置开始叠加到混音上

        Args:
            mixed_audio: 已混音的音频
            target_sample_rate: 目标采样率
            start_offset_ms: 节拍器开始播放的偏移时间（毫秒）

        Returns:
            叠加节拍器后的音频
        """
        try:
            # 加载节拍器
            metronome = AudioSegment.from_file(self.config.metronome_path)

            # 调整采样率匹配
            if metronome.frame_rate != target_sample_rate:
                metronome = metronome.set_frame_rate(target_sample_rate)

            # 调整声道匹配（转为单声道再根据需要扩展）
            if mixed_audio.channels == 2 and metronome.channels == 1:
                metronome = metronome.set_channels(2)

            # 调整音量
            metronome = metronome.apply_gain(self.config.metronome_volume)

            # 计算需要的节拍器长度
            mixed_duration_ms = len(mixed_audio)

            # 从节拍器开头截取，长度要覆盖从偏移位置到混音结束
            needed_duration_ms = mixed_duration_ms - int(start_offset_ms)
            if needed_duration_ms <= 0:
                print("[WARNING] 节拍器偏移时间超过混音时长")
                return mixed_audio

            # 截取节拍器（从开头开始）
            end_ms = min(needed_duration_ms, len(metronome))
            metronome_segment = metronome[:end_ms]

            # 叠加节拍器（从偏移位置开始）
            result = mixed_audio.overlay(metronome_segment, position=int(start_offset_ms))

            print(f"[INFO] 节拍器已叠加 (音量: {self.config.metronome_volume}dB, 从{start_offset_ms/1000:.3f}s开始, 使用{len(metronome_segment)/1000:.1f}s)")
            return result

        except Exception as e:
            print(f"[WARNING] 叠加节拍器失败: {e}")
            return mixed_audio

    def mix_with_beat_alignment(
        self,
        input_folder: str,
        output_path: str,
        file_order: Optional[List[str]] = None
    ) -> bool:
        """
        使用首拍对齐混音（新逻辑）

        策略：
        1. 检测节拍器的首拍位置（在第4步完成）
        2. 第一首歌的首拍对齐到节拍器的首拍
        3. 后续歌曲的首拍对齐到节拍器的下一个重拍（每隔4拍）
        4. 这样所有歌曲和节拍器完美对齐

        计算公式：
        - 每拍时长 = 60 / BPM 秒
        - 重拍间隔 = 4拍
        - 歌曲开始时间 = 目标重拍时间 - 歌曲首拍时间戳

        Args:
            input_folder: 输入音频文件夹
            output_path: 输出文件路径
            file_order: 自定义文件顺序

        Returns:
            是否成功
        """
        try:
            # 获取音频文件列表
            if file_order:
                audio_files = [
                    os.path.join(input_folder, f) for f in file_order
                    if os.path.exists(os.path.join(input_folder, f))
                ]
            else:
                audio_files = self._get_audio_files(input_folder)

            if not audio_files:
                print(f"[ERROR] 在 {input_folder} 中未找到音频文件")
                return False

            print(f"[INFO] 找到 {len(audio_files)} 个音频文件")
            print(f"[INFO] 节拍器BPM: {self.config.metronome_bpm}")
            print("-" * 60)

            # 计算节拍参数
            beat_duration_ms = (60.0 / self.config.metronome_bpm) * 1000  # 每拍时长(毫秒)
            measure_duration_ms = beat_duration_ms * 4  # 每小节时长（4拍）

            # 存储处理后的音频和它们的位置
            song_positions = []

            # 第一首歌从0秒开始，记录它的首拍时间作为节拍器起始偏移
            first_song_first_beat_ms = 0.0

            for idx, file_path in enumerate(audio_files, 1):
                file_name = os.path.basename(file_path)

                # 获取歌曲首拍时间戳
                song_first_beat_ms = 0.0
                if self.alignment_manager:
                    song_first_beat = self.alignment_manager.get_first_beat(file_path)
                    if song_first_beat is not None:
                        song_first_beat_ms = song_first_beat * 1000

                # 第一首歌：从0秒开始播放，记录首拍时间
                # 后续歌曲：首拍对齐到节拍器的下一个重拍，且不能与前一首歌重叠
                if idx == 1:
                    song_start_ms = 0  # 第一首歌从0开始
                    first_song_first_beat_ms = song_first_beat_ms
                    target_beat_time_ms = song_first_beat_ms  # 节拍器从第一首歌首拍开始
                    prev_song_end_ms = 0  # 用于检查重叠
                    print(f"[{idx}/{len(audio_files)}] {file_name}")
                    print(f"       歌曲首拍: {song_first_beat_ms/1000:.3f}s, 从0s开始播放")
                    print(f"       节拍器将从 {song_first_beat_ms/1000:.3f}s 开始")
                else:
                    # 计算这首歌应该对齐到节拍器的哪个重拍
                    # 从第一首歌首拍开始计算经过了多少拍
                    target_beat_time_ms = first_song_first_beat_ms + (current_measure * measure_duration_ms)
                    song_start_ms = target_beat_time_ms - song_first_beat_ms

                    # 确保不与上一首歌重叠：开始时间必须 >= 上一首歌的结束时间
                    if song_start_ms < prev_song_end_ms:
                        # 需要往后找下一个重拍，直到不重叠
                        original_start_ms = song_start_ms
                        while song_start_ms < prev_song_end_ms:
                            current_measure += 1  # 往后推一个小节
                            target_beat_time_ms = first_song_first_beat_ms + (current_measure * measure_duration_ms)
                            song_start_ms = target_beat_time_ms - song_first_beat_ms
                        print(f"[{idx}/{len(audio_files)}] {file_name}")
                        print(f"       歌曲首拍: {song_first_beat_ms/1000:.3f}s, "
                              f"目标重拍: {target_beat_time_ms/1000:.3f}s")
                        print(f"       [调整] 避免重叠: 从 {original_start_ms/1000:.3f}s 推迟到 {song_start_ms/1000:.3f}s")
                    else:
                        print(f"[{idx}/{len(audio_files)}] {file_name}")
                        print(f"       歌曲首拍: {song_first_beat_ms/1000:.3f}s, "
                              f"目标重拍: {target_beat_time_ms/1000:.3f}s")

                # 加载音频
                audio = self._load_audio(file_path)
                original_duration_ms = len(audio)

                # 响度归一化
                audio = self._apply_loudness_normalization(audio)

                song_positions.append({
                    'file_path': file_path,
                    'file_name': file_name,
                    'audio': audio,
                    'start_ms': song_start_ms,
                    'first_beat_ms': song_first_beat_ms,
                    'duration_ms': original_duration_ms
                })

                # 计算当前歌曲结束时间，用于检查下一首歌是否重叠
                song_end_ms = song_start_ms + original_duration_ms
                prev_song_end_ms = song_end_ms  # 更新上一首歌的结束时间

                # 计算下一首应该对齐到哪个重拍
                elapsed_from_metronome_start = song_end_ms - first_song_first_beat_ms
                current_beat = int(np.ceil(elapsed_from_metronome_start / beat_duration_ms))
                # 对齐到下一个重拍（4的倍数）
                next_measure_beat = ((current_beat // 4) + 1) * 4
                current_measure = next_measure_beat / 4

                if idx == 1:
                    print(f"       时长: {original_duration_ms/1000:.1f}s")
                    print(f"       结束于第{current_beat}拍（从节拍器开始算），下一首对齐到第{int(next_measure_beat)}拍")
                else:
                    print(f"       混音位置: {song_start_ms/1000:.3f}s, "
                          f"时长: {original_duration_ms/1000:.1f}s")
                    print(f"       结束于第{current_beat}拍，下一首对齐到第{int(next_measure_beat)}拍")
            
            print("-" * 60)
            print("[INFO] 开始合成...")
            
            # 计算总时长
            max_end_ms = max(
                pos['start_ms'] + len(pos['audio']) 
                for pos in song_positions
            )
            
            # 创建空白音频
            if song_positions:
                sample_rate = song_positions[0]['audio'].frame_rate
                channels = song_positions[0]['audio'].channels
                sample_width = song_positions[0]['audio'].sample_width

                # 创建空白音频
                mixed = AudioSegment.silent(
                    duration=int(max_end_ms),
                    frame_rate=sample_rate
                )

                # 将每首歌叠加到对应位置
                for pos in song_positions:
                    mixed = mixed.overlay(
                        pos['audio'],
                        position=int(pos['start_ms'])
                    )

                # 叠加节拍器（如果提供了节拍器文件）
                if self.config.metronome_path and os.path.exists(self.config.metronome_path):
                    print(f"[INFO] 叠加节拍器: {self.config.metronome_path}")
                    # 节拍器从第一首歌的首拍时间开始，这样第一首歌的首拍正好对齐节拍器的第0拍
                    mixed = self._overlay_metronome(mixed, sample_rate, start_offset_ms=first_song_first_beat_ms)

                # 确保输出目录存在
                output_dir = os.path.dirname(output_path)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)

                # 导出
                print(f"[INFO] 导出中...")
                mixed.export(
                    output_path,
                    format=self.config.output_format,
                    parameters=["-q:a", "0"] if self.config.output_quality == "high" else []
                )
                
                final_duration = len(mixed) / 1000
                print("=" * 60)
                print("[SUMMARY] 首拍对齐混音完成")
                print("=" * 60)
                print(f"输入文件数: {len(audio_files)}")
                print(f"总时长: {final_duration:.1f}s ({final_duration/60:.1f}分钟)")
                print(f"输出文件: {output_path}")
                print(f"节拍器BPM: {self.config.metronome_bpm}")
                print("=" * 60)
                
                return True
            
        except Exception as e:
            print(f"[ERROR] 首拍对齐混音失败: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """
    模块4主函数 - 测试入口

    使用示例:
        # 混音整个文件夹
        python audio_mixer.py --input ../audio_output --output ../running_mix.wav

        # 限制单首歌时长
        python audio_mixer.py --input ../audio_output --output ../running_mix.wav --max-duration 180
    """
    parser = argparse.ArgumentParser(
        description="音频拼接与混音工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法
  python audio_mixer.py --input ../audio_output --output ../running_mix.wav

  # 限制单首歌时长(秒)
  python audio_mixer.py --input ../audio_output --output ../running_mix.wav --max-duration 180

  # 自定义目标响度
  python audio_mixer.py --input ../audio_output --output ../running_mix.wav --loudness -14
        """
    )

    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="输入音频文件夹路径"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=os.path.join(project_dir, "running_mix.wav"),
        help=f"输出文件路径 (默认: {os.path.join(project_dir, 'running_mix.wav')})"
    )
    parser.add_argument(
        "--loudness",
        type=float,
        default=-16.0,
        help="目标响度(dBFS) (默认: -16)"
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help="单首歌最大时长(秒)，默认不限制"
    )

    args = parser.parse_args()

    # 转换为绝对路径
    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)

    print("[RunBeat] 音频拼接与混音模块")
    print("=" * 60)

    # 创建配置
    config = MixConfig(
        target_loudness=args.loudness,
    )

    print(f"[CONFIG] 目标响度: {config.target_loudness}dBFS")
    print("-" * 60)

    # 创建混音器并执行
    mixer = AudioMixer(config)
    success = mixer.mix_folder(input_path, output_path)

    if success:
        print(f"\n[DONE] 混音完成: {output_path}")
        return 0
    else:
        print("\n[ERROR] 混音失败")
        return 1


if __name__ == "__main__":
    exit(main())
