#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块1: 批量音频 BPM 识别 (batch_bpm_detector.py)

批量分析音频文件夹，检测每首歌的原始 BPM（每分钟节拍数）。

输入:
    音频文件夹路径（支持 .mp3 / .wav / .flac / .m4a / .ogg / .aac）

输出:
    data/song_bpm_list.json
    格式: [{"file_path": "...", "file_name": "...", "original_bpm": 120.5, "confidence": 0.85, ...}]

两种检测方法:
    1. librosa — 纯 Python 实现，无需额外依赖
       在音频的不同位置采样多次，取中位数作为最终 BPM
       置信度 = 1 - std/20（多次采样结果越一致，置信度越高）
       参数: num_samples（采样次数，默认 3）、sample_duration（每次采样时长，默认 30s）

    2. mixxx — 基于 Mixxx DJ 软件分析引擎，更快更准
       使用 mixxx-analyzer 库，支持批量分析模式
       置信度默认为 1.0（可配合 beatgrid 信息做更精确判断）
       额外输出: key（调性）、camelot（Camelot 记谱）

特性:
    - 断点续传: 自动跳过已在 JSON 中的歌曲（通过 skip_existing 控制）
    - 置信度标注: 方便判断检测结果是否可靠
    - 批量加速: mixxx 支持 analyze_many() 一次分析多个文件

类和数据结构:
    BPMAnalysisResult   dataclass — BPM 分析结果
    BPMDetector         抽象基类 — 定义 analyze() 接口
    LibrosaBPMDetector  librosa 实现 — 多点采样 + 中位数
    MixxxBPMDetector    mixxx 实现 — 支持批量分析
    BatchBPMDetector    批量检测编排器 — 文件夹遍历 + 结果管理 + CLI
"""

import os
import json
import re
import warnings
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

import numpy as np


def natural_sort_key(s) -> List:
    """自然排序键函数 - 正确处理数字顺序"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]

# 忽略 librosa 的警告信息
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class BPMAnalysisResult:
    """BPM分析结果数据类"""
    file_path: str          # 音频文件完整路径
    file_name: str          # 音频文件名
    original_bpm: float     # 识别出的原始BPM
    confidence: float       # 识别置信度 (0-1)
    duration: float         # 音频时长(秒)
    sample_rate: int        # 采样率
    analysis_samples: int   # 用于分析的采样次数
    method: str             # 使用的检测方法
    key: str = ""           # 调性（mixxx 专用）
    camelot: str = ""       # Camelot 记谱（mixxx 专用）


class BPMDetector(ABC):
    """BPM检测器基类"""
    
    @abstractmethod
    def analyze(self, file_path: str) -> Optional[BPMAnalysisResult]:
        """分析单个音频文件的BPM"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """检测器名称"""
        pass


class LibrosaBPMDetector(BPMDetector):
    """基于librosa的传统BPM检测器"""
    
    def __init__(
        self,
        sample_duration: float = 30.0,
        num_samples: int = 3,
    ):
        self.sample_duration = sample_duration
        self.num_samples = num_samples
        self._librosa = None
        self._soundfile = None
        
    def _ensure_imports(self):
        """延迟导入，避免未使用时加载"""
        if self._librosa is None:
            import librosa
            import soundfile as sf
            self._librosa = librosa
            self._soundfile = sf
    
    @property
    def name(self) -> str:
        return "librosa"
    
    def analyze(self, file_path: str) -> Optional[BPMAnalysisResult]:
        """使用librosa分析BPM"""
        self._ensure_imports()
        librosa = self._librosa
        
        try:
            # 加载音频
            y, sr = librosa.load(file_path, sr=None, mono=True)
            duration = librosa.get_duration(y=y, sr=sr)
            
            # 如果音频太短，直接分析全部
            if duration <= self.sample_duration:
                onset_env = librosa.onset.onset_strength(y=y, sr=sr)
                bpm_values = [librosa.beat.tempo(onset_envelope=onset_env, sr=sr)]
            else:
                # 在音频的不同位置采样多次
                bpm_values = []
                hop_length = int((duration - self.sample_duration) / max(self.num_samples - 1, 1) * sr)

                for i in range(self.num_samples):
                    start_sample = int(i * hop_length) if hop_length > 0 else 0
                    end_sample = min(start_sample + int(self.sample_duration * sr), len(y))

                    if end_sample - start_sample < sr * 5:
                        continue

                    y_sample = y[start_sample:end_sample]
                    onset_env = librosa.onset.onset_strength(y=y_sample, sr=sr)
                    bpm = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
                    bpm_values.append(bpm)
            
            if not bpm_values:
                return None
            
            # 使用中位数作为最终BPM
            final_bpm = float(np.median(bpm_values))
            
            # 计算置信度
            if len(bpm_values) > 1:
                std = np.std(bpm_values)
                confidence = max(0, min(1, 1 - std / 20))
            else:
                confidence = 0.5
            
            return BPMAnalysisResult(
                file_path=file_path,
                file_name=os.path.basename(file_path),
                original_bpm=round(final_bpm, 2),
                confidence=round(confidence, 2),
                duration=round(duration, 2),
                sample_rate=sr,
                analysis_samples=len(bpm_values),
                method=self.name
            )
            
        except Exception as e:
            print(f"[ERROR] librosa分析失败 {os.path.basename(file_path)}: {e}")
            return None


class MixxxBPMDetector(BPMDetector):
    """基于 Mixxx 分析引擎的 BPM 检测器（更快更准）"""

    def __init__(self, use_batch: bool = True):
        self.use_batch = use_batch
        self._mixxx = None

    def _ensure_imports(self):
        if self._mixxx is None:
            from mixxx_analyzer import analyze, analyze_many
            self._mixxx_analyze = analyze
            self._mixxx_analyze_many = analyze_many

    @property
    def name(self) -> str:
        return "mixxx"

    def analyze(self, file_path: str) -> Optional[BPMAnalysisResult]:
        self._ensure_imports()
        try:
            result = self._mixxx_analyze(file_path)
            if result.bpm is None:
                return None

            duration = 0.0
            try:
                import soundfile as sf
                info = sf.info(file_path)
                duration = info.duration
                sr = info.samplerate
            except Exception:
                sr = 44100

            return BPMAnalysisResult(
                file_path=file_path,
                file_name=os.path.basename(file_path),
                original_bpm=round(result.bpm, 2),
                confidence=1.0,
                duration=round(duration, 2),
                sample_rate=int(sr),
                analysis_samples=len(result.beatgrid) if result.beatgrid else 0,
                method=self.name,
                key=result.key or "",
                camelot=result.camelot or "",
            )
        except Exception as e:
            print(f"[ERROR] mixxx分析失败 {os.path.basename(file_path)}: {e}")
            return None

    def analyze_many(self, file_paths: List[str]) -> List[Optional[BPMAnalysisResult]]:
        self._ensure_imports()
        results = []
        try:
            batch_results = self._mixxx_analyze_many(file_paths)
            result_map = {r.file: r for r in batch_results}
            for fp in file_paths:
                r = result_map.get(fp)
                if r is None or r.bpm is None:
                    results.append(None)
                    continue
                duration = 0.0
                sr = 44100
                try:
                    import soundfile as sf
                    info = sf.info(fp)
                    duration = info.duration
                    sr = info.samplerate
                except Exception:
                    pass
                results.append(BPMAnalysisResult(
                    file_path=fp,
                    file_name=os.path.basename(fp),
                    original_bpm=round(r.bpm, 2),
                    confidence=1.0,
                    duration=round(duration, 2),
                    sample_rate=int(sr),
                    analysis_samples=len(r.beatgrid) if r.beatgrid else 0,
                    method=self.name,
                    key=r.key or "",
                    camelot=r.camelot or "",
                ))
            return results
        except Exception as e:
            print(f"[ERROR] mixxx 批量分析失败: {e}")
            return [None] * len(file_paths)


class BatchBPMDetector:
    """
    批量BPM检测器
    
    支持多种检测方法，可批量分析音频文件的原生BPM
    """
    
    # 支持的音频格式
    SUPPORTED_FORMATS = ('.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac')
    
    def __init__(
        self,
        output_file: str = "song_bpm_list.json",
        method: str = "librosa",
        confidence_threshold: float = 0.6,
        **detector_kwargs
    ):
        """
        初始化BPM检测器

        Args:
            output_file: 输出JSON文件路径
            method: 检测方法 ("librosa" 或 "mixxx")
            confidence_threshold: 置信度阈值
            **detector_kwargs: 传递给具体检测器的参数
        """
        self.output_file = output_file
        self.confidence_threshold = confidence_threshold

        # 创建检测器
        if method == "librosa":
            self.detector = LibrosaBPMDetector(**detector_kwargs)
        elif method == "mixxx":
            self.detector = MixxxBPMDetector(**detector_kwargs)
        else:
            raise ValueError(f"不支持的检测方法: {method}")
        
        print(f"[INFO] 使用检测方法: {self.detector.name}")
        
        # 加载已存在的识别结果
        self.existing_results: Dict[str, BPMAnalysisResult] = {}
        self._load_existing_results()
    
    def _load_existing_results(self) -> None:
        """加载已存在的识别结果"""
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data:
                        result = BPMAnalysisResult(**item)
                        self.existing_results[result.file_path] = result
                print(f"[INFO] 已加载 {len(self.existing_results)} 条历史识别记录")
            except Exception as e:
                print(f"[WARNING] 加载历史记录失败: {e}")
                self.existing_results = {}
    
    def _save_results(self, results: List[BPMAnalysisResult]) -> None:
        """保存识别结果到JSON文件"""
        data = [asdict(r) for r in results]
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[INFO] 结果已保存至: {self.output_file}")
    
    def _get_audio_files(self, folder_path: str) -> List[str]:
        """获取文件夹中的所有音频文件"""
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"文件夹不存在: {folder_path}")
        
        audio_files = []
        for ext in self.SUPPORTED_FORMATS:
            audio_files.extend(folder.glob(f"*{ext}"))
            audio_files.extend(folder.glob(f"*{ext.upper()}"))
        
        audio_files = sorted(list(set([str(f) for f in audio_files])), key=natural_sort_key)
        return audio_files
    
    def analyze_folder(
        self,
        folder_path: str,
        skip_existing: bool = True,
        verbose: bool = True
    ) -> List[BPMAnalysisResult]:
        audio_files = self._get_audio_files(folder_path)

        if not audio_files:
            print(f"[WARNING] 在 {folder_path} 中未找到支持的音频文件")
            return []

        print(f"[INFO] 找到 {len(audio_files)} 个音频文件")
        print("-" * 60)

        results: List[BPMAnalysisResult] = []
        skipped_count = 0
        failed_count = 0

        new_files = []
        for file_path in audio_files:
            if skip_existing and file_path in self.existing_results:
                results.append(self.existing_results[file_path])
                skipped_count += 1
                if verbose:
                    print(f"[SKIP] 跳过: {os.path.basename(file_path)} - BPM: {self.existing_results[file_path].original_bpm}")
            else:
                new_files.append(file_path)

        if new_files:
            if hasattr(self.detector, 'analyze_many') and len(new_files) > 1:
                if verbose:
                    print(f"[INFO] 使用批量模式分析 {len(new_files)} 个新文件...")
                batch_results = self.detector.analyze_many(new_files)
                for file_path, result in zip(new_files, batch_results):
                    file_name = os.path.basename(file_path)
                    if result:
                        results.append(result)
                        confidence_icon = "[OK]" if result.confidence >= self.confidence_threshold else "[LOW]"
                        if verbose:
                            print(f"[ANALYZE] {file_name}: BPM={result.original_bpm:.1f} ({result.method}) {confidence_icon}")
                    else:
                        failed_count += 1
                        if verbose:
                            print(f"[FAIL] {file_name}")
            else:
                for idx, file_path in enumerate(new_files, 1):
                    file_name = os.path.basename(file_path)
                    if verbose:
                        print(f"[{idx}/{len(new_files)}] [ANALYZE] {file_name}...", end=" ")
                    result = self.detector.analyze(file_path)
                    if result:
                        results.append(result)
                        confidence_icon = "[OK]" if result.confidence >= self.confidence_threshold else "[LOW]"
                        if verbose:
                            print(f"BPM={result.original_bpm:.1f} ({result.method}) {confidence_icon}")
                    else:
                        failed_count += 1
                        if verbose:
                            print("[FAIL]")

        print("-" * 60)
        print(f"[INFO] 分析完成: 成功 {len(results) - skipped_count}, 跳过 {skipped_count}, 失败 {failed_count}")

        self._save_results(results)
        return results
    
    def print_summary(self, results: List[BPMAnalysisResult]) -> None:
        """打印分析结果摘要"""
        if not results:
            print("[INFO] 没有分析结果")
            return
        
        print("\n" + "=" * 60)
        print("[SUMMARY] BPM识别结果汇总")
        print("=" * 60)
        
        sorted_results = sorted(results, key=lambda x: x.original_bpm)
        
        print(f"{'文件名':<35} {'BPM':>8} {'方法':>12} {'置信度':>8}")
        print("-" * 60)
        
        for r in sorted_results:
            name = r.file_name[:33] + ".." if len(r.file_name) > 35 else r.file_name
            method_short = r.method[:10]
            confidence_icon = "✓" if r.confidence >= self.confidence_threshold else "?"
            print(f"{name:<35} {r.original_bpm:>8.1f} {method_short:>12} {r.confidence:>7.0%}{confidence_icon}")
        
        print("-" * 60)
        bpms = [r.original_bpm for r in results]
        methods = set(r.method for r in results)
        print(f"BPM范围: {min(bpms):.1f} - {max(bpms):.1f}, 平均: {np.mean(bpms):.1f}")
        print(f"检测方法: {', '.join(methods)}")
        print("=" * 60)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="批量音频BPM识别工具 - 支持librosa/mixxx两种方法",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用librosa（默认，快速）
  python batch_bpm_detector.py -i ./audio_input

  # 使用mixxx（更快更准，推荐）
  python batch_bpm_detector.py -i ./audio_input --method mixxx

  # 指定输出文件
  python batch_bpm_detector.py -i ./audio_input -o ./output/bpm_list.json
        """
    )
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    parser.add_argument(
        "-i", "--input",
        type=str,
        default=os.path.join(project_dir, "audio_input"),
        help="音频文件夹路径 (默认: 项目根目录下的 audio_input/)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=os.path.join(project_dir, "data", "song_bpm_list.json"),
        help="输出JSON文件路径"
    )
    parser.add_argument(
        "-m", "--method",
        type=str,
        choices=["librosa", "mixxx"],
        default="librosa",
        help="BPM检测方法 (默认: librosa)"
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="不跳过已识别的文件"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="采样次数 (仅librosa有效)"
    )
    parser.add_argument(
        "--sample-duration",
        type=float,
        default=30.0,
        help="采样时长秒数 (仅librosa有效)"
    )
    
    args = parser.parse_args()
    
    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)
    
    print("[RunBeat] 批量BPM识别模块")
    print("=" * 60)
    
    # 准备检测器参数
    detector_kwargs = {}
    if args.method == "librosa":
        detector_kwargs = {
            "num_samples": args.samples,
            "sample_duration": args.sample_duration,
        }
    
    # 创建检测器
    try:
        detector = BatchBPMDetector(
            output_file=output_path,
            method=args.method,
            **detector_kwargs
        )
    except ImportError as e:
        print(f"[ERROR] {e}")
        return 1
    
    # 执行分析
    results = detector.analyze_folder(
        folder_path=input_path,
        skip_existing=not args.no_skip,
        verbose=True
    )
    
    # 打印摘要
    detector.print_summary(results)
    
    if results:
        print(f"\n[DONE] 完成！结果已保存至: {output_path}")
        return 0
    else:
        print("\n[ERROR] 未识别到任何音频文件的BPM")
        return 1


if __name__ == "__main__":
    exit(main())
