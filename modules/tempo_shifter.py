#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块3: 变速不变调处理 (tempo_shifter.py)

将音频从原始 BPM 变速到目标 BPM，保持音调不变（时间拉伸，非重采样）。

输入:
    - 音频文件 + 原始 BPM（从 data/song_bpm_list.json 读取）
    - 目标 BPM（用户设定，如 180）

输出:
    audio_output/shifted_songs/{原文件名}_{实际BPM}bpm.wav

实现策略（按优先级）:
    1. soundstretch CLI — 专业音频时间拉伸工具，基于相位声码器，音质最好
    2. librosa.effects.time_stretch — Python 库实现，音质略逊但无需额外安装

变速比例计算:
    rate = 目标BPM / 原始BPM
    例: 原 120BPM → 目标 180BPM → rate = 1.5 (加速 50%，时长变短)

两种模式:
    严格模式 (strict_mode=True):
        直接变速到目标 BPM，不管变化幅度多大
        例: 原 95 → 目标 180，rate = 1.89 (加速 89%，音质可能受损)

    非严格模式 (strict_mode=False):
        找目标 BPM 的因数/倍数中最接近原始 BPM 的值
        例: 原 95 → 目标 180 → 候选: [180/2=90, 180/3=60, 180/4=45, 180]
        → 选 90（95 到 90 变化最小），rate = 0.95
        优势: 变化幅度最小，音质损失最低

类和数据结构:
    TempoShiftResult  dataclass — 单首歌曲的处理结果
    TempoShifter      变速处理器 — 单文件/批量处理、CLI 入口
"""

import os
import json
import shutil
import subprocess
import argparse
import warnings
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass

import numpy as np
import librosa
import soundfile as sf

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class TempoShiftResult:
    """变速处理结果数据类"""
    input_path: str
    output_path: str
    original_bpm: float
    target_bpm: int
    stretch_rate: float
    method: str
    success: bool
    error_msg: Optional[str] = None


class TempoShifter:
    """
    变速不变调处理器
    
    将音频从原始BPM变速到目标BPM，保持音调不变。
    """
    
    def __init__(self, target_bpm: int = 180, method: str = "auto"):
        """
        初始化变速处理器
        
        Args:
            target_bpm: 目标BPM
            method: 处理方法 ("auto", "soundstretch", "librosa")
        """
        self.target_bpm = target_bpm
        self.method = method
        
        # 检测可用的处理方法
        self._check_available_methods()
    
    def _check_available_methods(self) -> None:
        """检查可用的处理方法"""
        self.has_soundstretch = shutil.which("soundstretch") is not None
        
        if self.method == "auto":
            if self.has_soundstretch:
                self.selected_method = "soundstretch"
            else:
                self.selected_method = "librosa"
        else:
            self.selected_method = self.method
            if self.method == "soundstretch" and not self.has_soundstretch:
                print("[WARNING] soundstretch 不可用，将使用 librosa")
                self.selected_method = "librosa"
    
    def _calculate_stretch_rate(self, original_bpm: float, strict_mode: bool = True) -> float:
        """
        计算变速比例
        
        Args:
            original_bpm: 原始BPM
            strict_mode: 是否严格模式。严格模式直接变速到目标BPM；
                        非严格模式会找目标BPM的因数，使变化幅度最小
            
        Returns:
            变速比例 (实际目标BPM / 原始BPM)
        """
        if original_bpm <= 0:
            raise ValueError(f"原始BPM必须大于0: {original_bpm}")
        
        if strict_mode:
            # 严格模式：直接变速到目标BPM
            actual_target_bpm = self.target_bpm
        else:
            # 非严格模式：找目标BPM的因数，使变化幅度最小
            actual_target_bpm = self._find_optimal_target_bpm(original_bpm)
        
        return actual_target_bpm / original_bpm
    
    def _find_optimal_target_bpm(self, original_bpm: float) -> int:
        """
        非严格模式下，找到使变化幅度最小的目标BPM
        
        算法：找目标BPM的因数（包括通过乘2得到的），
              使得 |original_bpm - candidate| / original_bpm 最小
        
        例如：
        - 原BPM=95，目标BPM=180
          候选：180, 90(180/2), 60(180/3), 45(180/4)...
          95到90的变化幅度最小，所以返回90
        
        Args:
            original_bpm: 原始BPM
            
        Returns:
            最优的目标BPM
        """
        target = self.target_bpm
        
        # 生成候选BPM列表（目标BPM的因数，以及通过乘2扩展的）
        candidates = set()
        
        # 添加目标BPM本身
        candidates.add(target)
        
        # 添加目标BPM的因数（通过除2, 除3, 除4...）
        for divisor in range(2, 9):  # 最多除到8
            candidate = target / divisor
            if candidate >= 40:  # BPM不能太低
                candidates.add(candidate)
        
        # 添加目标BPM的倍数（通过乘2）
        candidates.add(target * 2)
        
        # 转换为列表并排序
        candidates = sorted(list(candidates))
        
        # 找变化幅度最小的候选
        # 变化幅度 = |original_bpm - candidate| / original_bpm
        best_candidate = target
        min_change_ratio = abs(original_bpm - target) / original_bpm
        
        for candidate in candidates:
            change_ratio = abs(original_bpm - candidate) / original_bpm
            if change_ratio < min_change_ratio:
                min_change_ratio = change_ratio
                best_candidate = candidate
        
        return int(round(best_candidate))
    
    def _shift_with_soundstretch(
        self,
        input_path: str,
        output_path: str,
        stretch_rate: float
    ) -> bool:
        """
        使用 soundstretch 进行变速处理
        
        soundstretch 是一个高质量的音频时间拉伸工具，
        使用相位声码器算法，音质比 librosa 更好。
        
        Args:
            input_path: 输入音频路径
            output_path: 输出音频路径
            stretch_rate: 变速比例
            
        Returns:
            是否成功
        """
        try:
            # soundstretch 使用百分比参数
            # rate = 1.5 表示加速50%，对应 tempo +50
            # rate = 0.8 表示减速20%，对应 tempo -20
            tempo_change = (stretch_rate - 1) * 100
            
            # 构建命令
            cmd = [
                "soundstretch",
                input_path,
                output_path,
                "-tempo", f"{tempo_change:+.1f}",
                "-pitch", "0"  # 保持音调不变
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )
            
            if result.returncode != 0:
                print(f"[ERROR] soundstretch 失败: {result.stderr}")
                return False
            
            return True
            
        except subprocess.TimeoutExpired:
            print("[ERROR] soundstretch 超时")
            return False
        except Exception as e:
            print(f"[ERROR] soundstretch 异常: {e}")
            return False
    
    def _shift_with_librosa(
        self,
        input_path: str,
        output_path: str,
        stretch_rate: float
    ) -> bool:
        """
        使用 librosa 进行变速处理
        
        使用相位声码器实现时间拉伸，保持音调不变。
        音质略逊于 soundstretch，但无需额外安装。
        
        Args:
            input_path: 输入音频路径
            output_path: 输出音频路径
            stretch_rate: 变速比例
            
        Returns:
            是否成功
        """
        try:
            # 加载音频
            y, sr = librosa.load(input_path, sr=None, mono=False)
            
            # 处理单声道和立体声
            # rate > 1 表示加速（时长变短），rate < 1 表示减速（时长变长）
            # stretch_rate = 目标BPM / 原始BPM
            # 如果目标BPM > 原始BPM，需要加速，所以 rate = stretch_rate
            if y.ndim == 1:
                # 单声道
                y_stretched = librosa.effects.time_stretch(y, rate=stretch_rate)
            else:
                # 立体声 - 分别处理每个声道
                y_stretched = np.array([
                    librosa.effects.time_stretch(y[0], rate=stretch_rate),
                    librosa.effects.time_stretch(y[1], rate=stretch_rate)
                ])
            
            # 保存结果
            sf.write(output_path, y_stretched.T if y_stretched.ndim > 1 else y_stretched, sr)
            
            return True
            
        except Exception as e:
            print(f"[ERROR] librosa 处理失败: {e}")
            return False
    
    def process_file(
        self,
        input_path: str,
        original_bpm: float,
        output_dir: Optional[str] = None,
        output_name: Optional[str] = None,
        strict_mode: bool = True
    ) -> TempoShiftResult:
        """
        处理单个音频文件
        
        Args:
            input_path: 输入音频路径
            original_bpm: 原始BPM
            output_dir: 输出目录，为None时使用输入文件所在目录
            output_name: 输出文件名（不含路径），为None时自动生成
            strict_mode: 是否严格模式
            
        Returns:
            处理结果
        """
        # 计算变速比例（根据是否严格模式）
        try:
            stretch_rate = self._calculate_stretch_rate(original_bpm, strict_mode)
            actual_target_bpm = int(round(original_bpm * stretch_rate))
        except ValueError as e:
            return TempoShiftResult(
                input_path=input_path,
                output_path="",
                original_bpm=original_bpm,
                target_bpm=self.target_bpm,
                stretch_rate=0,
                method="",
                success=False,
                error_msg=str(e)
            )
        
        # 确定输出路径
        if output_name is None:
            input_name = Path(input_path).stem
            output_name = f"{input_name}_{actual_target_bpm}bpm.wav"
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, output_name)
        else:
            output_path = os.path.join(os.path.dirname(input_path), output_name)
        
        print(f"[PROCESS] {Path(input_path).name}")
        if strict_mode:
            print(f"          原始BPM: {original_bpm:.1f} → 目标BPM: {self.target_bpm} (严格模式)")
        else:
            print(f"          原始BPM: {original_bpm:.1f} → 实际BPM: {actual_target_bpm} (非严格模式，目标BPM={self.target_bpm})")
        print(f"          变速比例: {stretch_rate:.3f}x ({(stretch_rate-1)*100:+.1f}%)")
        print(f"          方法: {self.selected_method}")
        
        # 执行变速处理
        if self.selected_method == "soundstretch":
            success = self._shift_with_soundstretch(input_path, output_path, stretch_rate)
        else:
            success = self._shift_with_librosa(input_path, output_path, stretch_rate)
        
        if success:
            print(f"          [OK] 已保存: {output_name}")
        else:
            print(f"          [FAIL] 处理失败")
        
        return TempoShiftResult(
            input_path=input_path,
            output_path=output_path if success else "",
            original_bpm=original_bpm,
            target_bpm=actual_target_bpm,
            stretch_rate=stretch_rate,
            method=self.selected_method,
            success=success
        )
    
    def process_from_json(
        self,
        json_path: str,
        output_dir: Optional[str] = None,
        file_filter: Optional[List[str]] = None,
        strict_mode: bool = True
    ) -> List[TempoShiftResult]:
        """
        从 JSON 文件批量处理音频
        
        Args:
            json_path: song_bpm_list.json 文件路径
            output_dir: 输出目录
            file_filter: 只处理指定的文件名列表，为None时处理全部
            strict_mode: 是否严格模式
            
        Returns:
            处理结果列表
        """
        # 读取 JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            songs = json.load(f)
        
        print(f"[INFO] 从 {json_path} 加载了 {len(songs)} 首歌曲")
        print(f"[INFO] 目标BPM: {self.target_bpm}, 方法: {self.selected_method}, 严格模式: {strict_mode}")
        print("=" * 60)
        
        results = []
        for song in songs:
            file_name = song["file_name"]
            
            # 过滤
            if file_filter and file_name not in file_filter:
                continue
            
            result = self.process_file(
                input_path=song["file_path"],
                original_bpm=song["original_bpm"],
                output_dir=output_dir,
                strict_mode=strict_mode
            )
            results.append(result)
            print()
        
        return results
    
    def print_summary(self, results: List[TempoShiftResult]) -> None:
        """打印处理摘要"""
        total = len(results)
        success = sum(1 for r in results if r.success)
        failed = total - success
        
        print("=" * 60)
        print("[SUMMARY] 变速处理结果汇总")
        print("=" * 60)
        print(f"总文件数: {total}")
        print(f"成功: {success}")
        print(f"失败: {failed}")
        
        if success > 0:
            print("\n处理成功的文件:")
            for r in results:
                if r.success:
                    print(f"  ✓ {Path(r.input_path).name} ({r.original_bpm:.1f} → {r.target_bpm} BPM)")
        
        if failed > 0:
            print("\n处理失败的文件:")
            for r in results:
                if not r.success:
                    print(f"  ✗ {Path(r.input_path).name}: {r.error_msg or '未知错误'}")
        
        print("=" * 60)


def main():
    """
    模块3主函数 - 测试入口
    
    使用示例:
        # 从JSON批量处理
        python tempo_shifter.py --json ../song_bpm_list.json --target-bpm 180
        
        # 处理单个文件
        python tempo_shifter.py --input song.wav --original-bpm 120 --target-bpm 180
        
        # 指定输出目录
        python tempo_shifter.py --json ../song_bpm_list.json --target-bpm 180 -o ../audio_output
        
        # 强制使用librosa
        python tempo_shifter.py --json ../song_bpm_list.json --target-bpm 180 --method librosa
    """
    parser = argparse.ArgumentParser(
        description="变速不变调处理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 从JSON批量处理所有歌曲
  python tempo_shifter.py --json ../song_bpm_list.json --target-bpm 180
  
  # 处理单个文件
  python tempo_shifter.py --input song.wav --original-bpm 120 --target-bpm 180
  
  # 指定输出目录
  python tempo_shifter.py --json ../song_bpm_list.json --target-bpm 180 -o ../audio_output
  
  # 强制使用librosa方法
  python tempo_shifter.py --json ../song_bpm_list.json --target-bpm 180 --method librosa
        """
    )
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    # 输入方式（互斥）
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--json",
        type=str,
        help="song_bpm_list.json 文件路径"
    )
    input_group.add_argument(
        "--input",
        type=str,
        help="单个音频文件路径"
    )
    
    # 单个文件处理需要的参数
    parser.add_argument(
        "--original-bpm",
        type=float,
        help="原始BPM（仅单文件模式需要）"
    )
    
    # 通用参数
    parser.add_argument(
        "--target-bpm",
        type=int,
        required=True,
        help="目标BPM（必须指定）"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["auto", "soundstretch", "librosa"],
        default="auto",
        help="变速方法 (默认: auto，优先使用soundstretch)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=os.path.join(project_dir, "audio_output", "shifted_songs"),
        help=f"输出目录 (默认: {os.path.join(project_dir, 'audio_output', 'shifted_songs')})"
    )
    
    args = parser.parse_args()
    
    # 验证参数
    if args.input and not args.original_bpm:
        parser.error("单文件模式需要指定 --original-bpm")
    
    print("[RunBeat] 变速不变调处理模块")
    print("=" * 60)
    
    # 创建处理器
    shifter = TempoShifter(
        target_bpm=args.target_bpm,
        method=args.method
    )
    
    # 执行处理
    if args.json:
        # 批量处理
        results = shifter.process_from_json(
            json_path=os.path.abspath(args.json),
            output_dir=os.path.abspath(args.output)
        )
    else:
        # 单文件处理
        result = shifter.process_file(
            input_path=os.path.abspath(args.input),
            original_bpm=args.original_bpm,
            output_dir=os.path.abspath(args.output)
        )
        results = [result]
    
    # 打印摘要
    shifter.print_summary(results)
    
    # 返回状态码
    success_count = sum(1 for r in results if r.success)
    return 0 if success_count == len(results) else 1


if __name__ == "__main__":
    exit(main())
