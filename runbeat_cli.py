#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
RunBeat CLI - One-click running music generation pipeline

Pipeline:
  scan audio -> number & copy -> BPM detect(mixxx) -> tempo shift ->
  metronome -> beat align -> mix -> Audacity export

Usage:
  python runbeat_cli.py --input "D:\music"
  python runbeat_cli.py --input "D:\music" --bpm 165
  python runbeat_cli.py --input "D:\music" --bpm 180 --strict --output "D:\run-ready"
"""

import os
import sys
import json
import argparse
import re
import shutil
import subprocess
import textwrap
import traceback
from pathlib import Path
from datetime import datetime

# 设置 stdout 编码为 UTF-8，避免 Windows GBK 下 Unicode 符号异常
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import soundfile as sf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from modules import (
    BPMAnalysisResult,
    MetronomeGenerator, MetronomeConfig,
    TempoShifter,
    AudioMixer, MixConfig,
    FirstBeatDetector, BeatAlignmentManager,
)
from modules.batch_bpm_detector import MixxxBPMDetector

SUPPORTED_FORMATS = ('.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac')
OK_SYM = "[v]"
WARN_SYM = "[!]"
FAIL_SYM = "[x]"


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]


def scan_audio_files(input_dir, exclude_dir=None):
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    exclude_abs = os.path.abspath(exclude_dir) if exclude_dir else None
    audio_files = []
    for ext in SUPPORTED_FORMATS:
        for p in input_path.rglob(f"*{ext}"):
            if exclude_abs and os.path.abspath(str(p)).startswith(exclude_abs):
                continue
            audio_files.append(str(p))
        for p in input_path.rglob(f"*{ext.upper()}"):
            if exclude_abs and os.path.abspath(str(p)).startswith(exclude_abs):
                continue
            audio_files.append(str(p))
    return sorted(set(audio_files), key=natural_sort_key)


def get_audio_duration_ms(file_path):
    try:
        info = sf.info(file_path)
        return info.duration * 1000
    except Exception as e:
        print(f"  {WARN_SYM} 无法读取 {os.path.basename(file_path)} 时长: {e}")
        return 0


def run_pipeline(args):
    input_dir = os.path.abspath(args.input)
    target_bpm = args.bpm
    beats_per_measure = args.beats_per_measure
    strict_mode = args.strict
    output_dir = os.path.abspath(args.output) if args.output else None
    working_dir = None  # 临时工作目录，结束时清理

    # ========== 阶段0: 准备 ==========
    print("=" * 60)
    print("  RunBeat -- 跑步音乐生成流水线")
    print("=" * 60)
    print(f"  输入目录: {input_dir}")
    print(f"  目标BPM:  {target_bpm}")
    print(f"  拍号:     {beats_per_measure}/4")
    print(f"  变速模式: {'严格' if strict_mode else '非严格（音质优先）'}")
    if output_dir:
        print(f"  输出目录: {output_dir}")
    print(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    print("\n[阶段0] 扫描音频文件...")
    if not output_dir:
        output_dir = os.path.join(input_dir, "runbeat-output")

    # 扫描（排除 output_dir）
    all_audio_files = scan_audio_files(input_dir, exclude_dir=output_dir)
    if not all_audio_files:
        print(f"[ERROR] 在 {input_dir} 中未找到支持的音频文件 ({', '.join(SUPPORTED_FORMATS)})")
        return 1

    print(f"  找到 {len(all_audio_files)} 个音频文件")
    for f in all_audio_files:
        print(f"    {os.path.relpath(f, input_dir)}")

    os.makedirs(output_dir, exist_ok=True)

    # 工作目录：放到系统临时目录下（纯 ASCII 路径，mixxx 的 FFmpeg 不卡中文路径）
    import tempfile
    working_dir = tempfile.mkdtemp(prefix="runbeat_")
    shifted_dir = os.path.join(output_dir, "shifted_songs")
    metronome_dir = os.path.join(output_dir, "metronome")
    audacity_dir = os.path.join(output_dir, "audacity_export")
    for d in (working_dir, shifted_dir, metronome_dir, audacity_dir):
        os.makedirs(d, exist_ok=True)

    bpm_json_path = os.path.join(output_dir, "bpm_results.json")
    alignment_json_path = os.path.join(output_dir, "beat_alignments.json")
    mapping_path = os.path.join(output_dir, "file_mapping.json")

    print(f"\n  输出目录: {output_dir}")
    print(f"  BPM数据:  bpm_results.json")
    print(f"  变速后:   shifted_songs/")
    print(f"  节拍器:   metronome/")
    print(f"  Audacity: audacity_export/")

    try:
        return _run_pipeline_inner(
            input_dir, target_bpm, beats_per_measure, strict_mode, output_dir,
            working_dir, all_audio_files,
            bpm_json_path, alignment_json_path, mapping_path,
            shifted_dir, metronome_dir, audacity_dir, args,
        )
    finally:
        # 清理临时工作目录
        if working_dir and os.path.exists(working_dir):
            shutil.rmtree(working_dir, ignore_errors=True)


def _run_pipeline_inner(
    input_dir, target_bpm, beats_per_measure, strict_mode, output_dir,
    working_dir, all_audio_files,
    bpm_json_path, alignment_json_path, mapping_path,
    shifted_dir, metronome_dir, audacity_dir, args,
):
    """流水线核心逻辑（被 run_pipeline 的 try/finally 包裹，负责清理临时目录）"""

    # ========== 编号拷贝 ==========
    print(f"\n{'=' * 60}")
    print("[步骤1] 文件编号与拷贝")
    print("=" * 60)

    file_mapping = {}
    numbered_files = []  # (number, numbered_path, original_path)
    for idx, orig_path in enumerate(all_audio_files, 1):
        ext = os.path.splitext(orig_path)[1] or '.mp3'
        num_name = f"{idx}{ext}"
        num_path = os.path.join(working_dir, num_name)
        shutil.copy2(orig_path, num_path)
        file_mapping[num_name] = {
            "original_name": os.path.basename(orig_path),
            "original_path": orig_path,
        }
        numbered_files.append((idx, num_path, orig_path))
        print(f"  {idx:>2}. {os.path.basename(orig_path)} -> {num_name}")

    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(file_mapping, f, ensure_ascii=False, indent=2)
    print(f"  [INFO] 映射已保存: file_mapping.json")

    # ========== 阶段1: BPM检测 (纯 mixxx) ==========
    print(f"\n{'=' * 60}")
    print("[阶段1/5] BPM检测 (mixxx-analyzer)")
    print("=" * 60)

    detector = MixxxBPMDetector()
    bpm_results = []
    bpm_success = 0
    bpm_failed = 0

    existing_results = {}
    if os.path.exists(bpm_json_path):
        try:
            with open(bpm_json_path, 'r', encoding='utf-8') as f:
                for item in json.load(f):
                    existing_results[item['file_path']] = item
        except Exception:
            pass

    for idx, num_path, orig_path in numbered_files:
        num_name = os.path.basename(num_path)
        file_label = file_mapping[num_name]["original_name"]

        if num_path in existing_results:
            r = existing_results[num_path]
            result = BPMAnalysisResult(**r)
            bpm_results.append(result)
            print(f"  [{idx}/{len(numbered_files)}] [SKIP] {file_label}: BPM={result.original_bpm:.1f}")
            continue

        print(f"  [{idx}/{len(numbered_files)}] [ANALYZE] {file_label} ({num_name})...", end=" ")
        result = detector.analyze(num_path)
        if result:
            bpm_results.append(result)
            bpm_success += 1
            ci = "[OK]" if result.confidence >= 0.6 else "[LOW]"
            print(f"BPM={result.original_bpm:.1f} {ci}")
        else:
            bpm_failed += 1
            print("[FAIL]")

    bpm_data = []
    for r in bpm_results:
        from dataclasses import asdict
        bpm_data.append(asdict(r))
    with open(bpm_json_path, 'w', encoding='utf-8') as f:
        json.dump(bpm_data, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 结果已保存至: {bpm_json_path}")
    if not bpm_results:
        print("[ERROR] BPM检测失败，无有效结果")
        return 1
    print("-" * 60)
    print(f"[INFO] 分析完成: 成功 {bpm_success}, 失败 {bpm_failed}")

    # ========== 阶段2: 变速 ==========
    print(f"\n{'=' * 60}")
    print("[阶段2/5] 变速不变调处理")
    print("=" * 60)

    shifter = TempoShifter(target_bpm=target_bpm)
    shift_results = shifter.process_from_json(
        json_path=bpm_json_path,
        output_dir=shifted_dir,
        strict_mode=strict_mode,
    )
    successful_shifts = [r for r in shift_results if r.success]
    if not successful_shifts:
        print("[ERROR] 所有文件变速失败，无法继续")
        return 1

    # ========== 阶段3: 节拍器 ==========
    print(f"\n{'=' * 60}")
    print("[阶段3/5] 节拍器生成")
    print("=" * 60)

    estimated_total_duration = 0
    bpm_by_path = {r.file_path: r for r in bpm_results}
    for r in successful_shifts:
        bpm_r = bpm_by_path.get(r.input_path)
        if bpm_r and r.stretch_rate > 0:
            estimated_total_duration += bpm_r.duration / r.stretch_rate

    gap_per_song = 4 * 60.0 / target_bpm
    estimated_total_duration += len(successful_shifts) * gap_per_song + 30

    metronome_path = os.path.join(metronome_dir, f"metronome_{target_bpm}.wav")
    metronome_gen = MetronomeGenerator(MetronomeConfig(
        bpm=target_bpm, duration=max(estimated_total_duration, 60),
        beats_per_measure=beats_per_measure,
    ))
    metronome_gen.generate(metronome_path)

    # ========== 阶段4: 首拍检测 ==========
    print(f"\n{'=' * 60}")
    print("[阶段4/5] 首拍检测")
    print("=" * 60)

    alignment_manager = BeatAlignmentManager(json_path=alignment_json_path)
    shifted_files_sorted = sorted(
        [r.output_path for r in successful_shifts if os.path.exists(r.output_path)],
        key=natural_sort_key,
    )

    for shifted_path in shifted_files_sorted:
        try:
            result = FirstBeatDetector().detect(shifted_path)
            alignment_manager.set_first_beat(shifted_path, result.first_beat_time)
            print(f"  {OK_SYM} {os.path.basename(shifted_path)}: 首拍={result.first_beat_time:.3f}s")
        except Exception as e:
            print(f"  {WARN_SYM} {os.path.basename(shifted_path)}: 首拍检测失败 ({e}), 使用默认值0")
            alignment_manager.set_first_beat(shifted_path, 0.0)

    # ========== 阶段5: 混音 ==========
    print(f"\n{'=' * 60}")
    print("[阶段5/5] 首拍对齐混音")
    print("=" * 60)

    final_mix_path = os.path.join(output_dir, "final_mix.wav")
    mixer = AudioMixer(MixConfig(
        metronome_bpm=target_bpm,
        metronome_path=metronome_path,
        metronome_volume=args.metronome_db,
    ))
    mixer.alignment_manager = alignment_manager

    if not shifted_files_sorted:
        print("[ERROR] 没有变速成功后的文件可用于混音")
        return 1

    success = mixer.mix_with_beat_alignment(
        input_folder=shifted_dir,
        output_path=final_mix_path,
        file_order=[os.path.basename(f) for f in shifted_files_sorted],
    )
    if not success:
        print("[ERROR] 混音失败")
        return 1

    # ========== 阶段6: Audacity导出 ==========
    print(f"\n{'=' * 60}")
    print("[阶段6/6] 导出 Audacity 分轨文件")
    print("=" * 60)

    first_beat = alignment_manager.get_first_beat(shifted_files_sorted[0]) or 0.0
    beat_duration_ms = (60.0 / target_bpm) * 1000
    measure_duration_ms = beat_duration_ms * 4

    current_measure = 1
    prev_song_end_ms = 0
    ffmpeg_available = True

    for idx, shifted_path in enumerate(shifted_files_sorted):
        if not ffmpeg_available:
            break

        shifted_name = os.path.basename(shifted_path)
        # 从编号文件名反查原始名: 如 "1_90bpm.wav" -> 找映射中 "1.mp3"
        stem = shifted_name.split("_")[0]
        orig_name = file_mapping.get(f"{stem}.mp3", {}).get("original_name", shifted_name)

        song_first_beat_ms = (alignment_manager.get_first_beat(shifted_path) or 0.0) * 1000
        song_duration_ms = get_audio_duration_ms(shifted_path)

        if idx == 0:
            song_start_ms = 0
        else:
            target_beat = first_beat * 1000 + (current_measure * measure_duration_ms)
            song_start_ms = target_beat - song_first_beat_ms

            guard = 0
            while song_start_ms < prev_song_end_ms and guard < 200:
                current_measure += 1
                guard += 1
                target_beat = first_beat * 1000 + (current_measure * measure_duration_ms)
                song_start_ms = target_beat - song_first_beat_ms

        song_end_ms = song_start_ms + song_duration_ms
        prev_song_end_ms = song_end_ms

        elapsed = song_end_ms - first_beat * 1000
        current_beat = int(max(1, elapsed / beat_duration_ms))
        current_measure = ((current_beat // 4) + 1) / 4

        delay_ms = int(song_start_ms)
        # 用编号文件名避免中文路径问题
        audacity_file = os.path.join(audacity_dir, f"{idx+1:02d}_{os.path.basename(shifted_path)}")
        cmd = [
            'ffmpeg', '-y', '-i', shifted_path,
            '-af', f'adelay={delay_ms}|{delay_ms}:all=1',
            '-acodec', 'pcm_s16le', audacity_file,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=120)
            print(f"  {OK_SYM} {orig_name}: 前导静音 {song_start_ms/1000:.3f}s")
        except subprocess.CalledProcessError as e:
            print(f"  {WARN_SYM} {orig_name}: ffmpeg 处理失败 ({e})")
        except FileNotFoundError:
            print(f"  {WARN_SYM} ffmpeg 未安装，跳过 Audacity 分轨导出")
            ffmpeg_available = False

    if ffmpeg_available and shifted_files_sorted:
        metronome_delay = int(first_beat * 1000)
        mp = os.path.join(audacity_dir, "metronome_processed.wav")
        try:
            subprocess.run([
                'ffmpeg', '-y', '-i', metronome_path,
                '-af', f'adelay={metronome_delay}|{metronome_delay}:all=1',
                '-acodec', 'pcm_s16le', mp,
            ], check=True, capture_output=True, timeout=120)
            print(f"  {OK_SYM} metronome: 前导静音 {metronome_delay/1000:.3f}s")
        except Exception:
            pass

    with open(os.path.join(audacity_dir, "README.txt"), 'w', encoding='utf-8') as f:
        f.write(textwrap.dedent(f"""\
            RunBeat Audacity 分轨文件
            ============================
            生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

            使用方法:
            1. 打开 Audacity
            2. 文件 -> 导入 -> 音频...
            3. 选中 .wav 文件导入，已按正确时间点对齐
            4. 检查过渡和音量，微调后导出

            参数:
            - 目标BPM: {target_bpm}
            - 输入目录: {input_dir}
            - 变速模式: {"严格" if strict_mode else "非严格"}
        """).strip())
    print(f"  {OK_SYM} Audacity 使用说明已写入 README.txt")

    # ========== 完成 ==========
    total_ok = len(successful_shifts)
    failed_count = len(shift_results) - total_ok

    print(f"\n{'=' * 60}")
    print("  RunBeat 流水线完成!")
    print("=" * 60)
    print(f"  输入目录:    {input_dir}")
    print(f"  输出目录:    {output_dir}")
    print(f"  目标BPM:     {target_bpm}")
    print(f"  文件总数:    {len(all_audio_files)}")
    print(f"  成功:        {total_ok}")
    print(f"  失败:        {failed_count}")
    print(f"  最终混音:    final_mix.wav")
    print(f"  BPM数据:     bpm_results.json")
    print(f"  Audacity导出: audacity_export/")
    print(f"  完成时间:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 非音频文件
    all_dir = set()
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            all_dir.add(os.path.join(root, f))
    skipped_audio = all_dir - set(all_audio_files)
    if skipped_audio:
        print(f"\n跳过的非音频文件 ({len(skipped_audio)} 个):")
        for f in sorted(skipped_audio):
            print(f"  - {os.path.relpath(f, input_dir)}")

    failed_items = [r for r in shift_results if not r.success]
    if failed_items:
        print(f"\n处理失败的文件 ({len(failed_items)} 个):")
        for r in failed_items:
            stem = os.path.basename(r.input_path).split(".")[0]
            orig = file_mapping.get(f"{stem}.mp3", {}).get("original_name", os.path.basename(r.input_path))
            print(f"  {FAIL_SYM} {orig}: {r.error_msg or '未知错误'}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="RunBeat -- 一键跑步音乐生成流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            示例:
              # 默认 180 BPM
              python runbeat_cli.py --input "D:\\music"

              # 指定 BPM
              python runbeat_cli.py --input "D:\\music" --bpm 165

              # 严格模式 + 自定义输出目录
              python runbeat_cli.py --input "D:\\music" --bpm 180 --strict --output "D:\\run-ready"
        """),
    )
    parser.add_argument("-i", "--input", required=True, help="输入音频文件文件夹路径")
    parser.add_argument("--bpm", type=int, default=180, help="目标步频 BPM (默认: 180)")
    parser.add_argument("--beats-per-measure", type=int, default=4,
                        help="每小节拍数 (默认: 4, 三拍子设 6)")
    parser.add_argument("--output", default=None, help="输出目录 (默认: 输入目录/runbeat-output/)")
    parser.add_argument("--strict", action="store_true", help="严格模式: 精确变速到目标 BPM")
    parser.add_argument("--metronome-db", type=float, default=3.0, help="节拍器音量 dB (默认: 3)")

    args = parser.parse_args()
    try:
        sys.exit(run_pipeline(args))
    except Exception as e:
        print(f"\n[FATAL] 流水线异常: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
