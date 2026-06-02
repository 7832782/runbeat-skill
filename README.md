# RunBeat Skill 🏃‍♂️🎵

**/runbeat** — Proma Agent skill：一键生成跑步音乐串烧。

把任意音乐文件夹丢进去，自动走完五步流水线，输出匹配目标步频的混音。

## 快速开始

```bash
# 直接用（需要 Python 3.8+）
pip install librosa pydub soundfile numpy
/runbeat --input "D:\Music\run"

# 指定步频
/runbeat --input "D:\Music" --bpm 165

# 严格模式（精确变速，音质可能下降）
/runbeat --input "D:\Music" --bpm 180 --strict
```

## 流水线

```
输入音频 → BPM 检测 → 变速不变调 → 节拍器合成 → 首拍对齐 → 智能混音 → 输出
```

| 步骤 | 说明 |
|------|------|
| **BPM 检测** | mixxx-analyzer（QM 梳状滤波）主引擎，Librosa 备选兜底 |
| **变速不变调** | Phase Vocoder（STFT + 相位展开 + ISTFT），严格/非严格模式 |
| **节拍器合成** | 8 种音色，Audacity Rhythm Track 1:1 复刻 |
| **首拍对齐** | 低频分离 → 加权 onset → 节拍网格追踪 → 四维评分 |
| **智能混音** | 首拍对齐到重拍 + 防重叠 + 响度归一化 |

## 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` / `-i` | 必填 | 音频文件或文件夹路径 |
| `--bpm` | 180 | 目标步频 |
| `--output` | 输入目录/output | 输出目录 |
| `--strict` | 非严格 | 精确变速（音质可能下降） |
| `--metronome-db` | 3 | 节拍器音量 |
| `--beats-per-measure` | 4 | 每小节拍数 |

## 输出

- `final_mix.wav` — 最终混音
- `final_mix_track_*.wav` — 每首歌独立分轨
- `final_mix_metronome.wav` — 节拍器独立轨
- `song_bpm_list.json` — BPM 检测结果
- `beat_alignments.json` — 首拍对齐数据
- `audacity_export/` — Audacity 多轨项目文件

## 前置依赖

```bash
pip install librosa pydub soundfile numpy
```

FFmpeg（可选，用于 Audacity 导出）：
```bash
winget install ffmpeg
```

## 技术栈

- **BPM 检测**：mixxx-analyzer (QM Tempo Tracker) / Librosa
- **变速**：Phase Vocoder / pyrubberband
- **节拍器**：SciPy DSP + IIR 滤波 + JC 混响
- **混音**：pydub + numpy
- **UI**：PyQt6（完整版）/ CLI（Skill 版）

## 完整版

含 PyQt6 GUI 的完整版本见 [github.com/7832782/runbeat](https://github.com/7832782/runbeat)。

## License

MIT
