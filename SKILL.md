---
name: runbeat
description: "一键跑步音乐生成流水线。当用户说'做跑步音乐'、'跑步歌单'、'runbeat'、'做一个跑步音乐'、'帮我处理跑步音乐'或类似意图时触发。也适用于用户提出想把手里的音乐处理成适合跑步听的时候。可以指定目标步频（BPM），默认 180。需要用户提供存放音乐文件的文件夹路径。"
version: "1.0.0"
---

# RunBeat — 跑步音乐生成流水线

将用户指定的音频文件夹中的音乐文件，通过 BPM 检测 → 变速不变调 → 节拍器生成 → 首拍对齐 → 混音的完整流水线，自动生成适合跑步用的连续串烧音乐。

**完全自包含**：所有核心模块都内嵌在 skill 中，不依赖外部项目路径，搬到哪都能用。

## 核心功能

- **一键全自动**：扫描输入目录 → 流水线处理 → 输出最终混音
- **智能步频匹配**：自动检测每首歌的原始 BPM，变速到目标步频
- **首拍对齐**：每首歌的首拍对齐到节拍器重拍，节奏丝滑
- **节拍器叠加**：强拍/弱拍音色区分，帮助跟节奏
- **Audacity 分轨导出**：额外生成每首歌独立轨 + 节拍器轨（需 FFmpeg）

## 使用方式

```
/runbeat --input "D:\music"
/runbeat --input "D:\music" --bpm 165
/runbeat --input "D:\music" --bpm 180 --strict --output "D:\run-ready"
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input` / `-i` | 音频文件或文件夹路径 (必填) | — |
| `--bpm` | 目标步频 | 180 |
| `--beats-per-measure` | 每小节拍数 | 4 |
| `--output` | 输出目录 | `输入目录/runbeat-output/` |
| `--strict` | 精确变速到目标BPM（音质可能下降） | 非严格（音质优先） |
| `--metronome-db` | 节拍器音量 (dB) | 3 |

## 前置条件

1. **Python 3.8+** 及依赖包：
   ```
   pip install librosa pydub soundfile numpy
   ```
2. **FFmpeg**（可选，用于 Audacity 分轨导出）：
   ```
   winget install ffmpeg
   ```

## 工作流程

### Step 1: 确认用户意图

收到用户请求后，确认以下信息：

1. **输入路径** — 用户提供存放音频文件的文件夹路径
2. **目标 BPM** — 若用户没指定，默认 180
3. **输出目录** — 若用户没指定 `--output`，默认在输入目录下创建 `runbeat-output/`
4. **严格模式** — 默认非严格，用户可通过 `--strict` 开启
5. **节拍器音量** — 默认 0dB

### Step 2: 调用流水线

skill 目录下自带 `runbeat_cli.py` 和 `modules/` 包，直接运行：

```bash
SKILL_DIR="C:/Users/14480/.proma/agent-workspaces/workspace-1779546127083/skills/runbeat"
python "${SKILL_DIR}/runbeat_cli.py" --input "用户提供的路径" --bpm 180
```

如果用户指定了额外参数，拼接到命令中。

### Step 3: 汇报执行结果

流水线完成后，向用户汇报：
- **处理总结**：输入文件数、成功/失败数
- **输出位置**：最终混音、BPM 数据、Audacity 导出
- **失败文件**：文件名和原因

### Step 4: 后续建议（可选）

- 直接播放 `final_mix.wav` 开始跑步
- 对过渡不满意，打开 `audacity_export/` 在 Audacity 中精调

## 错误处理

- **依赖缺失**：提示用户先 `pip install` 对应包
- **无音频文件**：汇报目录中没有找到支持的音频格式
- **部分失败**：跳过失败文件继续处理，最终给出失败清单
- **FFmpeg 缺失**：核心混音不受影响，仅跳过 Audacity 分轨导出
