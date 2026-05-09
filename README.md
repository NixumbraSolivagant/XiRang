# XiRang — 粒子生命宇宙

> 你只写下了几行冰冷的数学公式，但屏幕上却涌现出了生生不息的热烈生命。

一个 GPU 加速的粒子生命 (Particle Life) 模拟器。基于 Jeffrey Ventrella 的"Clusters"思想，从纯数学法则中自发涌现出类似生物的结构——游动、聚集、自我复制。

---

## 核心物理

| 组件 | 描述 |
|---|---|
| **虚空创生** | 量子涨落机制，粒子从"无"中概率性诞生 |
| **4 种基子 Ψ₁–Ψ₄** | 红/蓝/绿/黄，仅凭"互动参数"区分，无预设生物概念 |
| **不对称交互矩阵 M** | 替代引力/电磁力，4×4 矩阵决定粒子间吸引/排斥强度 |
| **耗散结构** | 摩擦力迫使粒子形成稳定几何体来抵抗混乱 |
| **空间哈希网格** | O(N) 邻居查找，1M 粒子实时模拟成为可能 |

---

## 快速开始

```bash
pip install -r requirements.txt
python main.py
```

---

## 运行模式

### GUI 模式（需要图形界面）

```bash
python main.py
```

**键盘操作：**

| 键 | 功能 |
|---|---|
| `R` | 重置引力矩阵（随机生成新的一套物理法则） |
| `Space` | 暂停 / 继续 |
| `F` | 降低虚空涨落温度 |
| `S` | 保存截图 |
| `Q` / `Esc` | 退出 |

### 无头服务器模式（无图形界面）

```bash
python main.py --headless --particles 1000000 --steps 500 --record-every 1
```

生成帧序列到 `frames/`，用 ffmpeg 合成视频：

```bash
ffmpeg -framerate 60 -i frames/frame_%06d.png -c:v libx264 -pix_fmt yuv420p out.mp4
```

---

## 命令行参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--particles` / `-n` | `1000000` | 粒子数量 |
| `-W` | `1280` | 模拟宽度 (px) |
| `-H` | `720` | 模拟高度 (px) |
| `-r` | `50.0` | 最大交互半径 |
| `--r-min` | `2.0` | 核心排斥半径 |
| `-f` | `0.05` | 摩擦系数 |
| `-v` | `0.8` | 虚空涨落温度 (0–1) |
| `--fps` | `60` | 目标帧率 |
| `--headless` | — | 无 GUI 模式 |
| `--steps` | `500` | 无头模式步数 |
| `--output-dir` | `frames` | 帧输出目录 |
| `--record-every` | `1` | 每 N 帧保存一帧 |
| `--save-matrix` | — | 保存引力矩阵到 .npy |
| `--load-matrix` | — | 从 .npy 加载引力矩阵 |

---

## 项目结构

```
XiRang/
├── main.py                     入口，GUI / headless 分支
├── requirements.txt
├── particle_life/
│   ├── __init__.py
│   ├── config.py                物理常量与参数
│   ├── particle_types.py       GPU 张量状态管理
│   ├── spatial_hash.py         O(N) 空间哈希邻居查找
│   ├── physics.py              向量化 GPU 力学计算
│   ├── engine.py               模拟引擎，step 驱动
│   ├── renderer.py             Pygame GUI 渲染器
│   └── headless_recorder.py    纯 PIL 帧序列写入（无 GUI 依赖）
```

---

## 在其他机器部署

```bash
pip install -r requirements.txt
python main.py --headless --particles 1000000 --steps 500
```

> 注意：需要有 NVIDIA GPU 和对应 CUDA 驱动（PyTorch 会自动检测）。无头模式下 pygame 不是必须的，但 Pillow 必须安装。
