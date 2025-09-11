# YOLO Labeling Tool

[![English](https://img.shields.io/badge/English-007BFF?style=for-the-badge&logo=google-chrome)](README.md)
[![Русский](https://img.shields.io/badge/%D0%A0%D1%83%D1%81%D1%81%D0%BA%D0%B8%D0%B9-2E7D32?style=for-the-badge&logo=google-chrome)](README-ru.md)
[![Français](https://img.shields.io/badge/Fran%C3%A7ais-0055A4?style=for-the-badge&logo=google-chrome)](README-fr.md)
[![Deutsch](https://img.shields.io/badge/Deutsch-000000?style=for-the-badge&logo=google-chrome)](README-de.md)
[![日本語](https://img.shields.io/badge/%E6%97%A5%E6%9C%AC%E8%AA%9E-BC002D?style=for-the-badge&logo=google-chrome)](README-ja.md)
[![中文](https://img.shields.io/badge/%E4%B8%AD%E6%96%87-007A33?style=for-the-badge&logo=google-chrome)](README-zh.md)

功能强大的 YOLO 格式图像标注工具，支持使用神经网络进行自动标注。

## 概述

本应用提供完整的标注工作流程：
- 使用预训练 YOLO 模型（detect/OBB/segment）自动生成边界框和多边形
- 手动编辑标注（添加、删除、移动、调整大小）
- 实时与真实数据对比评估
- 项目管理并导出为训练数据集

**核心功能：**
- ✅ 两种标注模式：矩形框模式 和 多边形模式
- ✅ 基于模型的自动标注（YOLO detect/OBB/segment）
- ✅ 基于 IoU 的评估模式，可视化 FP/FN
- ✅ 类别颜色编码与自定义类别管理器
- ✅ 完整键盘导航与快捷键
- ✅ 可配置界面语言（支持多语言）

> 💡 **提示**：按 `H` 切换标注可见性，`N` 保存至训练集，`E` 启用评估模式。

---

## 快速开始

### 先决条件
- Python 3.8+
- Windows / Linux / macOS

### 安装
```bash
git clone https://github.com/V171/yolo-labeling-tool.git
cd yolo-labeling-tool
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install ultralytics opencv-python pyqt5 numpy
```

### 配置
编辑 `config.json`：
```json
{
  "model_path": "yolov8n.pt",
  "images_dir": "images",
  "train_dir": "TrainDataset",
  "language": "zh",
  "annotation_mode": "box",
  "iou_threshold": 0.5
}
```
- `model_path`：您的 YOLO 模型路径（.pt 文件）。
- `images_dir`：包含待标注原始图像的目录。
- `train_dir`：保存标注后图像作为真实数据的目录（可设为任意路径）。
- `language`：界面语言（`en`, `ru`, `fr`, `de`, `ja`, `zh`）。默认为 `en`。
- `annotation_mode`：默认模式（`box` 或 `poly`）。
- `iou_threshold`：评估匹配的最小 IoU 阈值。

> 📌 `images_dir` 和 `train_dir` 是**外部目录**，可位于工具文件夹外。您可以设置为任意位置。

### 运行
```bash
python labeler.py
```

> ⚠️ **性能提示**：首次启动后的第一次推理可能较慢，因为模型需要加载。对于大型图像文件夹，建议将其拆分为子文件夹以减少图像切换时间。

---

## 导航

| 功能 | 描述 |
|--------|-------------|
| **图像列表** | 通过菜单打开：`视图 > 显示/隐藏图像列表`。点击项目跳转至对应图像。背景色表示状态：白色（未处理）、黄色（已标注）、绿色（已保存至 Train）。 |
| **类别列表** | 通过菜单打开：`视图 > 显示/隐藏类别`。点击类别，将其分配给当前选中的标注。 |
| **评估摘要** | 通过菜单打开：`视图 > 显示/隐藏评估摘要`。显示当前图像的统计信息。 |

---

## 标注模式

| 模式 | 快捷键 | 描述 |
|------|----------|-------------|
| **矩形框模式** | `B` | 绘制矩形边界框 |
| **多边形模式** | `P` | 绘制自由形状多边形，可控制顶点 |

> 📌 在**多边形模式**下，放置至少两个点后，右键单击完成绘制。

---

## 快捷键

| 键 | 动作 |
|-----|--------|
| `←` / `→` | 上一张 / 下一张图像 |
| `Z` | 随机图像 |
| `N` | 将当前图像及标注保存至 Train 目录（移除置信度分数） |
| `R` | 重置标注（在当前图像上重新运行模型） |
| `H` | 切换标注可见性 |
| `V` | 重置视图（居中 + 缩放 1x） |
| `空格` | 暂时隐藏标注 |
| `Delete` | 删除选中标注 |
| `0-9` | 设置选中框的类别 ID（0-9） |
| `Ctrl+←/→/↑/↓` | 微调选中框大小 |
| `E` | 切换评估模式 |
| `C` | 切换类别名称显示（ID vs 名称） |
| `B` | 切换至矩形框模式 |
| `P` | 切换至多边形模式 |

> 🔍 **注意**：`N` 会将标注保存到 `train_dir` 并**移除置信度分数**，使其适用于训练。标注在工具内仍根据阈值设置保持可见。

---

## 文件格式

### YOLO 边界框格式
每个 `.txt` 文件与图像同名：
```
<class_id> <x_center> <y_center> <width> <height> [<score>]
```

### 多边形（用于 OBB 和分割）
多边形标注格式：
```
<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn> [<score>]
```

- 坐标已归一化为 `[0,1]`（相对于图像宽高）
- **置信度分数在保存至 `train_dir` 时被省略**（真实数据格式）

---

## 评估模式

启用后（按 `E` 键）：
- 将当前标注与 `train_dir` 中的真实数据进行比较
- 高亮显示：
  - 🔴 **假阳性 (FP)**：检测到但不在真实数据中
  - 🔵 **假阴性 (FN)**：存在于真实数据中但未被检测到
- **真阳性 (TP) 不会可视化** —— 默认视为正确。
- 要打开**错误列表**：
  1. 进入**评估摘要**面板（`视图 > 显示/隐藏评估摘要`）。
  2. **双击** **FP** 或 **FN** 列中某个类别的单元格。
  3. “错误列表”侧边栏将打开，仅显示该类型和类别的错误。
- 要跳转到错误：
  - **双击**“错误列表”中的某一项。应用程序将加载对应的图像。

> 📊 **错误列表行为**：
> - 仅显示**所选行类别**的错误。
> - 仅显示**点击类型（FP/FN）** 的错误。
> - 双击某行将在查看器中打开对应图像。

---

## 类别管理

使用**视图 > 显示/隐藏类别**打开类别列表。
点击任一类，将其分配给当前选中的标注。

要管理类别：
- **菜单 > 操作 > 管理类别...**
- 添加、重命名、删除或从模型重置类别
- 为每个类别分配自定义颜色

---

## 项目组织

```
your-project/
├── config.json
├── labeler.py
├── images/                 # 您的原始图像（在 config.json 中设置）
│   ├── img1.jpg
│   ├── img1.txt            # 自动生成或手动标注
│   └── ...
├── TrainDataset/           # 导出的训练数据（在 config.json 中设置）
│   ├── img1.jpg
│   └── img1.txt
└── README.md               # 此文件
```

> 📌 注释保存在 `images_dir` 中。使用 `N` 将图像和注释复制到 `train_dir`，并移除置信度分数。

---

## 故障排除

| 问题 | 解决方案 |
|-------|----------|
| 未加载模型 | 检查 `config.json` 中的 `model_path`；确保 `.pt` 文件存在 |
| 首次推理缓慢 | 正常现象 — 模型首次使用时加载 |
| 无标注显示 | 按 `H` 切换可见性；检查阈值滑块 |
| 类别名称错误 | 重新加载模型或使用类别管理器中的“从模型重置” |
| 错误列表未打开 | 在评估摘要中**双击 FP 或 FN 单元格** —— **不是**行标题 |
| 图像无法加载 | 确保 `config.json` 中的 `images_dir` 正确且包含有效图像文件 |

---

## 致谢

- [Ultralytics](https://github.com/ultralytics/ultralytics) 提供 YOLO
- [OpenCV](https://opencv.org/) 提供计算机视觉
- [PyQt5](https://pypi.org/project/PyQt5/) 提供图形界面框架
- [Qwen](https://chat.qwen.ai/) 提供 AI 辅助开发

© 2025 MIT 许可证
