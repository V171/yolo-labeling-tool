# YOLO Labeling Tool

[![English](https://img.shields.io/badge/English-007BFF?style=for-the-badge&logo=google-chrome)](README.md)
[![Русский](https://img.shields.io/badge/%D0%A0%D1%83%D1%81%D1%81%D0%BA%D0%B8%D0%B9-2E7D32?style=for-the-badge&logo=google-chrome)](README-ru.md)
[![Français](https://img.shields.io/badge/Fran%C3%A7ais-0055A4?style=for-the-badge&logo=google-chrome)](README-fr.md)
[![Deutsch](https://img.shields.io/badge/Deutsch-000000?style=for-the-badge&logo=google-chrome)](README-de.md)
[![日本語](https://img.shields.io/badge/%E6%97%A5%E6%9C%AC%E8%AA%9E-BC002D?style=for-the-badge&logo=google-chrome)](README-ja.md)
[![中文](https://img.shields.io/badge/%E4%B8%AD%E6%96%87-007A33?style=for-the-badge&logo=google-chrome)](README-zh.md)

A powerful tool for annotating images in YOLO format with support for automatic labeling using neural networks.

## Overview

This application provides a complete annotation workflow:
- Automatic bounding box and polygon generation via pretrained YOLO models (detect/OBB/segment)
- Manual editing of annotations (add, delete, move, resize)
- Real-time evaluation against ground truth data
- Project management and export to training datasets

**Key Features:**
- ✅ Dual annotation modes: Bounding Box & Polygon
- ✅ Model-based auto-labeling (YOLO detect/OBB/segment)
- ✅ IoU-based evaluation mode with FP/FN visualization
- ✅ Class color coding and custom class manager
- ✅ Full keyboard navigation and hotkeys
- ✅ Configurable UI language (multi-language supported)

> 💡 **Tip**: Use `H` to toggle annotations, `N` to save to train, `E` to enable evaluation mode.

---

## Quick Start

### Prerequisites
- Python 3.8+
- Windows / Linux / macOS

### Installation
```bash
git clone https://github.com/V171/yolo-labeling-tool.git
cd yolo-labeling-tool
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install ultralytics opencv-python pyqt5 numpy
```

### Configuration
Edit `config.json`:
```json
{
  "model_path": "yolov8n.pt",
  "images_dir": "images",
  "train_dir": "TrainDataset",
  "language": "en",
  "annotation_mode": "box",
  "iou_threshold": 0.5
}
```
- `model_path`: Path to your YOLO model (.pt).
- `images_dir`: Directory containing raw images for annotation.
- `train_dir`: Directory to save annotated images as ground truth (can be any path).
- `language`: UI language (`en`, `ru`, `fr`, `de`, `ja`, `zh`). Default is `en`.
- `annotation_mode`: Default mode (`box` or `poly`).
- `iou_threshold`: Minimum IoU for evaluation matching.

> 📌 The `images_dir` and `train_dir` are **external directories**, can be outside the tool's folder. You can set them to any location.

### Run
```bash
python labeler.py
```

> ⚠️ **Performance Note**: The first inference after launch may be slow due to model loading. For large image folders, consider splitting them into subfolders to reduce image switch times.

---

## Navigation

| Feature | Description |
|--------|-------------|
| **Image List** | Open via the menu: `View > Show/Hide Image List`. Click an item to jump to it. Background colors indicate status: White (unprocessed), Yellow (annotated), Green (saved to Train). |
| **Class List** | Open via the menu: `View > Show Classes`. Click a class to assign it to the selected annotation. |
| **Evaluation Summary** | Open via the menu: `View > Show Evaluation Summary`. Displays statistics for the current image. |

---

## Annotation Modes

| Mode | Shortcut | Description |
|------|----------|-------------|
| **Box** | `B` | Draw rectangular bounding boxes |
| **Polygon** | `P` | Draw free-form polygons with vertex control |

> 📌 In **Polygon mode**, right-click to finish drawing after placing at least 2 points.

---

## Hotkeys

| Key | Action |
|-----|--------|
| `←` / `→` | Previous / Next image |
| `Z` | Random image |
| `N` | Save current image + annotations to Train directory (removes confidence scores) |
| `R` | Reset annotations (re-run model on current image) |
| `H` | Toggle annotations visibility |
| `V` | Reset view (center + scale 1x) |
| `Space` | Temporarily hide annotations |
| `Delete` | Delete selected annotation |
| `0-9` | Set class ID (0-9) for selected box |
| `Ctrl+←/→/↑/↓` | Fine-tune selected box size |
| `E` | Toggle Evaluation Mode |
| `C` | Toggle class name display (ID vs Name) |
| `B` | Switch to Box Mode |
| `P` | Switch to Polygon Mode |

> 🔍 **Note**: `N` saves annotations to `train_dir` and **removes confidence scores** — making them suitable for training. Annotations remain visible in the tool based on the threshold setting.

---

## File Formats

### YOLO box format 
Each `.txt` file matches its image name:
```
<class_id> <x_center> <y_center> <width> <height> [<score>]
```

### Polygons (for OBB and segmentation)
For polygon annotations, format is:
```
<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn> [<score>]
```

- Coordinates are normalized `[0,1]` according to image width and height
- **Confidence score is omitted** when saved to `train_dir` (ground truth format)

---

## Evaluation Mode

When enabled (`E` key):
- Compares current annotations against ground truth in `train_dir`
- Highlights:
  - 🔴 **False Positives** (FP): Detected but not in GT
  - 🔵 **False Negatives** (FN): In GT but not detected
- **True Positives (TP) are NOT visualized** — they are assumed correct.
- To open the **Error List**:
  1.  Go to the **Evaluation Summary** panel (`View > Show Evaluation Summary`).
  2.  **Double-click** a cell in the **FP** or **FN** column for a specific class.
  3.  The **Error List** dock will open, showing only errors of that type and class.
- To navigate to an error:
  - **Double-click** an item in the **Error List**. The application will load the corresponding image.

> 📊 **Error List Behavior**:
> - Only errors from the **selected row's class** appear.
> - Only errors of the **type clicked (FP/FN)** appear.
> - Double-clicking a row opens the corresponding image in the viewer.

---

## Class Management

Use **View > Show Classes** to open the class list.
Click any class to assign it to the currently selected annotation.

To manage classes:
- **Menu > Action > Manage Classes...**
- Add, rename, remove, or reset classes from the model
- Assign custom colors per class

---

## Project Organization

```
your-project/
├── config.json
├── labeler.py
├── images/                 # Your raw images (set in config.json)
│   ├── img1.jpg
│   ├── img1.txt            # Auto-generated or manual annotations
│   └── ...
├── TrainDataset/           # Exported training data (set in config.json)
│   ├── img1.jpg
│   └── img1.txt
└── README.md               # This file
```

> 📌 Annotations are saved in `images_dir`. Use `N` to copy image+annotations to `train_dir` with confidence removed.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No model loaded | Check `model_path` in `config.json`; ensure `.pt` file exists |
| Slow first inference | Normal — model loads on first use |
| No annotations shown | Press `H` to toggle visibility; check threshold slider |
| Incorrect class names | Re-load model or use "Reset from Model" in Class Manager |
| Error list not opening | Double-click **FP** or **FN** cell in Evaluation Summary — **not** the row header |
| Images not loading | Ensure `images_dir` in `config.json` is correct and contains valid image files |

---

## Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLO
- [OpenCV](https://opencv.org/) for computer vision
- [PyQt5](https://pypi.org/project/PyQt5/) for the GUI framework
- [Qwen](https://chat.qwen.ai/) for AI-assisted development

© 2025 MIT License
