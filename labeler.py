import os
import sys
import random
import shutil
import json
import configparser  # For reading .lng files
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QWidget, QVBoxLayout, QHBoxLayout,
    QSlider, QPushButton, QFileDialog, QAction, QMenu, QLineEdit, QMessageBox,
    QInputDialog, QListWidget, QDockWidget, QListWidgetItem, QMenuBar,
    QColorDialog, QDialog
)
from PyQt5.QtCore import Qt, QPoint, QRect
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QKeyEvent, QMouseEvent

def calculate_iou(box1, box2):
    """
    Calculates Intersection over Union (IoU) for two bounding boxes.
    Args:
        box1 (tuple): First box in format (x1, y1, x2, y2, cls, conf).
        box2 (tuple): Second box in format (x1, y1, x2, y2, cls, conf).
    Returns:
        float: IoU value (0 to 1). 0 - no intersection, 1 - perfect overlap.
    """
    x1_1, y1_1, x2_1, y2_1 = box1[:4]
    x1_2, y1_2, x2_2, y2_2 = box2[:4]
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

class ImageViewer(QLabel):
    """Widget for displaying the image and annotations."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setAlignment(Qt.AlignTop)
        self.setMinimumSize(640, 480)
        self.scale_factor = 1.0
        self.offset = QPoint(0, 0)
        self.drag_start = None
        self.selected_box_idx = -1
        self.dragging = False
        self.drag_box_start = None
        self.drawing_box = False
        self.draw_start = None
        self.draw_end = None
        self.resizing = False
        self.resize_corner = -1
        self.corner_size = 8
        self.last_click_pos = None
        self.show_annotations = True
        self.show_class_names = True
        self.original_box = None

    def set_image(self, image):
        self.orig_image = image.copy()
        self.image = image
        self.update_display()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_display()

    def update_display(self):
        """Redraws the image and annotations."""
        if not hasattr(self, 'image') or self.image is None:
            return
        h, w, _ = self.image.shape
        scaled_w = int(w * self.scale_factor)
        scaled_h = int(h * self.scale_factor)
        scaled_img = cv2.resize(self.image, (scaled_w, scaled_h))
        bytes_per_line = 3 * scaled_w
        qimg = QImage(scaled_img.data, scaled_w, scaled_h, bytes_per_line,
                      QImage.Format_BGR888)
        canvas = QPixmap(self.parent.width(), self.parent.height())
        canvas.fill(Qt.black)
        painter = QPainter(canvas)
        painter.drawImage(self.offset.x(), self.offset.y(), qimg)

        if self.show_annotations:
            visible_boxes = [box for box in self.parent.current_boxes
                           if box[5] >= self.parent.current_threshold
                           and box[4] in self.parent.class_names]
            for box in visible_boxes:
                if len(box) < 6:
                    continue
                x1, y1, x2, y2, cls, conf = box
                sx1 = int(x1 * self.scale_factor) + self.offset.x()
                sy1 = int(y1 * self.scale_factor) + self.offset.y()
                sx2 = int(x2 * self.scale_factor) + self.offset.x()
                sy2 = int(y2 * self.scale_factor) + self.offset.y()
                color = self.parent.get_class_color(cls)
                painter.setPen(Qt.NoPen)
                fill_color = QColor(color)
                fill_color.setAlpha(30)
                painter.setBrush(fill_color)
                painter.drawRect(sx1, sy1, sx2 - sx1, sy2 - sy1)
                painter.setBrush(Qt.NoBrush)
                original_index = self.parent.current_boxes.index(box)
                if original_index == self.selected_box_idx:
                    pen = QPen(color, 3, Qt.SolidLine)
                else:
                    pen = QPen(color, 2, Qt.SolidLine)
                painter.setPen(pen)
                painter.drawRect(sx1, sy1, sx2 - sx1, sy2 - sy1)
                if original_index == self.selected_box_idx:
                    corner_size = self.corner_size
                    corners = [
                        QRect(sx1 - corner_size//2, sy1 - corner_size//2, corner_size, corner_size),
                        QRect(sx2 - corner_size//2, sy1 - corner_size//2, corner_size, corner_size),
                        QRect(sx1 - corner_size//2, sy2 - corner_size//2, corner_size, corner_size),
                        QRect(sx2 - corner_size//2, sy2 - corner_size//2, corner_size, corner_size)
                    ]
                    painter.setBrush(Qt.yellow)
                    for corner in corners:
                        painter.drawRect(corner)
                    painter.setBrush(Qt.NoBrush)
                painter.setPen(QPen(Qt.white, 1))
                if self.show_class_names and cls in self.parent.class_names:
                    class_text = self.parent.class_names[cls]
                else:
                    class_text = str(cls)
                painter.drawText(sx1 + 5, sy1 + 15, f"{class_text}: {conf:.2f}")

        if self.drawing_box and self.draw_start and self.draw_end:
            start_x = int(self.draw_start.x() * self.scale_factor) + self.offset.x()
            start_y = int(self.draw_start.y() * self.scale_factor) + self.offset.y()
            end_x = int(self.draw_end.x() * self.scale_factor) + self.offset.x()
            end_y = int(self.draw_end.y() * self.scale_factor) + self.offset.y()
            pen = QPen(Qt.cyan, 2, Qt.DashLine)
            painter.setPen(pen)
            painter.drawRect(
                min(start_x, end_x),
                min(start_y, end_y),
                abs(end_x - start_x),
                abs(end_y - start_y)
            )
        painter.end()
        self.setPixmap(canvas)

    def wheelEvent(self, event):
        zoom_in = event.angleDelta().y() > 0
        old_scale = self.scale_factor
        if zoom_in:
            self.scale_factor *= 1.1
        else:
            self.scale_factor /= 1.1
        self.scale_factor = max(0.1, min(self.scale_factor, 5.0))
        cursor_pos = event.pos()
        if old_scale != 0:
            img_x = (cursor_pos.x() - self.offset.x()) / old_scale
            img_y = (cursor_pos.y() - self.offset.y()) / old_scale
            new_x = cursor_pos.x() - img_x * self.scale_factor
            new_y = cursor_pos.y() - img_y * self.scale_factor
            self.offset = QPoint(int(new_x), int(new_y))
        self.update_display()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            orig_pos = self.screen_to_image_pos(event.pos())
            self.last_click_pos = orig_pos
            self.resizing = False
            self.resize_corner = -1
            if self.selected_box_idx >= 0:
                if self.selected_box_idx < len(self.parent.current_boxes):
                    box = self.parent.current_boxes[self.selected_box_idx]
                    if len(box) < 4:
                        return
                    x1, y1, x2, y2 = box[:4]
                    corners = [
                        QPoint(x1, y1), QPoint(x2, y1),
                        QPoint(x1, y2), QPoint(x2, y2)
                    ]
                    for i, corner in enumerate(corners):
                        if self.distance(orig_pos, corner) < 10:
                            self.resizing = True
                            self.resize_corner = i
                            self.drag_box_start = orig_pos
                            self.original_box = box[:]
                            return
            visible_boxes = [box for box in self.parent.current_boxes
                           if box[5] >= self.parent.current_threshold
                           and box[4] in self.parent.class_names]
            clicked_boxes_indices = []
            for i, box in enumerate(self.parent.current_boxes):
                if box[5] < self.parent.current_threshold or box[4] not in self.parent.class_names:
                    continue
                if len(box) < 4:
                    continue
                x1, y1, x2, y2 = box[:4]
                if x1 <= orig_pos.x() <= x2 and y1 <= orig_pos.y() <= y2:
                    clicked_boxes_indices.append(i)
            if clicked_boxes_indices:
                if self.selected_box_idx in clicked_boxes_indices:
                    current_idx = clicked_boxes_indices.index(self.selected_box_idx)
                    next_idx = (current_idx + 1) % len(clicked_boxes_indices)
                    self.selected_box_idx = clicked_boxes_indices[next_idx]
                else:
                    self.selected_box_idx = clicked_boxes_indices[0]
                self.drag_box_start = orig_pos
                self.dragging = True
                self.original_box = self.parent.current_boxes[self.selected_box_idx][:]
                self.update_display()
                self.parent.update_status()
                return
            self.drag_start = event.pos()
            self.dragging = False
            self.selected_box_idx = -1
        elif event.button() == Qt.RightButton:
            self.show_context_menu(event.pos())

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            orig_pos = self.screen_to_image_pos(event.pos())
            if self.drawing_box:
                if not self.draw_start:
                    self.draw_start = orig_pos
                else:
                    self.draw_end = orig_pos
                self.update_display()
            elif self.resizing and self.selected_box_idx >= 0:
                start_pos = self.drag_box_start
                dx = orig_pos.x() - start_pos.x()
                dy = orig_pos.y() - start_pos.y()
                box = list(self.parent.current_boxes[self.selected_box_idx])
                if self.resize_corner == 0:
                    box[0] += dx; box[1] += dy
                elif self.resize_corner == 1:
                    box[2] += dx; box[1] += dy
                elif self.resize_corner == 2:
                    box[0] += dx; box[3] += dy
                elif self.resize_corner == 3:
                    box[2] += dx; box[3] += dy
                self.parent.current_boxes[self.selected_box_idx] = tuple(box)
                self.drag_box_start = orig_pos
                self.update_display()
            elif self.dragging and self.selected_box_idx >= 0:
                start_pos = self.drag_box_start
                dx = orig_pos.x() - start_pos.x()
                dy = orig_pos.y() - start_pos.y()
                box = list(self.parent.current_boxes[self.selected_box_idx])
                box[0] += dx; box[1] += dy; box[2] += dx; box[3] += dy
                self.parent.current_boxes[self.selected_box_idx] = tuple(box)
                self.drag_box_start = orig_pos
                self.update_display()
            elif self.drag_start is not None:
                diff = event.pos() - self.drag_start
                self.offset += diff
                self.drag_start = event.pos()
                self.update_display()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.drawing_box and self.draw_start and self.draw_end:
                x1 = min(self.draw_start.x(), self.draw_end.x())
                y1 = min(self.draw_start.y(), self.draw_end.y())
                x2 = max(self.draw_start.x(), self.draw_end.x())
                y2 = max(self.draw_start.y(), self.draw_end.y())
                if hasattr(self, 'image') and self.image is not None:
                    img_h, img_w, _ = self.image.shape
                    x1 = max(0, min(x1, img_w)); y1 = max(0, min(y1, img_h))
                    x2 = max(0, min(x2, img_w)); y2 = max(0, min(y2, img_h))
                if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
                    cls = self.parent.get_selected_class()
                    new_box = (x1, y1, x2, y2, cls, 0.99)
                    self.parent.current_boxes.append(new_box)
                    self.selected_box_idx = len(self.parent.current_boxes) - 1
                    self.parent.save_annotations_to_file()
            elif self.resizing and self.selected_box_idx >= 0 and self.original_box:
                current_box = self.parent.current_boxes[self.selected_box_idx]
                if (self.original_box[0] != current_box[0] or
                    self.original_box[1] != current_box[1] or
                    self.original_box[2] != current_box[2] or
                    self.original_box[3] != current_box[3]):
                    box = list(current_box); box[5] = 0.99
                    self.parent.clip_box_to_image_bounds(box)
                    self.parent.current_boxes[self.selected_box_idx] = tuple(box)
                    self.parent.save_annotations_to_file()
            elif self.dragging and self.selected_box_idx >= 0 and self.original_box:
                current_box = self.parent.current_boxes[self.selected_box_idx]
                if (self.original_box[0] != current_box[0] or
                    self.original_box[1] != current_box[1] or
                    self.original_box[2] != current_box[2] or
                    self.original_box[3] != current_box[3]):
                    box = list(current_box); box[5] = 0.99
                    self.parent.clip_box_to_image_bounds(box)
                    self.parent.current_boxes[self.selected_box_idx] = tuple(box)
                    self.parent.save_annotations_to_file()
            self.drag_start = None
            self.drag_box_start = None
            self.drawing_box = False
            self.draw_start = None
            self.draw_end = None
            self.original_box = None
            self.resizing = False
            self.dragging = False
            self.update_display()
            self.parent.update_status()

    def distance(self, p1, p2):
        return ((p1.x() - p2.x()) ** 2 + (p1.y() - p2.y()) ** 2) ** 0.5

    def screen_to_image_pos(self, pos):
        if self.scale_factor != 0:
            x = (pos.x() - self.offset.x()) / self.scale_factor
            y = (pos.y() - self.offset.y()) / self.scale_factor
            return QPoint(int(x), int(y))
        return QPoint(0, 0)

    def show_context_menu(self, pos):
        """Shows context menu for box operations."""
        menu = QMenu(self)
        # Actions will be translated using parent's tr method
        self.add_box_action = menu.addAction(self.parent.tr("Add Box"))
        self.delete_action = menu.addAction(self.parent.tr("Delete Box"))
        self.change_class_action = menu.addAction(self.parent.tr("Change Class"))
        self.resize_action = menu.addAction(self.parent.tr("Resize"))
        action = menu.exec_(self.mapToGlobal(pos))
        if action == self.add_box_action:
            self.drawing_box = True
            self.draw_start = None
            self.draw_end = None
        elif action == self.delete_action and 0 <= self.selected_box_idx < len(self.parent.current_boxes):
            del self.parent.current_boxes[self.selected_box_idx]
            self.selected_box_idx = -1
            self.update_display()
            self.parent.update_status()
            self.parent.save_annotations_to_file()
        elif action == self.change_class_action and 0 <= self.selected_box_idx < len(self.parent.current_boxes):
            new_class, ok = QInputDialog.getInt(
                self, self.parent.tr("Change Class"),
                self.parent.tr("Enter new class:"),
                value=self.parent.current_boxes[self.selected_box_idx][4],
                min=0, max=100
            )
            if ok:
                box = list(self.parent.current_boxes[self.selected_box_idx])
                box[4] = new_class
                self.parent.current_boxes[self.selected_box_idx] = tuple(box)
                self.update_display()
                self.parent.save_annotations_to_file()
        elif action == self.resize_action and 0 <= self.selected_box_idx < len(self.parent.current_boxes):
            current_size = int(self.parent.current_boxes[self.selected_box_idx][2] -
                           self.parent.current_boxes[self.selected_box_idx][0])
            new_size, ok = QInputDialog.getInt(
                self, self.parent.tr("Resize"),
                self.parent.tr("Enter new size:"),
                value=current_size, min=10, max=500
            )
            if ok:
                box = list(self.parent.current_boxes[self.selected_box_idx])
                cx = (box[0] + box[2]) // 2
                cy = (box[1] + box[3]) // 2
                half_size = new_size // 2
                box[0] = cx - half_size; box[1] = cy - half_size
                box[2] = cx + half_size; box[3] = cy + half_size
                self.parent.clip_box_to_image_bounds(box)
                self.parent.current_boxes[self.selected_box_idx] = tuple(box)
                self.update_display()
                self.parent.save_annotations_to_file()

class ClassManagerDialog(QDialog):
    """Dialog for managing custom classes."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle(self.parent.tr("Class Manager"))
        self.setGeometry(200, 200, 400, 300)
        self.init_ui()

    def init_ui(self):
        from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QPushButton
        layout = QVBoxLayout()
        self.class_list = QListWidget()
        self.update_class_list()
        layout.addWidget(self.class_list)
        btn_layout = QHBoxLayout()
        self.btn_add = QPushButton(self.parent.tr("Add"))
        self.btn_add.clicked.connect(self.add_class)
        btn_layout.addWidget(self.btn_add)
        self.btn_remove = QPushButton(self.parent.tr("Remove"))
        self.btn_remove.clicked.connect(self.remove_class)
        btn_layout.addWidget(self.btn_remove)
        self.btn_rename = QPushButton(self.parent.tr("Rename"))
        self.btn_rename.clicked.connect(self.rename_class)
        btn_layout.addWidget(self.btn_rename)
        self.btn_reset = QPushButton(self.parent.tr("Reset from Model"))
        self.btn_reset.clicked.connect(self.reset_classes)
        btn_layout.addWidget(self.btn_reset)
        layout.addLayout(btn_layout)
        ok_cancel_layout = QHBoxLayout()
        self.btn_ok = QPushButton(self.parent.tr("OK"))
        self.btn_ok.clicked.connect(self.accept)
        ok_cancel_layout.addWidget(self.btn_ok)
        self.btn_cancel = QPushButton(self.parent.tr("Cancel"))
        self.btn_cancel.clicked.connect(self.reject)
        ok_cancel_layout.addWidget(self.btn_cancel)
        layout.addLayout(ok_cancel_layout)
        self.setLayout(layout)

    def update_class_list(self):
        self.class_list.clear()
        for class_id, class_name in self.parent.class_names.items():
            color = self.parent.get_class_color(class_id)
            item = QListWidgetItem(f"{class_id}: {class_name}")
            item.setBackground(color)
            text_color = self.parent.get_contrast_text_color(color)
            item.setForeground(QColor(text_color))
            self.class_list.addItem(item)

    def add_class(self):
        new_id = 0
        while new_id in self.parent.class_names:
            new_id += 1
        name, ok = QInputDialog.getText(self, self.parent.tr("Add Class"),
                                      self.parent.tr("Enter class name:"))
        if ok and name:
            self.parent.class_names[new_id] = name
            self.parent.get_class_color(new_id)
            self.update_class_list()
            self.parent.update_class_list()

    def remove_class(self):
        current_item = self.class_list.currentItem()
        if current_item:
            class_id = int(current_item.text().split(":")[0])
            reply = QMessageBox.question(
                self, self.parent.tr("Confirm"),
                self.parent.tr("Are you sure you want to remove class {name} (ID: {id})?").format(
                    name=self.parent.class_names[class_id], id=class_id),
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                del self.parent.class_names[class_id]
                if class_id in self.parent.class_colors:
                    del self.parent.class_colors[class_id]
                self.update_class_list()
                self.parent.update_class_list()

    def rename_class(self):
        current_item = self.class_list.currentItem()
        if current_item:
            class_id = int(current_item.text().split(":")[0])
            old_name = self.parent.class_names[class_id]
            new_name, ok = QInputDialog.getText(
                self, self.parent.tr("Rename Class"),
                self.parent.tr("Enter new class name:"), text=old_name)
            if ok and new_name and new_name != old_name:
                self.parent.class_names[class_id] = new_name
                self.update_class_list()
                self.parent.update_class_list()

    def reset_classes(self):
        reply = QMessageBox.question(
            self, self.parent.tr("Confirm"),
            self.parent.tr("Are you sure you want to reset classes from the model? All custom changes will be lost."),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            if hasattr(self.parent.model, 'names'):
                self.parent.class_names = self.parent.model.names.copy()
            else:
                self.parent.class_names = {}
            self.parent.class_colors = {}
            self.update_class_list()
            self.parent.update_class_list()

class YOLOLabeler(QMainWindow):
    """Main application window."""
    def __init__(self):
        super().__init__()
        self.language = 'ru' # Default language
        self.setWindowTitle("YOLO Labeling Tool")
        self.setGeometry(100, 100, 1400, 800)
        self.config_file = "config.json"
        self.model = None
        self.image_paths = []
        self.current_idx = 0
        self.current_threshold = 0.25
        self.current_boxes = []
        self.class_names = {}
        self.class_colors = {}
        self.selected_class_id = 0
        self.translations = {} # Stores loaded translations
        self.load_config()
        self.load_translations() # Load translations for the default/current language
        self.init_ui()
        self.load_model()
        self.create_dirs()
        if os.path.exists(self.images_dir):
            self.load_folder(self.images_dir)
        self.showMaximized()

    def load_translations(self):
        """Loads translations from a .lng file based on the current language."""
        lang_file = f"{self.language}.lng"
        if not os.path.exists(lang_file):
            print(f"Translation file {lang_file} not found. Using keys as fallback.")
            self.translations = {}
            return
        config = configparser.ConfigParser()
        config.optionxform = str # Preserve key case
        try:
            # Specify encoding to handle UTF-8 correctly (ensure file is saved without BOM for ja.lng)
            config.read(lang_file, encoding='utf-8')
            if 'UI' in config:
                self.translations = dict(config['UI'])
            else:
                print(f"Section [UI] not found in {lang_file}")
                self.translations = {}
        except Exception as e:
            print(f"Error loading translation from {lang_file}: {e}")
            self.translations = {}

    def tr(self, key):
        """Translates a key to the current language."""
        return self.translations.get(key, key)

    def init_ui(self):
        self.setWindowTitle(self.tr("YOLO Labeling Tool"))
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        toolbar_layout = QHBoxLayout()
        self.btn_open = QPushButton(self.tr("Open Folder"))
        self.btn_open.clicked.connect(self.open_folder)
        self.btn_open.setFocusPolicy(Qt.NoFocus)
        toolbar_layout.addWidget(self.btn_open)

        self.btn_first = QPushButton(self.tr("⏮️ First (Home)"))
        self.btn_first.clicked.connect(self.first_image)
        self.btn_first.setFocusPolicy(Qt.NoFocus)
        toolbar_layout.addWidget(self.btn_first)
        self.btn_prev_10 = QPushButton(self.tr("⏪ -10 (PgDn)"))
        self.btn_prev_10.clicked.connect(self.prev_10_images)
        self.btn_prev_10.setFocusPolicy(Qt.NoFocus)
        toolbar_layout.addWidget(self.btn_prev_10)
        self.btn_prev = QPushButton(self.tr("◀ Previous (←)"))
        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_prev.setFocusPolicy(Qt.NoFocus)
        toolbar_layout.addWidget(self.btn_prev)
        self.btn_next = QPushButton(self.tr("Next (→) ▶"))
        self.btn_next.clicked.connect(self.next_image)
        self.btn_next.setFocusPolicy(Qt.NoFocus)
        toolbar_layout.addWidget(self.btn_next)
        self.btn_next_10 = QPushButton(self.tr("+10 (PgUp) ⏩"))
        self.btn_next_10.clicked.connect(self.next_10_images)
        self.btn_next_10.setFocusPolicy(Qt.NoFocus)
        toolbar_layout.addWidget(self.btn_next_10)
        self.btn_last = QPushButton(self.tr("Last (End) ⏭️"))
        self.btn_last.clicked.connect(self.last_image)
        self.btn_last.setFocusPolicy(Qt.NoFocus)
        toolbar_layout.addWidget(self.btn_last)
        self.btn_random = QPushButton(self.tr("Random (Z)"))
        self.btn_random.clicked.connect(self.random_image)
        self.btn_random.setFocusPolicy(Qt.NoFocus)
        toolbar_layout.addWidget(self.btn_random)
        self.btn_reset = QPushButton(self.tr("Reset Annotations (R)"))
        self.btn_reset.clicked.connect(self.reset_annotations)
        self.btn_reset.setFocusPolicy(Qt.NoFocus)
        toolbar_layout.addWidget(self.btn_reset)
        self.btn_train = QPushButton(self.tr("Save to Train (N)"))
        self.btn_train.clicked.connect(lambda: self.save_annotations("train"))
        self.btn_train.setFocusPolicy(Qt.NoFocus)
        toolbar_layout.addWidget(self.btn_train)

        self.btn_toggle_annotations = QPushButton(self.tr("Hide Annotations (H)"))
        self.btn_toggle_annotations.setCheckable(True)
        self.btn_toggle_annotations.clicked.connect(self.toggle_annotations)
        self.btn_toggle_annotations.setFocusPolicy(Qt.NoFocus)
        toolbar_layout.addWidget(self.btn_toggle_annotations)

        self.btn_toggle_class_display = QPushButton(self.tr("Numbers (C)"))
        self.btn_toggle_class_display.clicked.connect(self.toggle_class_display)
        self.btn_toggle_class_display.setFocusPolicy(Qt.NoFocus)
        toolbar_layout.addWidget(self.btn_toggle_class_display)

        self.btn_reset_view = QPushButton(self.tr("Reset View (V)"))
        self.btn_reset_view.clicked.connect(self.reset_view)
        self.btn_reset_view.setFocusPolicy(Qt.NoFocus)
        toolbar_layout.addWidget(self.btn_reset_view)

        toolbar_layout.addStretch()
        self.threshold_label = QLabel(self.tr("Threshold:"))
        toolbar_layout.addWidget(self.threshold_label)
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(25)
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        self.threshold_slider.setFocusPolicy(Qt.NoFocus)
        toolbar_layout.addWidget(self.threshold_slider)
        self.threshold_value = QLabel("0.25")
        toolbar_layout.addWidget(self.threshold_value)
        main_layout.addLayout(toolbar_layout)

        self.image_viewer = ImageViewer(self)
        main_layout.addWidget(self.image_viewer)

        self.create_image_list()
        self.create_class_list()
        self.create_language_menu()
        self.status_bar = self.statusBar()
        self.update_status()

    def makeBold(self, current, previous):
        if not current: return
        from PyQt5.QtGui import QFont
        font = current.font()
        sz = font.pointSize()
        font.setBold(True)
        font.setPointSize(14)
        current.setFont(font)
        if previous:
            font.setBold(False)
            font.setPointSize(sz)
            previous.setFont(font)

    def create_image_list(self):
        self.dock = QDockWidget(self.tr("Image List"), self)
        self.dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.image_list = QListWidget()
        self.image_list.itemClicked.connect(self.on_image_selected)
        self.image_list.currentItemChanged.connect(self.makeBold)
        self.dock.setWidget(self.image_list)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock)
        self.dock.hide()
        self.toggle_list_action = QAction(self.tr("Show/Hide List"), self)
        self.toggle_list_action.triggered.connect(self.toggle_image_list)
        self.menuBar().addAction(self.toggle_list_action)

    def create_class_list(self):
        self.class_dock = QDockWidget(self.tr("Classes"), self)
        self.class_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.class_list = QListWidget()
        self.class_list.itemClicked.connect(self.on_class_selected)
        self.class_list.currentItemChanged.connect(self.makeBold)
        self.class_dock.setWidget(self.class_list)
        self.addDockWidget(Qt.RightDockWidgetArea, self.class_dock)
        self.class_dock.hide()
        self.toggle_class_list_action = QAction(self.tr("Show Classes"), self)
        self.toggle_class_list_action.triggered.connect(self.toggle_class_list)
        self.menuBar().addAction(self.toggle_class_list_action)
        self.manage_classes_action = QAction(self.tr("Manage Classes"), self)
        self.manage_classes_action.triggered.connect(self.manage_classes)
        self.menuBar().addAction(self.manage_classes_action)

    def create_language_menu(self):
        """Creates the language selection menu."""
        if not hasattr(self, 'language_menu'):
            self.language_menu = self.menuBar().addMenu(self.tr("Language"))
            self.language_actions = {}
            # Define available languages
            langs = {
                'ru': 'Русский',
                'en': 'English',
                'fr': 'Français',
                'de': 'Deutsch',
                'ja': '日本語',
                'zh': '中文'
            }
            for code, name in langs.items():
                action = QAction(name, self)
                action.setCheckable(True)
                action.setData(code)
                action.triggered.connect(self.on_language_changed)
                self.language_menu.addAction(action)
                self.language_actions[code] = action
        if self.language in self.language_actions:
            self.language_actions[self.language].setChecked(True)

    def on_language_changed(self):
        """Handles language change from the menu."""
        action = self.sender()
        if action and action.isChecked():
            new_language = action.data()
            if new_language and new_language != self.language:
                self.language = new_language
                for code, act in self.language_actions.items():
                    act.setChecked(code == self.language)
                self.load_translations()
                self.update_ui_texts()
                self.image_viewer.update_display()
                self.update_status()
                self.save_config()

    def update_ui_texts(self):
        """Updates the text of UI elements to the current language."""
        self.setWindowTitle(self.tr("YOLO Labeling Tool"))
        self.btn_open.setText(self.tr("Open Folder"))
        self.btn_first.setText(self.tr("⏮️ First (Home)"))
        self.btn_prev_10.setText(self.tr("⏪ -10 (PgDn)"))
        self.btn_prev.setText(self.tr("◀ Previous (←)"))
        self.btn_next.setText(self.tr("Next (→) ▶"))
        self.btn_next_10.setText(self.tr("+10 (PgUp) ⏩"))
        self.btn_last.setText(self.tr("Last (End) ⏭️"))
        self.btn_random.setText(self.tr("Random (Z)"))
        self.btn_reset.setText(self.tr("Reset Annotations (R)"))
        self.btn_train.setText(self.tr("Save to Train (N)"))
        if self.image_viewer.show_annotations:
            self.btn_toggle_annotations.setText(self.tr("Hide Annotations (H)"))
        else:
            self.btn_toggle_annotations.setText(self.tr("Show Annotations (H)"))
        if self.image_viewer.show_class_names:
            self.btn_toggle_class_display.setText(self.tr("Numbers (C)"))
        else:
            self.btn_toggle_class_display.setText(self.tr("Names (C)"))
        self.btn_reset_view.setText(self.tr("Reset View (V)"))
        self.threshold_label.setText(self.tr("Threshold:"))
        self.dock.setWindowTitle(self.tr("Image List"))
        self.class_dock.setWindowTitle(self.tr("Classes"))
        self.toggle_list_action.setText(self.tr("Show/Hide List"))
        self.toggle_class_list_action.setText(self.tr("Show Classes"))
        self.manage_classes_action.setText(self.tr("Manage Classes"))
        if hasattr(self, 'language_menu'):
            self.language_menu.setTitle(self.tr("Language"))
        self.update_class_list()

    def toggle_image_list(self):
        if self.dock.isHidden():
            self.dock.show()
        else:
            self.dock.hide()

    def toggle_class_list(self):
        if self.class_dock.isHidden():
            self.class_dock.show()
            self.update_class_list()
        else:
            self.class_dock.hide()

    def manage_classes(self):
        dialog = ClassManagerDialog(self)
        dialog.setWindowTitle(self.tr("Class Manager"))
        dialog.btn_add.setText(self.tr("Add"))
        dialog.btn_remove.setText(self.tr("Remove"))
        dialog.btn_rename.setText(self.tr("Rename"))
        dialog.btn_reset.setText(self.tr("Reset from Model"))
        dialog.btn_ok.setText(self.tr("OK"))
        dialog.btn_cancel.setText(self.tr("Cancel"))
        if dialog.exec_() == QDialog.Accepted:
            self.save_config()
            self.image_viewer.update_display()
            self.update_status()

    def toggle_annotations(self):
        self.image_viewer.show_annotations = not self.image_viewer.show_annotations
        if self.image_viewer.show_annotations:
            self.btn_toggle_annotations.setText(self.tr("Hide Annotations (H)"))
        else:
            self.btn_toggle_annotations.setText(self.tr("Show Annotations (H)"))
        self.image_viewer.update_display()

    def toggle_class_display(self):
        self.image_viewer.show_class_names = not self.image_viewer.show_class_names
        if self.image_viewer.show_class_names:
            self.btn_toggle_class_display.setText(self.tr("Numbers (C)"))
        else:
            self.btn_toggle_class_display.setText(self.tr("Names (C)"))
        self.image_viewer.update_display()

    def reset_view(self):
        if hasattr(self.image_viewer, 'image') and self.image_viewer.image is not None:
            h, w, _ = self.image_viewer.image.shape
            view_width = self.image_viewer.width()
            view_height = self.image_viewer.height()
            self.image_viewer.offset = QPoint(
                (view_width - w) // 2, (view_height - h) // 2)
            self.image_viewer.scale_factor = 1.0
            self.image_viewer.update_display()

    def on_image_selected(self, item):
        index = self.image_list.row(item)
        if 0 <= index < len(self.image_paths):
            self.current_idx = index
            self.load_current_image()

    def on_class_selected(self, item):
        class_id = int(item.text().split(":")[0])
        self.selected_class_id = class_id
        if 0 <= self.image_viewer.selected_box_idx < len(self.current_boxes):
            box = list(self.current_boxes[self.image_viewer.selected_box_idx])
            box[4] = class_id
            self.current_boxes[self.image_viewer.selected_box_idx] = tuple(box)
            self.image_viewer.update_display()
            self.save_annotations_to_file()
            self.update_status()

    def get_selected_class(self):
        return self.selected_class_id

    def load_config(self):
        """Loads application settings from config.json."""
        default_config = {
            "model_path": r"D:\japanwork\ultralytics\runs\detect\train34\weights\last.pt",
            "images_dir": r'K:\japanwork\VideoProject\DataSet-Tools\dataset',
            "train_dir": "TrainVideoProject",
            "last_image_index": 0,
            "selected_class_id": 0,
            "class_names": {},
            "language": "ru"
        }
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            else:
                config = default_config
                with open(self.config_file, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error loading config: {e}")
            config = default_config
        self.model_path = config.get("model_path", default_config["model_path"])
        self.images_dir = config.get("images_dir", default_config["images_dir"])
        self.train_dir = config.get("train_dir", default_config["train_dir"])
        self.last_image_index = config.get("last_image_index", default_config["last_image_index"])
        self.selected_class_id = config.get("selected_class_id", default_config["selected_class_id"])
        self.class_names = config.get("class_names", default_config["class_names"])
        self.class_names = {int(k):v for k,v in self.class_names.items()}
        self.language = config.get("language", default_config["language"])

    def save_config(self):
        """Saves current settings to config.json."""
        config = {
            "model_path": self.model_path,
            "images_dir": self.images_dir,
            "train_dir": self.train_dir,
            "last_image_index": self.current_idx,
            "selected_class_id": self.selected_class_id,
            "class_names": self.class_names,
            "language": self.language
        }
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving config: {e}")

    def get_class_color(self, class_id):
        """Returns a QColor for a given class ID."""
        if class_id not in self.class_colors:
            colors = [
                QColor(255, 0, 0), QColor(0, 255, 0), QColor(0, 0, 255),
                QColor(255, 255, 0), QColor(255, 0, 255), QColor(0, 255, 255),
                QColor(255, 165, 0), QColor(128, 0, 128), QColor(0, 128, 0),
                QColor(128, 128, 0), QColor(0, 0, 128), QColor(128, 0, 0),
                QColor(128, 128, 128), QColor(255, 192, 203), QColor(165, 42, 42),
            ]
            color_index = class_id % len(colors)
            self.class_colors[class_id] = colors[color_index]
        return self.class_colors[class_id]

    def get_contrast_text_color(self, bg_color):
        """Returns black or white text color for better contrast."""
        r, g, b = bg_color.red(), bg_color.green(), bg_color.blue()
        brightness = (r * 299 + g * 587 + b * 114) / 1000
        return Qt.black if brightness > 128 else Qt.white

    def load_model(self):
        try:
            self.model = YOLO(self.model_path)
            self.status_bar.showMessage(self.tr("Model loaded: {path}").format(path=self.model_path))
            model_classes = getattr(self.model, 'names', {})
            if not self.class_names:
                self.class_names = model_classes.copy()
            self.update_class_list()
        except Exception as e:
            QMessageBox.critical(self, self.tr("Error"),
                               self.tr("Could not load model: {error}").format(error=str(e)))

    def update_class_list(self):
        self.class_list.clear()
        for class_id, class_name in self.class_names.items():
            color = self.get_class_color(class_id)
            item = QListWidgetItem(f"{class_id}: {class_name}")
            item.setBackground(color)
            text_color = self.get_contrast_text_color(color)
            item.setForeground(QColor(text_color))
            self.class_list.addItem(item)
            if class_id == self.selected_class_id:
                self.class_list.setCurrentItem(item)

    def create_dirs(self):
        Path(self.train_dir).mkdir(parents=True, exist_ok=True)
        self.load_processed_images()

    def load_processed_images(self):
        self.train_images = set()
        if os.path.exists(self.train_dir):
            for filename in os.listdir(self.train_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.train_images.add(filename)

    def load_folder(self, folder):
        self.images_dir = folder
        self.image_paths = sorted([
            os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.image_list.clear()
        for i, path in enumerate(self.image_paths):
            filename = os.path.basename(path)
            item = QListWidgetItem(filename)
            txt_name = os.path.splitext(filename)[0] + ".txt"
            txt_path = os.path.join(os.path.dirname(path), txt_name)
            is_annotated = os.path.exists(txt_path)
            if filename in self.train_images:
                item.setBackground(QColor(144, 238, 144))
            elif is_annotated:
                item.setBackground(QColor(255, 255, 144))
            else:
                item.setBackground(QColor(255, 255, 255))
            self.image_list.addItem(item)
            # if i == self.current_idx: # Not needed as setCurrentRow is called later
            #     self.image_list.setCurrentItem(item)
        if self.image_paths:
            self.current_idx = min(self.last_image_index, len(self.image_paths) - 1)
            self.load_current_image()
            self.reset_view()
        else:
            QMessageBox.warning(self, self.tr("Warning"), self.tr("No images in folder"))

    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder:
            self.load_folder(folder)
            self.images_dir = folder
            self.save_config()

    def first_image(self):
        if self.image_paths:
            self.current_idx = 0
            self.load_current_image()

    def last_image(self):
        if self.image_paths:
            self.current_idx = len(self.image_paths) - 1
            self.load_current_image()

    def prev_10_images(self):
        if self.image_paths:
            self.current_idx = max(0, self.current_idx - 10)
            self.load_current_image()

    def next_10_images(self):
        if self.image_paths:
            self.current_idx = min(len(self.image_paths) - 1, self.current_idx + 10)
            self.load_current_image()

    def prev_image(self):
        if self.image_paths and self.current_idx > 0:
            self.current_idx -= 1
            self.load_current_image()

    def next_image(self):
        if self.image_paths and self.current_idx < len(self.image_paths) - 1:
            self.current_idx += 1
            self.load_current_image()

    def random_image(self):
        if self.image_paths:
            self.current_idx = random.randint(0, len(self.image_paths) - 1)
            self.load_current_image()

    def reset_annotations(self):
        if self.image_paths:
            image_path = self.image_paths[self.current_idx]
            image_name = os.path.basename(image_path)
            txt_name = os.path.splitext(image_name)[0] + ".txt"
            txt_path = os.path.join(self.images_dir, txt_name)
            if os.path.exists(txt_path):
                os.remove(txt_path)
            self.create_annotations_file(image_path)
            self.load_current_image()

    def load_current_image(self):
        if not self.image_paths:
            return
        sbox = None
        if self.image_viewer.selected_box_idx >= 0:
            sbox = self.current_boxes[self.image_viewer.selected_box_idx]
        self.image_viewer.selected_box_idx = -1
        image_path = self.image_paths[self.current_idx]
        image_name = os.path.basename(image_path)
        self.image_list.setCurrentRow(self.current_idx)
        image = cv2.imread(image_path)
        if image is None:
            QMessageBox.warning(self, self.tr("Error"),
                              self.tr("Could not load image: {path}").format(path=image_path))
            return
        self.image_name = image_name
        self.txt_name = os.path.splitext(self.image_name)[0] + ".txt"
        self.load_annotations(image_path)
        if sbox:
             for i, box in enumerate(self.current_boxes):
                 if (box[4] == sbox[4] and calculate_iou(box, sbox) > 0.5 and
                     box[4] >= self.current_threshold):
                     self.image_viewer.selected_box_idx = i
                     break
        self.image_viewer.set_image(image)
        self.image_viewer.update_display()
        self.update_status()

    def load_annotations(self, image_path):
        image_name = os.path.basename(image_path)
        txt_name = os.path.splitext(image_name)[0] + ".txt"
        txt_path = os.path.join(self.images_dir, txt_name)
        if os.path.exists(txt_path):
            self.current_boxes = self.read_yolo_annotations(txt_path, image_path)
        else:
            self.create_annotations_file(image_path)
            self.current_boxes = self.read_yolo_annotations(txt_path, image_path)

    def create_annotations_file(self, image_path):
        print(image_path)
        image = cv2.imread(image_path)
        if image is None:
            return
        results = self.model.predict(image, conf=0.001)
        image_name = os.path.basename(image_path)
        txt_name = os.path.splitext(image_name)[0] + ".txt"
        txt_path = os.path.join(self.images_dir, txt_name)
        with open(txt_path, 'w') as f:
            if results and len(results) > 0:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    img_h, img_w, _ = image.shape
                    x_center = (x1 + x2) / 2 / img_w
                    y_center = (y1 + y2) / 2 / img_h
                    width = (x2 - x1) / img_w
                    height = (y2 - y1) / img_h
                    f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {conf:.6f}\n")
        item = self.image_list.item(self.current_idx)
        item.setBackground(QColor(255, 255, 144))

    def read_yolo_annotations(self, txt_path, image_path):
        boxes = []
        image = cv2.imread(image_path)
        if image is None:
            return boxes
        img_h, img_w, _ = image.shape
        try:
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        conf = float(parts[5]) if len(parts) > 5 else 0.99
                        x1 = int((x_center - width/2) * img_w)
                        y1 = int((y_center - height/2) * img_h)
                        x2 = int((x_center + width/2) * img_w)
                        y2 = int((y_center + height/2) * img_h)
                        boxes.append((x1, y1, x2, y2, cls, conf))
        except Exception as e:
            print(self.tr("Error reading annotation file {path}: {error}").format(path=txt_path, error=e))
        return boxes

    def save_annotations_to_file(self):
        if not hasattr(self, 'image_name') or not self.image_paths:
            return
        image_path = self.image_paths[self.current_idx]
        image = cv2.imread(image_path)
        if image is None:
            return
        img_h, img_w, _ = image.shape
        txt_name = os.path.splitext(self.image_name)[0] + ".txt"
        txt_path = os.path.join(self.images_dir, txt_name)
        with open(txt_path, 'w') as f:
            for box in self.current_boxes:
                x1, y1, x2, y2, cls, conf = box
                x_center = (x1 + x2) / 2 / img_w
                y_center = (y1 + y2) / 2 / img_h
                width = (x2 - x1) / img_w
                height = (y2 - y1) / img_h
                f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {conf:.6f}\n")
        item = self.image_list.item(self.current_idx)
        item.setBackground(QColor(255, 255, 144))

    def update_threshold(self, value):
        self.current_threshold = value / 100.0
        self.threshold_value.setText(f"{self.current_threshold:.2f}")
        self.image_viewer.update_display()
        self.update_status()

    def update_status(self):
        if self.image_paths:
            visible_boxes = [box for box in self.current_boxes
                           if box[5] >= self.current_threshold
                           and box[4] in self.class_names]
            if 0 <= self.image_viewer.selected_box_idx < len(self.current_boxes):
                box = self.current_boxes[self.image_viewer.selected_box_idx]
                x1, y1, x2, y2, cls, conf = box
                class_name = self.class_names.get(cls, self.tr("Unknown"))
                selected_info = self.tr("Box [{x1},{y1},{x2},{y2}] Class: {cls} ({cls_name}) Conf: {conf:.2f}").format(
                    x1=x1, y1=y1, x2=x2, y2=y2, cls=cls, cls_name=class_name, conf=conf)
            else:
                selected_info = self.tr("None")
            status = self.tr("Image: {current}/{total} | Boxes: {visible} (total: {total_boxes}) | Selected: {selected_info}").format(
                current=self.current_idx + 1, total=len(self.image_paths),
                visible=len(visible_boxes), total_boxes=len(self.current_boxes),
                selected_info=selected_info)
            self.status_bar.showMessage(status)
        else:
            self.status_bar.showMessage(self.tr("No images"))

    def clip_box_to_image_bounds(self, box):
        if hasattr(self.image_viewer, 'image') and self.image_viewer.image is not None:
            img_h, img_w, _ = self.image_viewer.image.shape
            box[0] = max(0, min(box[0], img_w))
            box[1] = max(0, min(box[1], img_h))
            box[2] = max(0, min(box[2], img_w))
            box[3] = max(0, min(box[3], img_h))

    def save_annotations(self, destination="train"):
        if not hasattr(self, 'image_name') or not self.image_paths:
            return
        dest_dir = self.train_dir
        src_path = self.image_paths[self.current_idx]
        dst_path = os.path.join(dest_dir, self.image_name)
        shutil.copy(src_path, dst_path)
        src_txt_path = os.path.join(self.images_dir, self.txt_name)
        dst_txt_path = os.path.join(dest_dir, self.txt_name)
        image_path = self.image_paths[self.current_idx]
        image = cv2.imread(image_path)
        if image is not None:
            img_h, img_w, _ = image.shape
            with open(dst_txt_path, 'w') as f:
                for box in self.current_boxes:
                    if box[5] >= self.current_threshold and box[4] in self.class_names:
                        x1, y1, x2, y2, cls, conf = box
                        x_center = (x1 + x2) / 2 / img_w
                        y_center = (y1 + y2) / 2 / img_h
                        width = (x2 - x1) / img_w
                        height = (y2 - y1) / img_h
                        f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        self.train_images.add(self.image_name)
        item = self.image_list.item(self.current_idx)
        item.setBackground(QColor(144, 238, 144))

    def closeEvent(self, event):
        self.save_config()
        event.accept()

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        if key == Qt.Key_Home: self.first_image()
        elif key == Qt.Key_End: self.last_image()
        elif key == Qt.Key_PageDown: self.prev_10_images()
        elif key == Qt.Key_PageUp: self.next_10_images()
        elif key == Qt.Key_Z: self.random_image()
        elif key == Qt.Key_N:
            self.save_annotations("train")
            self.next_image()
        elif key == Qt.Key_R: self.reset_annotations()
        elif key == Qt.Key_Left and not (event.modifiers() & Qt.ControlModifier):
            self.prev_image()
        elif key == Qt.Key_Right and not (event.modifiers() & Qt.ControlModifier):
            self.next_image()
        elif (key == Qt.Key_Delete and
              0 <= self.image_viewer.selected_box_idx < len(self.current_boxes)):
            del self.current_boxes[self.image_viewer.selected_box_idx]
            self.image_viewer.selected_box_idx = -1
            self.image_viewer.update_display()
            self.update_status()
            self.save_annotations_to_file()
        elif (Qt.Key_0 <= key <= Qt.Key_9 and
              0 <= self.image_viewer.selected_box_idx < len(self.current_boxes)):
            new_class = key - Qt.Key_0
            box = list(self.current_boxes[self.image_viewer.selected_box_idx])
            box[4] = new_class; box[5] = 0.99
            self.selected_class_id = new_class
            self.current_boxes[self.image_viewer.selected_box_idx] = tuple(box)
            self.image_viewer.update_display()
            self.save_annotations_to_file()
            self.update_status()
        elif (key == Qt.Key_Up and event.modifiers() & Qt.ControlModifier and
              0 <= self.image_viewer.selected_box_idx < len(self.current_boxes)):
            self.resize_box(0, -5, 0, 5)
        elif (key == Qt.Key_Down and event.modifiers() & Qt.ControlModifier and
              0 <= self.image_viewer.selected_box_idx < len(self.current_boxes)):
            self.resize_box(0, 5, 0, -5)
        elif (key == Qt.Key_Left and event.modifiers() & Qt.ControlModifier and
              0 <= self.image_viewer.selected_box_idx < len(self.current_boxes)):
            self.resize_box(5, 0, -5, 0)
        elif (key == Qt.Key_Right and event.modifiers() & Qt.ControlModifier and
              0 <= self.image_viewer.selected_box_idx < len(self.current_boxes)):
            self.resize_box(-5, 0, 5, 0)
        elif key == Qt.Key_Space:
            self.image_viewer.show_annotations = False
            self.image_viewer.update_display()
        elif key == Qt.Key_H: self.toggle_annotations()
        elif key == Qt.Key_V: self.reset_view()
        elif key == Qt.Key_C: self.toggle_class_display()
        else: super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Space:
            self.image_viewer.show_annotations = True
            self.image_viewer.update_display()
        super().keyReleaseEvent(event)

    def resize_box(self, dx1=0, dy1=0, dx2=0, dy2=0):
        if 0 <= self.image_viewer.selected_box_idx < len(self.current_boxes):
            box = list(self.current_boxes[self.image_viewer.selected_box_idx])
            box[0] += dx1; box[1] += dy1; box[2] += dx2; box[3] += dy2
            self.clip_box_to_image_bounds(box)
            self.current_boxes[self.image_viewer.selected_box_idx] = tuple(box)
            self.image_viewer.update_display()
            self.save_annotations_to_file()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = YOLOLabeler()
    window.show()
    sys.exit(app.exec_())