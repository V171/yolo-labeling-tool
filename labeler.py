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
    QColorDialog, QDialog, QButtonGroup, QRadioButton
)
from PyQt5.QtCore import Qt, QPoint, QRect, QPointF
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QKeyEvent, QMouseEvent, QPolygonF, QFontMetrics

# --- Constants for Annotation Modes ---
MODE_BOX = "box"
MODE_POLY = "poly"

# --- Box Class ---
class Box:
    """
    Represents an annotation, either a bounding box or a polygon.
    """
    def __init__(self, mode, cls, conf, data):
        """
        Args:
            mode (str): MODE_BOX or MODE_POLY.
            cls (int): Class ID.
            conf (float): Confidence score.
            data (list): For box: [x1, y1, x2, y2]. For poly: [(x1, y1), (x2, y2), ...].
        """
        self.mode = mode
        self.cls = cls
        self.conf = conf
        if mode == MODE_BOX:
            if len(data) != 4:
                raise ValueError("Box data must have 4 coordinates [x1, y1, x2, y2]")
            # Ensure coordinates are numbers and ordered correctly
            x1, y1, x2, y2 = data
            self.x1 = min(x1, x2)
            self.y1 = min(y1, y2)
            self.x2 = max(x1, x2)
            self.y2 = max(y1, y2)
            self.points = None
        elif mode == MODE_POLY:
            if len(data) < 3:
                 raise ValueError("Polygon data must have at least 3 points [(x1, y1), ...]")
            # Ensure points are tuples/lists of numbers
            self.points = [(float(x),float(y)) for x, y in data]
            self.x1, self.y1, self.x2, self.y2 = None, None, None, None
            # Calculate bounding box for easier selection/handling
            if self.points:
                xs, ys = zip(*self.points)
                self.x1 = min(xs)
                self.y1 = min(ys)
                self.x2 = max(xs)
                self.y2 = max(ys)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def to_yolo_format(self, img_w, img_h):
        """
        Converts box data to YOLO format string.
        Box: "class_id x_center y_center width height [conf]"
        Poly: "class_id x1 y1 x2 y2 ... xn yn [conf]"
        """
        if self.mode == MODE_BOX and self.x1 is not None:
            x_center = ((self.x1 + self.x2) / 2) / img_w
            y_center = ((self.y1 + self.y2) / 2) / img_h
            width = abs(self.x2 - self.x1) / img_w
            height = abs(self.y2 - self.y1) / img_h
            return f"{self.cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {self.conf:.6f}"
        elif self.mode == MODE_POLY and self.points:
            normalized_points = []
            for x, y in self.points:
                x_norm = max(0, min(x / img_w, 1)) # Clip to [0,1]
                y_norm = max(0, min(y / img_h, 1))
                normalized_points.extend([x_norm, y_norm])
            points_str = " ".join([f"{p:.6f}" for p in normalized_points])
            return f"{self.cls} {points_str} {self.conf:.6f}"
        return ""

    def to_yolo_format_no_conf(self, img_w, img_h):
        """Same as to_yolo_format but without confidence (for saving to train)."""
        yolo_str = self.to_yolo_format(img_w, img_h)
        if yolo_str:
            parts = yolo_str.strip().split()
            if len(parts) > 1: # At least class_id and data
                return " ".join(parts[:-1]) # Exclude last part (conf)
        return ""

    def get_bounding_rect(self):
        """Returns (x1, y1, x2, y2) for selection/collision."""
        return self.x1, self.y1, self.x2, self.y2

    def get_points(self):
        """Returns list of points for polygons."""
        return self.points

    def move(self, dx, dy):
        """Moves the box/polygon."""
        if self.mode == MODE_BOX:
            self.x1 += dx
            self.y1 += dy
            self.x2 += dx
            self.y2 += dy
        elif self.mode == MODE_POLY and self.points:
            self.points = [(x + dx, y + dy) for x, y in self.points]
            # Update bounding box
            if self.points:
                xs, ys = zip(*self.points)
                self.x1 = min(xs)
                self.y1 = min(ys)
                self.x2 = max(xs)
                self.y2 = max(ys)

    def resize_point(self, point_index, dx, dy):
        """Resizes/moves a polygon point."""
        if self.mode == MODE_POLY and self.points and 0 <= point_index < len(self.points):
            px, py = self.points[point_index]
            new_px = px + dx
            new_py = py + dy
            self.points[point_index] = (new_px, new_py)
            # Update bounding box
            if self.points:
                xs, ys = zip(*self.points)
                self.x1 = min(xs)
                self.y1 = min(ys)
                self.x2 = max(xs)
                self.y2 = max(ys)

    def is_point_inside(self, point):
        """Checks if a point is inside the box/polygon."""
        px, py = point.x(), point.y()
        if self.mode == MODE_BOX:
            x1, y1, x2, y2 = self.get_bounding_rect()
            if x1 is None: return False
            return x1 <= px <= x2 and y1 <= py <= y2
        elif self.mode == MODE_POLY and self.points and len(self.points) >= 3:
            # Simple point-in-polygon check (ray casting)
            n = len(self.points)
            inside = False
            p1x, p1y = self.points[0]
            for i in range(1, n + 1):
                p2x, p2y = self.points[i % n]
                if py > min(p1y, p2y):
                    if py <= max(p1y, p2y):
                        if px <= max(p1x, p2x):
                            if p1y != p2y:
                                xinters = (py - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                            if p1x == p2x or px <= xinters:
                                inside = not inside
                p1x, p1y = p2x, p2y
            return inside
        return False

    def __repr__(self):
        if self.mode == MODE_BOX:
            return f"Box(mode={self.mode}, cls={self.cls}, conf={self.conf}, x1={self.x1}, y1={self.y1}, x2={self.x2}, y2={self.y2})"
        else:
            return f"Box(mode={self.mode}, cls={self.cls}, conf={self.conf}, points={self.points})"

def calculate_iou(box1: Box, box2: Box):
    """
    Calculates Intersection over Union (IoU) for two bounding boxes or polygons.
    For boxes: Standard axis-aligned IoU.
    For polygons: Approximation using bounding boxes or cv2.contourArea if available.
    Mixed modes: IoU of bounding boxes.
    """
    x1_1, y1_1, x2_1, y2_1 = box1.get_bounding_rect()
    x1_2, y1_2, x2_2, y2_2 = box2.get_bounding_rect()

    # Handle None values for bounding rects
    if any(coord is None for coord in [x1_1, y1_1, x2_1, y2_1, x1_2, y1_2, x2_2, y2_2]):
        return 0.0

    # --- Bounding Box IoU (used for Box-Box, Poly-Poly approx, or Box-Poly) ---
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area_bbox = (x_right - x_left) * (y_bottom - y_top)
    box1_area_bbox = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area_bbox = (x2_2 - x1_2) * (y2_2 - y1_2)

    # --- Polygon IoU (if both are polygons and OpenCV is available) ---
    # This is a simplified approximation.
    # A more accurate method would involve calculating the intersection polygon.
    if box1.mode == MODE_POLY and box2.mode == MODE_POLY and box1.points and box2.points:
        try:
            # Convert points to numpy arrays suitable for cv2
            pts1 = np.array(box1.points, dtype=np.int32)
            pts2 = np.array(box2.points, dtype=np.int32)

            # Create blank masks (assuming image size is large enough, or get from context)
            # For a more accurate IoU, image dimensions should be known here.
            # As a workaround, we can create a mask based on the union of bounding boxes.
            # This is still an approximation.
            margin = 10
            mask_w = int(max(x2_1, x2_2) - min(x1_1, x1_2)) + 2 * margin
            mask_h = int(max(y2_1, y2_2) - min(y1_1, y1_2)) + 2 * margin
            if mask_w > 0 and mask_h > 0:
                mask1 = np.zeros((mask_h, mask_w), dtype=np.uint8)
                mask2 = np.zeros((mask_h, mask_h), dtype=np.uint8) # Typo fix: mask_h, mask_w
                offset_x = min(x1_1, x1_2) - margin
                offset_y = min(y1_1, y1_2) - margin

                # Offset points
                pts1_offset = pts1 - np.array([offset_x, offset_y], dtype=np.int32)
                pts2_offset = pts2 - np.array([offset_x, offset_y], dtype=np.int32)

                cv2.fillPoly(mask1, [pts1_offset], 1)
                cv2.fillPoly(mask2, [pts2_offset], 1)

                intersection_mask = cv2.bitwise_and(mask1, mask2)
                union_mask = cv2.bitwise_or(mask1, mask2)

                intersection_area_poly = np.sum(intersection_mask)
                union_area_poly = np.sum(union_mask)

                if union_area_poly == 0:
                    return 1.0 if intersection_area_poly > 0 else 0.0
                return intersection_area_poly / union_area_poly

        except Exception as e:
            print(f"Error calculating polygon IoU, falling back to bbox IoU: {e}")
            # Fall back to bbox IoU if polygon calculation fails
            pass

    # --- Fallback to Bounding Box IoU ---
    union_area_bbox = box1_area_bbox + box2_area_bbox - intersection_area_bbox
    if union_area_bbox == 0:
        return 1.0 if intersection_area_bbox > 0 else 0.0

    return intersection_area_bbox / union_area_bbox


class ImageViewer(QLabel):
    """Widget for displaying the image and annotations."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setAlignment(Qt.AlignTop)
        self.setMinimumSize(640, 480)
        self.setMouseTracking(True)
        self.scale_factor = 1.0
        self.offset = QPoint(0, 0)
        self.drag_start = None
        self.selected_box_idx = -1
        self.next_box_idx = -1
        self.dragging = False
        self.drag_box_start = None
        self.drawing_box = False
        self.draw_start = None
        self.draw_end = None
        self.temp_poly_points = [] # Points for new polygon being drawn
        self.resizing = False
        self.resize_corner = -1 # For bbox: 0-3, For poly: point index
        self.corner_size = 8
        self.last_click_pos = None
        self.show_annotations = True
        self.show_class_names = True
        self.original_box = None # Store original Box object for undo

    def set_image(self, image):
        self.orig_image = image.copy()
        self.image = image
        self.update_display()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_display()

    def scale_point(self, point):
        """Converts image coordinates to screen coordinates. Returns (x, y) tuple."""
        if isinstance(point, (tuple, list)):
            x, y = point
        else: # QPoint or QPointF
            x, y = point.x(), point.y()
        scaled_x = int(x * self.scale_factor) + self.offset.x()
        scaled_y = int(y * self.scale_factor) + self.offset.y()
        return (scaled_x, scaled_y) # Return tuple as expected by caller

    def unscale_point(self, point):
        """Converts screen coordinates to image coordinates."""
        if self.scale_factor != 0:
            x = (point.x() - self.offset.x()) / self.scale_factor
            y = (point.y() - self.offset.y()) / self.scale_factor
            return QPointF(x, y)
        return QPointF(0, 0)

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
        metrics = QFontMetrics(painter.font())

        if self.show_annotations:
            # Filter boxes by threshold and class existence
            visible_boxes = [box for box in self.parent.current_boxes
                           if box.conf >= self.parent.current_threshold
                           and box.cls in self.parent.class_names]

            for box in visible_boxes:
                cls, conf = box.cls, box.conf
                color = self.parent.get_class_color(cls)
                pen = QPen(color, 2, Qt.SolidLine)
                painter.setPen(pen)
                #painter.setBrush(Qt.NoBrush)
                fill_color = QColor(color)
                fill_color.setAlpha(30)
                painter.setBrush(fill_color)
                painter.setPen(pen)

                if box.mode == MODE_BOX:
                    x1, y1, x2, y2 = box.get_bounding_rect()
                    if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
                        sx1, sy1 = self.scale_point((x1, y1))
                        sx2, sy2 = self.scale_point((x2, y2))
                        rect = QRect(sx1, sy1, sx2 - sx1, sy2 - sy1)
                        painter.drawRect(rect)

                        if self.parent.current_boxes.index(box) == self.selected_box_idx:
                            pen.setWidth(3)
                            painter.setPen(pen)
                            painter.drawRect(rect)
                            painter.setPen(QPen(color, 2, Qt.SolidLine)) # Reset pen

                            # Draw resize corners for bbox
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
                            #painter.setBrush(Qt.NoBrush)

                elif box.mode == MODE_POLY:
                    points = box.get_points()
                    if points and len(points) > 1: # Need at least 2 points to draw a line/polygon
                        scaled_points = [QPointF(*self.scale_point(p)) for p in points] # Convert to QPointF for QPolygonF
                        polygon = QPolygonF(scaled_points)
                        if len(points) > 2: # Draw filled polygon if 3 or more points
                            painter.drawPolygon(polygon)

                        if self.parent.current_boxes.index(box) == self.selected_box_idx:
                             pen.setWidth(3)
                             painter.setPen(pen)
                             painter.drawPolyline(polygon)
                             if len(points) > 2:
                                 painter.setPen(Qt.NoPen)
                                 #painter.setBrush(fill_color)
                                 painter.drawPolygon(polygon)
                                 #painter.setPen(pen)
                                 #painter.setBrush(Qt.NoBrush)
                             painter.setPen(QPen(color, 2, Qt.SolidLine))

                             # Draw resize/move points for Poly
                             painter.setBrush(Qt.yellow)
                             for p in scaled_points: # p is QPointF
                                 painter.drawEllipse(p, self.corner_size//2, self.corner_size//2)
                             #painter.setBrush(Qt.NoBrush)

                # Draw class name and confidence
                painter.setPen(QPen(Qt.white, 1))
                if self.show_class_names and cls in self.parent.class_names:
                    class_text = self.parent.class_names[cls]
                else:
                    class_text = str(cls)

                # Position text near the annotation (simple approach)
                text_x, text_y = 10, 10 # Default position
                text=f"{class_text}: {conf:.2f}"
                x1, y1, x2, y2 = box.get_bounding_rect()
                text_bb=metrics.boundingRect(text)
                if x1 is not None and y1 is not None:
                    text_x, text_y = self.scale_point(((x1+x2-text_bb.width())/2, min(y1+15+cls*text_bb.height(),y2))) # Now correctly unpacks tuple

                painter.drawText(text_x, text_y, text)

        # Draw temporary polygon during drawing
        if self.drawing_box and self.temp_poly_points:
             if self.parent.annotation_mode == MODE_POLY:
                 painter.setPen(QPen(Qt.cyan, 2, Qt.DashLine))
                 painter.setBrush(Qt.NoBrush)
                 scaled_temp_points = [QPointF(*self.scale_point(p)) for p in self.temp_poly_points] # Convert to QPointF
                 if len(scaled_temp_points) >= 1:
                     # Draw lines between existing points
                     if len(scaled_temp_points) > 1:
                         polygon = QPolygonF(scaled_temp_points)
                         painter.drawPolyline(polygon)
                     # Draw point markers
                     painter.setPen(QPen(Qt.red, 4))
                     for p in scaled_temp_points[:-1]:
                         painter.drawPoint(p)
                     painter.setPen(QPen(Qt.cyan, 2, Qt.DashLine)) # Reset pen

        # Draw temporary bbox during drawing (original logic)
        if (self.drawing_box and self.parent.annotation_mode == MODE_BOX and
            self.draw_start and self.draw_end):
            start_x, start_y = self.scale_point(self.draw_start) # Unpack tuple
            end_x, end_y = self.scale_point(self.draw_end)       # Unpack tuple
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

    def find_closest_point_index(self, point, points, threshold=10):
        """Find the index of the closest point within threshold."""
        min_dist = float('inf')
        closest_idx = -1
        for i, p in enumerate(points):
            # Use QPointF distance for consistency
            p_qpoint = QPointF(p[0], p[1])
            dist = ((point.x() - p_qpoint.x())**2 + (point.y() - p_qpoint.y())**2)**0.5
            if dist < min_dist and dist < threshold:
                min_dist = dist
                closest_idx = i
        return closest_idx

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            orig_pos = self.unscale_point(event.pos())

            if self.drawing_box:
                # Start drawing new annotation
                self.selected_box_idx = -1

                # Initialize drawing for different modes
                if self.parent.annotation_mode == MODE_BOX:
                        self.draw_start = orig_pos
                        self.draw_end = orig_pos
                elif self.parent.annotation_mode == MODE_POLY:
                    if len(self.temp_poly_points)==0:
                        self.temp_poly_points = [orig_pos,orig_pos] # Start polygon with first point
                    else:
                        self.temp_poly_points.append(orig_pos)
                return
            self.last_click_pos = orig_pos
            self.resizing = False
            self.resize_corner = -1

            # Handle resizing/moving existing annotations
            if (self.selected_box_idx >= 0 and
                self.selected_box_idx < len(self.parent.current_boxes)):
                box = self.parent.current_boxes[self.selected_box_idx]

                # --- Resizing/Editing Logic ---
                if box.mode == MODE_BOX:
                    x1, y1, x2, y2 = box.get_bounding_rect()
                    if x1 is not None:
                        corners = [QPointF(x1, y1), QPointF(x2, y1),
                                   QPointF(x1, y2), QPointF(x2, y2)]
                        for i, corner in enumerate(corners):
                            if ((orig_pos.x() - corner.x())**2 +
                                (orig_pos.y() - corner.y())**2)**0.5 < 10:
                                self.resizing = True
                                self.resize_corner = i
                                self.drag_box_start = orig_pos
                                # Deep copy for original state
                                self.original_box = Box(box.mode, box.cls, box.conf, [box.x1, box.y1, box.x2, box.y2])
                                return

                elif box.mode == MODE_POLY:
                     points = box.get_points()
                     if points:
                         closest_idx = self.find_closest_point_index(orig_pos, points, 10)
                         if closest_idx != -1:
                             self.resizing = True
                             self.resize_corner = closest_idx # Index of point to move
                             self.drag_box_start = orig_pos
                             # Deep copy points for original state
                             original_points = [(p[0], p[1]) for p in points]
                             self.original_box = Box(box.mode, box.cls, box.conf, original_points)
                             return

            # Handle selecting existing annotations
            visible_boxes = [box for box in self.parent.current_boxes
                           if box.conf >= self.parent.current_threshold
                           and box.cls in self.parent.class_names]
            clicked_boxes_indices = []
            for i, box in enumerate(self.parent.current_boxes):
                if (box.conf < self.parent.current_threshold or
                    box.cls not in self.parent.class_names):
                    continue

                if box.is_point_inside(orig_pos):
                    clicked_boxes_indices.append(i)

            if clicked_boxes_indices:
                if self.selected_box_idx not in clicked_boxes_indices:
                    self.selected_box_idx = clicked_boxes_indices[0]
                    self.next_box_idx = self.selected_box_idx
                else: 
                    current_idx = clicked_boxes_indices.index(self.selected_box_idx)
                    next_idx = (current_idx + 1) % len(clicked_boxes_indices)
                    self.next_box_idx = clicked_boxes_indices[next_idx]


                self.drag_box_start = orig_pos
                self.dragging = True
                # Deep copy for original state
                if self.parent.current_boxes[self.selected_box_idx].mode == MODE_BOX:
                    b = self.parent.current_boxes[self.selected_box_idx]
                    self.original_box = Box(b.mode, b.cls, b.conf, [b.x1, b.y1, b.x2, b.y2])
                else: # MODE_POLY
                    b = self.parent.current_boxes[self.selected_box_idx]
                    original_points = [(p[0], p[1]) for p in b.points]
                    self.original_box = Box(b.mode, b.cls, b.conf, original_points)
                self.update_display()
                self.parent.update_status()
                return

            # Handle image moving
            self.drag_start = event.pos()
            self.dragging = False

        elif event.button() == Qt.RightButton:
            # Right-click to finish polygon or show context menu
            if (self.drawing_box and self.parent.annotation_mode == MODE_POLY
                and len(self.temp_poly_points) >= 2): # Allow 2 points to close as a line, or >2 for poly
                # Finish polygon drawing
                self.finish_polygon_drawing()
            else:
                self.show_context_menu(event.pos())

    def mouseMoveEvent(self, event):
        orig_pos = self.unscale_point(event.pos())
        if event.buttons() & Qt.LeftButton:
            # Handle drawing new annotations
            if self.drawing_box:
                if self.parent.annotation_mode == MODE_BOX:
                    self.draw_end = orig_pos
                    self.update_display()

            # Handle resizing existing annotations
            elif self.resizing and self.selected_box_idx >= 0 and self.original_box:
                start_pos = self.drag_box_start
                dx = orig_pos.x() - start_pos.x()
                dy = orig_pos.y() - start_pos.y()
                # Modify the current box based on original state
                box = self.parent.current_boxes[self.selected_box_idx]
                #original = self.original_box

                if box.mode == MODE_BOX:
                    # Apply delta to the specific corner or move the whole box
                    if 0 <= self.resize_corner <= 3: # Resizing corner
                        new_x1, new_y1, new_x2, new_y2 = box.x1, box.y1, box.x2, box.y2
                        if self.resize_corner == 0: # Top-left
                            new_x1 += dx; new_y1 += dy
                        elif self.resize_corner == 1: # Top-right
                            new_x2 += dx; new_y1 += dy
                        elif self.resize_corner == 2: # Bottom-left
                            new_x1 += dx; new_y2 += dy
                        elif self.resize_corner == 3: # Bottom-right
                            new_x2 += dx; new_y2 += dy
                        # Ensure correct order
                        box.x1 = min(new_x1, new_x2)
                        box.y1 = min(new_y1, new_y2)
                        box.x2 = max(new_x1, new_x2)
                        box.y2 = max(new_y1, new_y2)
                    else: # Should not happen, but just in case, move
                        box.move(dx, dy) # Use the move method

                elif box.mode == MODE_POLY:
                    if (0 <= self.resize_corner < len(box.points)):
                        # Move the specific point based on original state
                        # box.resize_point handles updating the bbox
                        box.resize_point(self.resize_corner, dx, dy)

                self.drag_box_start = orig_pos
                self.update_display()

            # Handle dragging (moving) existing annotations
            elif self.dragging and self.selected_box_idx >= 0 and self.original_box:
                start_pos = self.drag_box_start
                dx = orig_pos.x() - start_pos.x()
                dy = orig_pos.y() - start_pos.y()
                # Modify the current box based on original state
                box = self.parent.current_boxes[self.selected_box_idx]
                original = self.original_box

                # Use the move method which handles both modes
                box.move(dx, dy)

                self.drag_box_start = orig_pos
                self.update_display()

            # Handle panning the image
            elif self.drag_start is not None:
                diff = event.pos() - self.drag_start
                self.offset += diff
                self.drag_start = event.pos()
                self.update_display()
        elif self.drawing_box:
            if self.parent.annotation_mode == MODE_POLY:
                # Update the last point to follow the cursor
                if len(self.temp_poly_points) > 0:
                   self.temp_poly_points[-1] = orig_pos # Move last point
                self.update_display()

    def clip_current_box_to_image(self):
        if self.selected_box_idx<0 or self.image is None: 
            return
        current_box = self.parent.current_boxes[self.selected_box_idx]        
        img_h, img_w, _ = self.image.shape
        if current_box.mode == MODE_BOX:
            current_box.x1 = max(0, min(current_box.x1, img_w)); current_box.y1 = max(0, min(current_box.y1, img_h))
            current_box.x2 = max(0, min(current_box.x2, img_w)); current_box.y2 = max(0, min(current_box.y2, img_h))
        elif current_box.mode == MODE_POLY:
            for point_index, point in enumerate(current_box.points):
                px, py = point
                px = max(0, min(px, img_w))
                py = max(0, min(py, img_h))
                current_box.points[point_index] = (px, py)


    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Handle finishing drawing new annotations
            if self.drawing_box:
                if self.parent.annotation_mode == MODE_BOX and self.draw_start and self.draw_end:
                    x1 = min(self.draw_start.x(), self.draw_end.x())
                    y1 = min(self.draw_start.y(), self.draw_end.y())
                    x2 = max(self.draw_start.x(), self.draw_end.x())
                    y2 = max(self.draw_start.y(), self.draw_end.y())
                    if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
                        cls = self.parent.get_selected_class()
                        new_box = Box(MODE_BOX, cls, 0.99, [x1, y1, x2, y2])
                        self.parent.current_boxes.append(new_box)
                        self.selected_box_idx = len(self.parent.current_boxes) - 1
                        self.clip_current_box_to_image()
                        self.parent.save_annotations_to_file()
                    self.drawing_box = False

                # For Poly, drawing is handled on mouse press/move/right click
                # No action needed here for Poly as they are finalized elsewhere

            # Handle saving changes after resizing/dragging
            elif (self.resizing or self.dragging) and self.selected_box_idx >= 0 and self.original_box:
                current_box = self.parent.current_boxes[self.selected_box_idx]
                box_changed = False
                self.clip_current_box_to_image()
                if self.original_box.mode == MODE_BOX:
                    if (self.original_box.x1 != current_box.x1 or
                        self.original_box.y1 != current_box.y1 or
                        self.original_box.x2 != current_box.x2 or
                        self.original_box.y2 != current_box.y2):
                        box_changed = True
                elif self.original_box.mode == MODE_POLY:
                    if self.original_box.points != current_box.points:
                        box_changed = True

                if box_changed:
                    # Update confidence after manual edit
                    current_box.conf = 0.99
                    # Clipping is handled inside save_annotations_to_file or resize logic if needed
                    self.parent.save_annotations_to_file()
                else:
                    self.selected_box_idx=self.next_box_idx

            # Reset drawing/dragging states
            self.drag_start = None
            self.drag_box_start = None
            self.draw_start = None
            self.draw_end = None
            # Do not reset temp_poly_points here, it's managed in finish_polygon_drawing and init
            self.original_box = None
            self.resizing = False
            self.dragging = False
            self.update_display()
            self.parent.update_status()

    def finish_polygon_drawing(self):
        """Finalize the polygon drawing process."""
        if len(self.temp_poly_points) >= 2: # Need at least 2 points
            cls = self.parent.get_selected_class()
            # Store points as a list of tuples
            points_list = [(p.x(), p.y()) for p in self.temp_poly_points[:-1]]
            try:
                new_box = Box(MODE_POLY, cls, 0.99, points_list)
                self.parent.current_boxes.append(new_box)
                self.selected_box_idx = len(self.parent.current_boxes) - 1
                self.clip_current_box_to_image()
                self.parent.save_annotations_to_file()
            except ValueError as e:
                print(f"Error creating polygon: {e}")
        self.drawing_box = False
        self.temp_poly_points = []
        self.update_display()
        self.parent.update_status()

    def distance(self, p1, p2):
        if isinstance(p1, (QPoint, QPointF)) and isinstance(p2, (QPoint, QPointF)):
            return ((p1.x() - p2.x()) ** 2 + (p1.y() - p2.y()) ** 2) ** 0.5
        elif isinstance(p1, (tuple, list)) and isinstance(p2, (tuple, list)):
             return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
        return float('inf')

    def show_context_menu(self, pos):
        """Shows context menu for box operations."""
        menu = QMenu(self)
        self.add_box_action = menu.addAction(self.parent.tr("Add Annotation"))
        self.delete_action = menu.addAction(self.parent.tr("Delete Annotation"))
        self.change_class_action = menu.addAction(self.parent.tr("Change Class"))
        # Resize action is less relevant for Poly, but could be adapted
        # self.resize_action = menu.addAction(self.parent.tr("Resize"))
        action = menu.exec_(self.mapToGlobal(pos))
        if action == self.add_box_action:
            self.drawing_box = True
            self.draw_start = None
            self.draw_end = None
            self.temp_poly_points = []
        elif (action == self.delete_action and
              0 <= self.selected_box_idx < len(self.parent.current_boxes)):
            del self.parent.current_boxes[self.selected_box_idx]
            self.selected_box_idx = -1
            self.update_display()
            self.parent.update_status()
            self.parent.save_annotations_to_file()
        elif (action == self.change_class_action and
              0 <= self.selected_box_idx < len(self.parent.current_boxes)):
            new_class, ok = QInputDialog.getInt(
                self, self.parent.tr("Change Class"),
                self.parent.tr("Enter new class:"),
                value=self.parent.current_boxes[self.selected_box_idx].cls,
                min=0, max=100
            )
            if ok:
                self.parent.current_boxes[self.selected_box_idx].cls = new_class
                self.update_display()
                self.parent.save_annotations_to_file()
        # elif action == self.resize_action and 0 <= self.selected_box_idx < len(self.parent.current_boxes):
        #     # Implementation depends on mode
        #     pass

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
        self.current_boxes = [] # List of Box objects
        self.class_names = {}
        self.class_colors = {}
        self.selected_class_id = 0
        self.translations = {} # Stores loaded translations
        self.annotation_mode = MODE_BOX # Default mode
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
        self.btn_first = QPushButton(self.tr("⏮ First (Home)"))
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
        self.btn_last = QPushButton(self.tr("Last (End) ⏭"))
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

        # --- Mode Selection ---
        self.mode_group = QButtonGroup(self)
        self.mode_box_rb = QRadioButton(self.tr("Box"))
        self.mode_box_rb.setFocusPolicy(Qt.NoFocus)
        self.mode_poly_rb = QRadioButton(self.tr("Poly"))
        self.mode_poly_rb.setFocusPolicy(Qt.NoFocus)
        self.mode_group.addButton(self.mode_box_rb, 0)
        self.mode_group.addButton(self.mode_poly_rb, 1)
        self.mode_box_rb.setChecked(True)
        self.mode_group.buttonClicked.connect(self.change_annotation_mode)
        toolbar_layout.addWidget(QLabel(self.tr("Mode:")))
        toolbar_layout.addWidget(self.mode_box_rb)
        toolbar_layout.addWidget(self.mode_poly_rb)
        # --- End Mode Selection ---

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

    def change_annotation_mode(self, button):
        """Handles changing the annotation mode."""
        if button == self.mode_box_rb:
            self.annotation_mode = MODE_BOX
        elif button == self.mode_poly_rb:
            self.annotation_mode = MODE_POLY
        self.image_viewer.selected_box_idx = -1 # Deselect when mode changes
        self.image_viewer.update_display()
        self.update_status()
        self.save_config() # Save mode preference

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
        self.btn_first.setText(self.tr("⏮ First (Home)"))
        self.btn_prev_10.setText(self.tr("⏪ -10 (PgDn)"))
        self.btn_prev.setText(self.tr("◀ Previous (←)"))
        self.btn_next.setText(self.tr("Next (→) ▶"))
        self.btn_next_10.setText(self.tr("+10 (PgUp) ⏩"))
        self.btn_last.setText(self.tr("Last (End) ⏭"))
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
        # Update mode radio buttons
        self.mode_box_rb.setText(self.tr("Box"))
        self.mode_poly_rb.setText(self.tr("Poly"))
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
        if (hasattr(self.image_viewer, 'image') and
            self.image_viewer.image is not None):
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
        if (0 <= self.image_viewer.selected_box_idx < len(self.current_boxes)):
            self.current_boxes[self.image_viewer.selected_box_idx].cls = class_id
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
            "language": "ru",
            "annotation_mode": MODE_BOX # Load mode
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
        self.annotation_mode = config.get("annotation_mode", default_config["annotation_mode"])

    def save_config(self):
        """Saves current settings to config.json."""
        config = {
            "model_path": self.model_path,
            "images_dir": self.images_dir,
            "train_dir": self.train_dir,
            "last_image_index": self.current_idx,
            "selected_class_id": self.selected_class_id,
            "class_names": self.class_names,
            "language": self.language,
            "annotation_mode": self.annotation_mode # Save mode
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
        if self.image_viewer.selected_box_idx >= 0 and self.image_viewer.selected_box_idx < len(self.current_boxes):
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
        # Simple selection logic
        if sbox:
             for i, box in enumerate(self.current_boxes):
                 # Use the new IoU function
                 if (sbox.cls == box.cls and calculate_iou(sbox, box) > 0.5 and
                     box.conf >= self.current_threshold):
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
        print(f"Creating annotations for {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            return
        image_name = os.path.basename(image_path)
        txt_name = os.path.splitext(image_name)[0] + ".txt"
        txt_path = os.path.join(self.images_dir, txt_name)
        try:
            # --- Predict based on mode ---
            # Note: Ultralytics YOLOv8 'detect' task produces boxes.
            # 'segment' task produces masks which can be converted to polygons.
            # There is no standard 'obb' task in YOLOv8 as of my knowledge cutoff.
            # We will use 'detect' for Box mode and 'segment' for Poly mode.
            results = self.model(image, conf=0.001)
            with open(txt_path, 'w') as f:
                if results and len(results) > 0:
                    img_h, img_w, _ = image.shape
                    # --- Handle different result types ---
                    if results[0].obb is not None:
                        for rect,conf,cls in zip(results[0].obb.xyxyxyxy,results[0].obb.conf,results[0].obb.cls):
                            # Create Box object
                            points_list = [(float(p[0]), float(p[1])) for p in rect]
                            poly_obj = Box(MODE_POLY, int(cls), float(conf), points_list)
                            yolo_line = poly_obj.to_yolo_format(img_w, img_h)
                            if yolo_line:
                                f.write(yolo_line + "\n")
                    elif results[0].masks is not None:
                        masks = results[0].masks.xy # Get segmentation polygons (list of np arrays)
                        boxes = results[0].boxes # Get corresponding boxes for class/conf
                        # It's possible that masks.xy and boxes don't align perfectly,
                        # but usually, they should correspond.
                        for i, (mask, box) in enumerate(zip(masks, boxes)):
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            # mask is a numpy array of shape (N, 2) representing the polygon
                            if len(mask) >= 3: # Need at least 3 points for a polygon
                                points_list = [(float(p[0]), float(p[1])) for p in mask]
                                try:
                                    poly_obj = Box(MODE_POLY, cls, conf, points_list)
                                    yolo_line = poly_obj.to_yolo_format(img_w, img_h)
                                    if yolo_line:
                                        f.write(yolo_line + "\n")
                                except ValueError as e:
                                    print(f"Error creating polygon from mask: {e}")
                    elif results[0].boxes is not None:
                        for box in results[0].boxes:
                            xyxy = box.xyxy[0].tolist()
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            # Create Box object
                            box_obj = Box(MODE_BOX, cls, conf, xyxy)
                            yolo_line = box_obj.to_yolo_format(img_w, img_h)
                            if yolo_line:
                                f.write(yolo_line + "\n")
        except Exception as e:
            print(f"Error creating annotations file {txt_path}: {e}")
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
                    if len(parts) < 5: # Need at least class + some data
                        continue
                    try:
                        cls = int(parts[0])
                        if len(parts)%2:
                            data = [float(p) for p in parts[1:]] # All but last (conf)
                            conf = 0.99 # Last element is confidence
                        else:
                            data = [float(p) for p in parts[1:-1]] # All but last (conf)
                            conf = float(parts[-1]) # Last element is confidence

                        if len(data) == 4:
                            # Assume Bbox: x_center, y_center, width, height
                            x_center, y_center, width, height = data
                            x1 = (x_center - width/2) * img_w # Keep as float for Box
                            y1 = (y_center - height/2) * img_h
                            x2 = (x_center + width/2) * img_w
                            y2 = (y_center + height/2) * img_h
                            boxes.append(Box(MODE_BOX, cls, conf, [x1, y1, x2, y2]))

                        elif len(data) >= 6 and len(data) % 2 == 0:
                            # Assume points: x1,y1,x2,y2,... or x1,y1,x2,y2,x3,y3,...
                            points = [(data[i] * img_w, data[i+1] * img_h) for i in range(0, len(data), 2)] # Keep as float
                            # Create Box object, it will determine its mode internally if needed,
                            # but we know it's poly data here.
                            boxes.append(Box(MODE_POLY, cls, conf, points))

                    except (ValueError, IndexError) as ve:
                        print(self.tr("Error parsing annotation line in {path}: {error}").format(path=txt_path, error=ve))
                        continue

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
        try:
            with open(txt_path, 'w') as f:
                for box in self.current_boxes:
                    # Apply threshold and class filter before saving
                    if box.conf >= self.current_threshold and box.cls in self.class_names:
                        yolo_line = box.to_yolo_format(img_w, img_h)
                        if yolo_line:
                            f.write(yolo_line + "\n")

        except Exception as e:
            print(f"Error saving annotations to {txt_path}: {e}")

        item = self.image_list.item(self.current_idx)
        item.setBackground(QColor(255, 255, 144))

    def update_threshold(self, value):
        self.current_threshold = value / 100.0
        self.threshold_value.setText(f"{self.current_threshold:.2f}")
        self.image_viewer.update_display()
        self.update_status()

    def update_status(self):
        if self.image_paths:
            # Filter visible boxes based on mode and threshold
            visible_boxes = [box for box in self.current_boxes
                           if box.conf >= self.current_threshold
                           and box.cls in self.class_names]

            selected_info = self.tr("None")
            if (0 <= self.image_viewer.selected_box_idx < len(self.current_boxes)):
                box = self.current_boxes[self.image_viewer.selected_box_idx]
                class_name = self.class_names.get(box.cls, self.tr("Unknown"))
                if box.mode == MODE_BOX:
                    x1, y1, x2, y2 = box.get_bounding_rect()
                    if x1 is not None:
                        selected_info = self.tr("Box [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}] Class: {cls} ({cls_name}) Conf: {conf:.2f}").format(
                            x1=x1, y1=y1, x2=x2, y2=y2, cls=box.cls, cls_name=class_name, conf=box.conf)
                elif box.mode == MODE_POLY:
                    points = box.get_points()
                    num_points = len(points) if points else 0
                    # Show centroid or first point for position
                    pos_info = ""
                    if points:
                        cx = sum(p[0] for p in points) / len(points)
                        cy = sum(p[1] for p in points) / len(points)
                        pos_info = f"[{cx:.0f},{cy:.0f}] "
                    selected_info = self.tr("Poly ({num_points} pts) {pos}Class: {cls} ({cls_name}) Conf: {conf:.2f}").format(
                        num_points=num_points, pos=pos_info, cls=box.cls, cls_name=class_name, conf=box.conf)

            status = self.tr("Image: {current}/{total} | Boxes: {visible} (total: {total_boxes}) | Selected: {selected_info} | Mode: {mode}").format(
                current=self.current_idx + 1, total=len(self.image_paths),
                visible=len(visible_boxes), total_boxes=len(self.current_boxes),
                selected_info=selected_info, mode=self.annotation_mode.upper())
            self.status_bar.showMessage(status)
        else:
            self.status_bar.showMessage(self.tr("No images"))

    def save_annotations(self, destination="train"):
        if not hasattr(self, 'image_name') or not self.image_paths:
            return
        dest_dir = self.train_dir
        src_path = self.image_paths[self.current_idx]
        dst_path = os.path.join(dest_dir, self.image_name)
        shutil.copy(src_path, dst_path)
        # src_txt_path = os.path.join(self.images_dir, self.txt_name) # Not used directly
        dst_txt_path = os.path.join(dest_dir, self.txt_name)
        image_path = self.image_paths[self.current_idx]
        image = cv2.imread(image_path)
        if image is not None:
            img_h, img_w, _ = image.shape
            try:
                with open(dst_txt_path, 'w') as f:
                    for box in self.current_boxes:
                        # Apply threshold and class filter on save
                        if box.conf >= self.current_threshold and box.cls in self.class_names:
                            # Save without confidence for training datasets
                            yolo_line = box.to_yolo_format_no_conf(img_w, img_h)
                            if yolo_line:
                                f.write(yolo_line + "\n")

            except Exception as e:
                print(f"Error saving annotations to {dst_txt_path}: {e}")

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
            self.current_boxes[self.image_viewer.selected_box_idx].cls = new_class
            self.selected_class_id = new_class
            # Update confidence after class change
            self.current_boxes[self.image_viewer.selected_box_idx].conf = 0.99
            self.image_viewer.update_display()
            self.save_annotations_to_file()
            self.update_status()
        # Example: Ctrl+Up/Down/Left/Right could be used for fine-tuning
        # elif (key == Qt.Key_Up and event.modifiers() & Qt.ControlModifier and
        #       0 <= self.image_viewer.selected_box_idx < len(self.current_boxes)):
        #     self.resize_box(0, -5, 0, 5)
        # ... (other resize logic needs adaptation)
        elif key == Qt.Key_Space:
            self.image_viewer.show_annotations = False
            self.image_viewer.update_display()
        elif key == Qt.Key_H: self.toggle_annotations()
        elif key == Qt.Key_V: self.reset_view()
        elif key == Qt.Key_C: self.toggle_class_display()
        elif key == Qt.Key_B: self.mode_box_rb.setChecked(True); self.change_annotation_mode(self.mode_box_rb)
        elif key == Qt.Key_P: self.mode_poly_rb.setChecked(True); self.change_annotation_mode(self.mode_poly_rb)
        else: super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Space:
            self.image_viewer.show_annotations = True
            self.image_viewer.update_display()
        super().keyReleaseEvent(event)

    # def resize_box(self, dx1=0, dy1=0, dx2=0, dy2=0):
    #     # This needs significant adaptation for Poly
    #     if 0 <= self.image_viewer.selected_box_idx < len(self.current_boxes):
    #         box = self.current_boxes[self.image_viewer.selected_box_idx]
    #         if box.mode == MODE_BOX:
    #             # Correct order handling
    #             new_x1 = min(box.x1 + dx1, box.x2 + dx2)
    #             new_y1 = min(box.y1 + dy1, box.y2 + dy2)
    #             new_x2 = max(box.x1 + dx1, box.x2 + dx2)
    #             new_y2 = max(box.y1 + dy1, box.y2 + dy2)
    #             box.x1, box.y1, box.x2, box.y2 = new_x1, new_y1, new_x2, new_y2
    #             self.image_viewer.update_display()
    #             self.save_annotations_to_file()
    #         elif box.mode == MODE_POLY:
    #             # Logic to move/scale/rotate polygon points
    #             # This is complex and requires more state (center, scale, angle)
    #             pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = YOLOLabeler()

    # Set initial mode from config
    if window.annotation_mode == MODE_BOX:
        window.mode_box_rb.setChecked(True)
    elif window.annotation_mode == MODE_POLY:
        window.mode_poly_rb.setChecked(True)

    window.show()
    sys.exit(app.exec_())
