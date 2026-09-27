"""
Canvas Manager Module
Maintains drawing and mask layers, manages stroke histories for lossless undo/clear,
applies velocity-adaptive EMA coordinate smoothing with outlier rejection,
and provides raw stroke coordinate structures for letter segmentation and ML.
"""

import os
import time
from datetime import datetime
import numpy as np
import cv2

# Vibrant BGR Color Palette
COLOR_PALETTE = {
    "Cyan": (255, 230, 0),
    "Magenta": (255, 0, 230),
    "Lime": (50, 255, 50),
    "Yellow": (0, 235, 255),
    "Orange": (0, 140, 255),
    "White": (255, 255, 255),
}


class CanvasManager:
    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height

        # Drawing color layer and binary alpha mask (0 = transparent, 255 = drawn ink)
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.mask = np.zeros((height, width), dtype=np.uint8)

        # Structured stroke history for Undo & ML representation
        # Each stroke: {'type': 'draw', 'points': [(x, y), ...], 'color': (b,g,r), 'thickness': int}
        #           or {'type': 'erase', 'points': [(x, y, r), ...]}
        self.strokes_history = []
        self.current_stroke = None

        # Current tool state
        self.palette = COLOR_PALETTE
        self.color_name = "Cyan"
        self.color = self.palette[self.color_name]
        self.brush_size = 7
        self.eraser_radius = 35
        self.is_eraser_tool = False

        # Smoothing state
        self.smooth_pt = None
        self.last_draw_time = 0

        # Snapshot directory
        self.snapshots_dir = os.path.join(os.path.dirname(__file__), "snapshots")
        os.makedirs(self.snapshots_dir, exist_ok=True)

    def set_color(self, name: str):
        if name in self.palette:
            self.color_name = name
            self.color = self.palette[name]
            self.is_eraser_tool = False

    def toggle_eraser_tool(self):
        self.is_eraser_tool = not self.is_eraser_tool

    def set_brush_size(self, size: int):
        self.brush_size = max(2, min(40, size))

    def smooth_point(self, raw_pt: tuple, max_jump: float = 160.0) -> tuple:
        """
        Velocity-adaptive EMA smoothing with outlier rejection:
        - At high speeds / sharp turns: high alpha (0.85) to preserve letter shapes & corners.
        - At slow speeds: low alpha (0.40) to eliminate hand tremor.
        - Single-frame jumps > max_jump: resets stroke anchor to prevent stray lines.
        """
        if raw_pt is None:
            self.smooth_pt = None
            return None

        if self.smooth_pt is None:
            self.smooth_pt = raw_pt
            return raw_pt

        dx = raw_pt[0] - self.smooth_pt[0]
        dy = raw_pt[1] - self.smooth_pt[1]
        dist = float(np.hypot(dx, dy))

        # Reject outlier jump
        if dist > max_jump:
            self.smooth_pt = raw_pt
            return raw_pt

        # Adaptive alpha based on velocity
        alpha = float(np.clip(0.40 + (dist / 75.0) * 0.45, 0.40, 0.88))

        sx = int(alpha * raw_pt[0] + (1 - alpha) * self.smooth_pt[0])
        sy = int(alpha * raw_pt[1] + (1 - alpha) * self.smooth_pt[1])
        self.smooth_pt = (sx, sy)
        return self.smooth_pt

    def start_stroke(self):
        """Prepares for a new continuous air stroke."""
        self.current_stroke = None
        self.smooth_pt = None

    def draw_to(self, pt: tuple):
        """Draws anti-aliased line with smooth round caps onto canvas and mask."""
        if pt is None:
            return

        if self.smooth_pt is None:
            self.smooth_pt = pt
            if not self.is_eraser_tool:
                self.current_stroke = {
                    'type': 'draw',
                    'points': [pt],
                    'color': self.color,
                    'thickness': self.brush_size
                }
            else:
                self.current_stroke = {
                    'type': 'erase',
                    'points': [(pt[0], pt[1], self.eraser_radius)]
                }
            return

        p1 = self.smooth_pt
        p2 = pt

        if self.is_eraser_tool:
            # Mask-based erasure: zeroes mask and color layer
            cv2.circle(self.canvas, p2, self.eraser_radius, (0, 0, 0), -1)
            cv2.circle(self.mask, p2, self.eraser_radius, 0, -1)
            if self.current_stroke is None or self.current_stroke['type'] != 'erase':
                self.current_stroke = {'type': 'erase', 'points': []}
            self.current_stroke['points'].append((p2[0], p2[1], self.eraser_radius))
        else:
            # Draw on color canvas and mask
            cv2.line(self.canvas, p1, p2, self.color, self.brush_size, cv2.LINE_AA)
            cv2.circle(self.canvas, p2, self.brush_size // 2, self.color, -1, cv2.LINE_AA)
            cv2.line(self.mask, p1, p2, 255, self.brush_size, cv2.LINE_AA)
            cv2.circle(self.mask, p2, self.brush_size // 2, 255, -1, cv2.LINE_AA)

            if self.current_stroke is None or self.current_stroke['type'] != 'draw':
                self.current_stroke = {
                    'type': 'draw',
                    'points': [p1],
                    'color': self.color,
                    'thickness': self.brush_size
                }
            self.current_stroke['points'].append(p2)

        self.smooth_pt = p2
        self.last_draw_time = time.time()

    def erase_at(self, pt: tuple, radius: int = None):
        """Erases directly around a given coordinate via mask layer."""
        if pt is None:
            return
        r = radius or self.eraser_radius
        cv2.circle(self.canvas, pt, r, (0, 0, 0), -1)
        cv2.circle(self.mask, pt, r, 0, -1)

        if self.current_stroke is None or self.current_stroke['type'] != 'erase':
            self.current_stroke = {'type': 'erase', 'points': []}
        self.current_stroke['points'].append((pt[0], pt[1], r))

        self.smooth_pt = pt
        self.last_draw_time = time.time()

    def end_stroke(self):
        """Finalizes the active stroke and commits to history. Returns the committed stroke or None."""
        committed = None
        if self.current_stroke is not None:
            if len(self.current_stroke.get('points', [])) > 0:
                self.strokes_history.append(self.current_stroke)
                committed = self.current_stroke
            self.current_stroke = None
        self.smooth_pt = None
        return committed

    def undo(self) -> bool:
        """Removes the last stroke and cleanly reconstructs the canvas layers."""
        if not self.strokes_history:
            return False
        self.strokes_history.pop()
        self._rebuild_canvas()
        return True

    def clear(self):
        """Clears all strokes and resets canvas layers."""
        self.strokes_history.clear()
        self.current_stroke = None
        self.smooth_pt = None
        self.canvas.fill(0)
        self.mask.fill(0)

    def has_content(self) -> bool:
        return bool(len(self.strokes_history) > 0 or np.any(self.mask > 0))

    def get_raw_strokes(self) -> list:
        """
        Returns list of drawn strokes as list of coordinate sequences:
        [ [(x1,y1), (x2,y2), ...], [(x1,y1), ...] ]
        """
        strokes = []
        for s in self.strokes_history:
            if s.get('type') == 'draw' and len(s.get('points', [])) > 1:
                strokes.append(list(s['points']))
        return strokes

    def _rebuild_canvas(self):
        """Losslessly redraws canvas and mask layers from stroke history."""
        self.canvas.fill(0)
        self.mask.fill(0)

        for stroke in self.strokes_history:
            stype = stroke.get('type')
            points = stroke.get('points', [])
            if not points:
                continue

            if stype == 'draw':
                col = stroke.get('color', (255, 255, 255))
                th = stroke.get('thickness', 6)
                for p1, p2 in zip(points[:-1], points[1:]):
                    cv2.line(self.canvas, p1, p2, col, th, cv2.LINE_AA)
                    cv2.circle(self.canvas, p2, th // 2, col, -1, cv2.LINE_AA)
                    cv2.line(self.mask, p1, p2, 255, th, cv2.LINE_AA)
                    cv2.circle(self.mask, p2, th // 2, 255, -1, cv2.LINE_AA)
            elif stype == 'erase':
                for item in points:
                    ex, ey, er = item
                    cv2.circle(self.canvas, (ex, ey), er, (0, 0, 0), -1)
                    cv2.circle(self.mask, (ex, ey), er, 0, -1)

    def save_snapshot(self) -> str:
        """Saves clean PNG export using mask layer without black eraser artifacts."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"air_writing_{timestamp}.png"
        filepath = os.path.join(self.snapshots_dir, filename)

        inv_mask = cv2.bitwise_not(self.mask)
        white_bg = np.full_like(self.canvas, 255)
        bg_part = cv2.bitwise_and(white_bg, white_bg, mask=inv_mask)
        fg_part = cv2.bitwise_and(self.canvas, self.canvas, mask=self.mask)
        export_img = cv2.add(bg_part, fg_part)

        cv2.imwrite(filepath, export_img)
        return filepath

    def composite(self, frame: np.ndarray, mode: str = "overlaid") -> np.ndarray:
        h, w, _ = frame.shape
        if (h, w) != (self.height, self.width):
            c_layer = cv2.resize(self.canvas, (w, h))
            m_layer = cv2.resize(self.mask, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            c_layer = self.canvas
            m_layer = self.mask

        if mode == "overlaid":
            inv_mask = cv2.bitwise_not(m_layer)
            frame_bg = cv2.bitwise_and(frame, frame, mask=inv_mask)
            canvas_fg = cv2.bitwise_and(c_layer, c_layer, mask=m_layer)
            return cv2.add(frame_bg, canvas_fg)

        elif mode == "side_by_side":
            split_frame = cv2.resize(frame, (w // 2, h))
            split_canvas = cv2.resize(c_layer, (w // 2, h))
            cv2.line(split_canvas, (0, 0), (0, h), (70, 70, 70), 2)
            return np.hstack([split_frame, split_canvas])

        elif mode == "canvas_only":
            return c_layer.copy()

        return frame
