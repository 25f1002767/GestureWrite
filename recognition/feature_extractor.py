"""
Stroke Feature Extractor & Normalizer
Normalizes raw air-drawn strokes to a canonical 128x128 representation,
resamples points equidistantly, extracts geometric features,
and renders canonical images for ML character recognition.
"""

import numpy as np
import cv2


class StrokeFeatureExtractor:
    def __init__(self, target_size: int = 128, margin: int = 14, num_resample_points: int = 64):
        self.target_size = target_size
        self.margin = margin
        self.num_resample_points = num_resample_points

    def normalize_strokes(self, raw_strokes: list) -> list:
        """
        Normalizes a collection of strokes:
        1. Finds bounding box across all strokes
        2. Preserves aspect ratio and scales into (target_size - 2*margin)
        3. Centers character within target_size x target_size
        """
        if not raw_strokes:
            return []

        all_points = []
        for stroke in raw_strokes:
            for pt in stroke:
                all_points.append(pt)

        if not all_points:
            return []

        pts = np.array(all_points, dtype=float)
        min_x, min_y = np.min(pts, axis=0)
        max_x, max_y = np.max(pts, axis=0)

        w = max(1.0, max_x - min_x)
        h = max(1.0, max_y - min_y)

        # Scale factor preserving aspect ratio
        usable_size = float(self.target_size - 2 * self.margin)
        scale = usable_size / max(w, h)

        # Centering offsets
        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0
        target_center = self.target_size / 2.0

        normalized_strokes = []
        for stroke in raw_strokes:
            if not stroke:
                continue
            s_arr = np.array(stroke, dtype=float)
            norm_s = (s_arr - np.array([cx, cy])) * scale + target_center
            normalized_strokes.append(norm_s.tolist())

        return normalized_strokes

    def resample_stroke(self, stroke_pts: list, num_points: int) -> np.ndarray:
        """Resamples a single stroke into equidistant arc-length points."""
        pts = np.array(stroke_pts, dtype=float)
        if len(pts) < 2:
            return np.repeat(pts, num_points, axis=0)

        # Calculate cumulative distances
        diffs = pts[1:] - pts[:-1]
        dists = np.hypot(diffs[:, 0], diffs[:, 1])
        cum_dists = np.insert(np.cumsum(dists), 0, 0.0)
        total_len = cum_dists[-1]

        if total_len == 0:
            return np.repeat(pts[:1], num_points, axis=0)

        target_dists = np.linspace(0, total_len, num_points)
        resampled_x = np.interp(target_dists, cum_dists, pts[:, 0])
        resampled_y = np.interp(target_dists, cum_dists, pts[:, 1])

        return np.column_stack([resampled_x, resampled_y])

    def resample_all_strokes(self, normalized_strokes: list) -> np.ndarray:
        """
        Resamples all strokes in a character to a fixed total of num_resample_points.
        Allocates points proportionally to stroke arc lengths.
        """
        if not normalized_strokes:
            return np.zeros((self.num_resample_points, 2), dtype=float)

        stroke_lengths = []
        for s in normalized_strokes:
            pts = np.array(s, dtype=float)
            if len(pts) < 2:
                stroke_lengths.append(1.0)
            else:
                d = np.sum(np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1])))
                stroke_lengths.append(max(1.0, float(d)))

        total_length = sum(stroke_lengths)
        proportions = [l / total_length for l in stroke_lengths]

        # Allocate points per stroke
        counts = [max(2, int(round(p * self.num_resample_points))) for p in proportions]
        # Adjust sum to exactly num_resample_points
        diff = self.num_resample_points - sum(counts)
        if diff != 0:
            max_idx = int(np.argmax(counts))
            counts[max_idx] = max(2, counts[max_idx] + diff)

        all_resampled = []
        for s, cnt in zip(normalized_strokes, counts):
            res = self.resample_stroke(s, cnt)
            all_resampled.append(res)

        combined = np.vstack(all_resampled)
        if len(combined) > self.num_resample_points:
            combined = combined[:self.num_resample_points]
        elif len(combined) < self.num_resample_points:
            pad = np.repeat(combined[-1:], self.num_resample_points - len(combined), axis=0)
            combined = np.vstack([combined, pad])

        return combined

    def render_normalized_image(self, normalized_strokes: list, line_thickness: int = 4) -> np.ndarray:
        """Renders canonical 128x128 image of normalized strokes."""
        img = np.zeros((self.target_size, self.target_size), dtype=np.uint8)
        for stroke in normalized_strokes:
            if len(stroke) < 2:
                if len(stroke) == 1:
                    cv2.circle(img, (int(stroke[0][0]), int(stroke[0][1])), line_thickness // 2, 255, -1)
                continue
            pts = np.array(stroke, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts], isClosed=False, color=255, thickness=line_thickness, lineType=cv2.LINE_AA)
            for p in stroke:
                cv2.circle(img, (int(p[0]), int(p[1])), line_thickness // 2, 255, -1, cv2.LINE_AA)
        return img

    def extract_feature_vector(self, raw_strokes: list) -> np.ndarray:
        """
        Extracts a comprehensive normalized feature vector:
        1. Normalized resampled coordinates (x, y) [0, 1]
        2. First-order directional derivatives (dx, dy)
        3. Spatial 8x8 density grid occupancy
        4. Aspect ratio and stroke count
        """
        if not raw_strokes:
            return np.zeros(self.num_resample_points * 4 + 64 + 2, dtype=np.float32)

        norm_strokes = self.normalize_strokes(raw_strokes)
        resampled = self.resample_all_strokes(norm_strokes)

        # 1. Coordinates normalized to [0, 1]
        norm_coords = (resampled / float(self.target_size)).flatten()

        # 2. Directional deltas
        diffs = np.diff(resampled, axis=0, prepend=resampled[:1])
        norm_diffs = (diffs / float(self.target_size)).flatten()

        # 3. Spatial 8x8 density grid
        rendered = self.render_normalized_image(norm_strokes, line_thickness=4)
        grid_8x8 = cv2.resize(rendered, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        grid_features = grid_8x8.flatten()

        # 4. Global shape metadata
        num_strokes = float(len(norm_strokes))
        all_pts = np.vstack(norm_strokes) if norm_strokes else np.zeros((1, 2))
        span_x = float(np.max(all_pts[:, 0]) - np.min(all_pts[:, 0])) / float(self.target_size)
        span_y = float(np.max(all_pts[:, 1]) - np.min(all_pts[:, 1])) / float(self.target_size)
        aspect_ratio = span_x / max(0.01, span_y)

        features = np.concatenate([
            norm_coords,
            norm_diffs,
            grid_features,
            [num_strokes, aspect_ratio]
        ]).astype(np.float32)

        return features
