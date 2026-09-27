"""
Dataset Manager Module
Handles sample storage, dataset loading, and sample generation for air-written characters.
Saves both raw/normalized stroke coordinates (JSON) and rendered images (PNG)
in data/dataset/<label>/ per Prompt Section 18.
"""

import os
import json
import glob
from datetime import datetime
import numpy as np
import cv2

from .feature_extractor import StrokeFeatureExtractor


class DatasetManager:
    def __init__(self, data_root: str = None):
        if data_root is None:
            # air_writing/data/dataset
            data_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "dataset")
        self.data_root = data_root
        os.makedirs(self.data_root, exist_ok=True)
        self.extractor = StrokeFeatureExtractor(target_size=128, margin=14, num_resample_points=64)

    def get_class_dir(self, label: str) -> str:
        # On Windows, filenames are case-insensitive, so 'a' and 'A' collide in the same folder!
        # To strictly support both A-Z and a-z reliably on Windows:
        # Store as 'upper_A', 'lower_a', etc.
        if label.isupper():
            folder_name = f"upper_{label}"
        elif label.islower():
            folder_name = f"lower_{label}"
        else:
            folder_name = f"char_{label}"

        class_dir = os.path.join(self.data_root, folder_name)
        os.makedirs(class_dir, exist_ok=True)
        return class_dir

    def save_sample(self, label: str, raw_strokes: list) -> str:
        """
        Saves character sample:
        1. JSON with raw coordinates, normalized coordinates, and metadata
        2. PNG with rendered 128x128 canonical thumbnail
        """
        if not label or not raw_strokes:
            return ""

        class_dir = self.get_class_dir(label)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        base_name = f"sample_{timestamp}"

        norm_strokes = self.extractor.normalize_strokes(raw_strokes)
        rendered_img = self.extractor.render_normalized_image(norm_strokes)

        # 1. Save JSON
        json_path = os.path.join(class_dir, f"{base_name}.json")
        sample_data = {
            "label": label,
            "timestamp": timestamp,
            "raw_strokes": raw_strokes,
            "normalized_strokes": norm_strokes,
            "stroke_count": len(raw_strokes)
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(sample_data, f, indent=2)

        # 2. Save PNG
        png_path = os.path.join(class_dir, f"{base_name}.png")
        cv2.imwrite(png_path, rendered_img)

        return json_path

    def get_dataset_stats(self) -> dict:
        """Returns sample count per character class."""
        stats = {}
        for folder in os.listdir(self.data_root):
            folder_path = os.path.join(self.data_root, folder)
            if os.path.isdir(folder_path):
                json_files = glob.glob(os.path.join(folder_path, "*.json"))
                stats[folder] = len(json_files)
        return stats

    def load_dataset(self) -> tuple:
        """
        Loads all dataset samples from disk and extracts feature vectors.
        Returns:
            X: numpy array of feature vectors (N, feature_dim)
            y: list of string labels (N,)
        """
        X = []
        y = []

        json_pattern = os.path.join(self.data_root, "*", "*.json")
        for j_path in glob.glob(json_pattern):
            try:
                with open(j_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                raw_strokes = data.get("raw_strokes", [])
                label = data.get("label", "")
                if not raw_strokes or not label:
                    continue

                feat = self.extractor.extract_feature_vector(raw_strokes)
                X.append(feat)
                y.append(label)
            except Exception as e:
                print(f"[DatasetManager] Error reading {j_path}: {e}")

        if not X:
            return np.empty((0, 322), dtype=np.float32), []

        return np.array(X, dtype=np.float32), y
