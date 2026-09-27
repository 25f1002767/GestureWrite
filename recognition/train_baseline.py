"""
recognition/train_baseline.py
Generates comprehensive synthetic stroke trajectories for uppercase English alphabet (A-Z)
with rich data augmentations (scale, tilt, aspect ratio, jitter, speed variation).
Trains a calibrated RBF SVM classifier and saves it to recognition/model/character_svm.joblib.
Also registers samples into data/dataset/ via DatasetManager.
"""

import os
import sys
import numpy as np
from recognition.feature_extractor import StrokeFeatureExtractor
from recognition.dataset_manager import DatasetManager
from recognition.recognizer import CharacterRecognizer, DEFAULT_MODEL_PATH


def get_canonical_templates():
    """
    Defines canonical geometric trajectories in a 0-100 coordinate box.
    Supports both single-stroke and multi-stroke representations.
    """
    templates = {}

    # A: Left diagonal, right diagonal, crossbar
    templates['A'] = [
        # Multi-stroke
        [[(50, 15), (20, 85)], [(50, 15), (80, 85)], [(32, 60), (68, 60)]],
        # Single-stroke continuous
        [[(20, 85), (50, 15), (80, 85), (68, 55), (32, 55)]]
    ]

    # B: Vertical stem + two circular lobes
    b_top = [(int(25 + 30 * np.cos(a)), int(32 + 18 * np.sin(a))) for a in np.linspace(-np.pi/2, np.pi/2, 12)]
    b_bot = [(int(25 + 32 * np.cos(a)), int(68 + 18 * np.sin(a))) for a in np.linspace(-np.pi/2, np.pi/2, 12)]
    templates['B'] = [
        [[(25, 15), (25, 85)], b_top, b_bot],
        [[(25, 85), (25, 15)] + b_top + b_bot]
    ]

    # C: Smooth open arc
    c_arc = [(int(55 + 32 * np.cos(a)), int(50 + 35 * np.sin(a))) for a in np.linspace(np.pi/4, 7*np.pi/4, 25)]
    templates['C'] = [[c_arc[::-1]]]

    # D: Vertical stem + large arc
    d_arc = [(int(25 + 40 * np.cos(a)), int(50 + 35 * np.sin(a))) for a in np.linspace(-np.pi/2, np.pi/2, 22)]
    templates['D'] = [
        [[(25, 15), (25, 85)], d_arc],
        [[(25, 85), (25, 15)] + d_arc + [(25, 85)]]
    ]

    # E: Vertical stem + 3 horizontal bars
    templates['E'] = [
        [[(25, 15), (25, 85)], [(25, 15), (75, 15)], [(25, 50), (65, 50)], [(25, 85), (75, 85)]],
        [[(75, 15), (25, 15), (25, 50), (65, 50), (25, 50), (25, 85), (75, 85)]]
    ]

    # F: Vertical stem + 2 horizontal bars
    templates['F'] = [
        [[(25, 15), (25, 85)], [(25, 15), (75, 15)], [(25, 50), (65, 50)]]
    ]

    # G: Arc + horizontal inward spur
    g_arc = [(int(55 + 32 * np.cos(a)), int(50 + 35 * np.sin(a))) for a in np.linspace(np.pi/4, 7*np.pi/4, 22)][::-1]
    templates['G'] = [
        [g_arc + [(55, 65), (75, 65), (75, 80)]],
        [g_arc, [(75, 80), (75, 55), (55, 55)]]
    ]

    # H: Two vertical legs + center crossbar
    templates['H'] = [
        [[(20, 15), (20, 85)], [(80, 15), (80, 85)], [(20, 50), (80, 50)]],
        [[(20, 15), (20, 85), (20, 50), (80, 50), (80, 15), (80, 85)]]
    ]

    # I: Vertical center stroke (optional top/bottom caps)
    templates['I'] = [
        [[(50, 15), (50, 85)]],
        [[(25, 15), (75, 15)], [(50, 15), (50, 85)], [(25, 85), (75, 85)]]
    ]

    # J: Downward vertical with left curve at bottom
    j_curve = [(55, 15), (55, 65)] + [(int(35 + 20 * np.cos(a)), int(65 + 18 * np.sin(a))) for a in np.linspace(0, np.pi, 12)]
    templates['J'] = [
        [j_curve],
        [[(30, 15), (80, 15)], j_curve]
    ]

    # K: Vertical stem + upper diagonal + lower diagonal
    templates['K'] = [
        [[(25, 15), (25, 85)], [(80, 15), (25, 50)], [(25, 50), (80, 85)]],
        [[(25, 15), (25, 85)], [(80, 15), (30, 50), (80, 85)]]
    ]

    # L: Vertical stroke down + base horizontal right
    templates['L'] = [
        [[(25, 15), (25, 85), (75, 85)]],
        [[(25, 15), (25, 85)], [(25, 85), (75, 85)]]
    ]

    # M: Up, diagonal down to center, diagonal up, down
    templates['M'] = [
        # Continuous
        [[(15, 85), (15, 15), (50, 60), (85, 15), (85, 85)]],
        # Down-up variation
        [[(15, 15), (15, 85), (50, 45), (85, 85), (85, 15)]],
        # Multi-stroke
        [[(15, 85), (15, 15)], [(15, 15), (50, 60), (85, 15)], [(85, 15), (85, 85)]]
    ]

    # N: Up, diagonal down, up
    templates['N'] = [
        [[(20, 85), (20, 15), (80, 85), (80, 15)]],
        [[(20, 85), (20, 15)], [(20, 15), (80, 85)], [(80, 85), (80, 15)]]
    ]

    # O: Elliptical closed loop
    o_loop = [(int(50 + 33 * np.cos(a)), int(50 + 38 * np.sin(a))) for a in np.linspace(0, 2*np.pi, 28)]
    templates['O'] = [[o_loop], [o_loop[::-1]]]

    # P: Vertical stem + top loop
    p_loop = [(int(25 + 32 * np.cos(a)), int(35 + 20 * np.sin(a))) for a in np.linspace(-np.pi/2, np.pi/2, 14)]
    templates['P'] = [
        [[(25, 85), (25, 15)] + p_loop + [(25, 55)]],
        [[(25, 15), (25, 85)], p_loop + [(25, 55)]]
    ]

    # Q: Loop + bottom right diagonal tail
    templates['Q'] = [
        [o_loop, [(55, 65), (85, 90)]],
        [o_loop + [(55, 65), (85, 90)]]
    ]

    # R: Stem + top loop + diagonal leg
    r_loop = [(int(25 + 32 * np.cos(a)), int(35 + 20 * np.sin(a))) for a in np.linspace(-np.pi/2, np.pi/2, 14)]
    templates['R'] = [
        [[(25, 15), (25, 85)], r_loop + [(25, 55)], [(25, 55), (80, 85)]],
        [[(25, 85), (25, 15)] + r_loop + [(25, 55), (80, 85)]]
    ]

    # S: Snake curve (top lobe + bottom lobe)
    s_top = [(int(50 + 26 * np.cos(a)), int(34 + 18 * np.sin(a))) for a in np.linspace(0, np.pi, 14)]
    s_bot = [(int(50 - 28 * np.cos(a)), int(68 + 19 * np.sin(a))) for a in np.linspace(0, np.pi, 14)]
    templates['S'] = [
        [s_top[::-1] + s_bot]
    ]

    # T: Top horizontal bar + center vertical stem
    templates['T'] = [
        [[(15, 15), (85, 15)], [(50, 15), (50, 85)]],
        [[(50, 85), (50, 15), (15, 15), (85, 15)]]
    ]

    # U: Down, bottom arc, up
    u_curve = [(25, 15), (25, 65)] + [(int(50 - 25 * np.cos(a)), int(65 + 20 * np.sin(a))) for a in np.linspace(0, np.pi, 14)] + [(75, 65), (75, 15)]
    templates['U'] = [
        [u_curve],
        [u_curve[::-1]]
    ]

    # V: Diagonal down to center, diagonal up
    templates['V'] = [
        [[(20, 15), (50, 85), (80, 15)]],
        [[(20, 15), (50, 85)], [(50, 85), (80, 15)]]
    ]

    # W: Down, up, down, up
    templates['W'] = [
        [[(15, 15), (32, 85), (50, 35), (68, 85), (85, 15)]],
        [[(15, 15), (35, 85)], [(35, 85), (50, 40)], [(50, 40), (65, 85)], [(65, 85), (85, 15)]]
    ]

    # X: Two crossing diagonals
    templates['X'] = [
        [[(20, 15), (80, 85)], [(80, 15), (20, 85)]],
        [[(20, 15), (80, 85), (50, 50), (80, 15), (20, 85)]]
    ]

    # Y: Diagonal fork + vertical stem
    templates['Y'] = [
        [[(20, 15), (50, 50)], [(80, 15), (50, 50)], [(50, 50), (50, 85)]],
        [[(20, 15), (50, 50), (50, 85)], [(80, 15), (50, 50)]]
    ]

    # Z: Top horizontal, diagonal down-left, base horizontal
    templates['Z'] = [
        [[(20, 15), (80, 15), (20, 85), (80, 85)]]
    ]

    return templates


def augment_stroke_sample(strokes: list) -> list:
    """Applies affine transforms and elastic jitter to simulate handwriting variability."""
    scale_x = np.random.uniform(0.80, 1.20)
    scale_y = np.random.uniform(0.80, 1.20)
    slant = np.random.uniform(-0.15, 0.15)
    angle = np.radians(np.random.uniform(-12, 12))
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    trans_x = np.random.uniform(-5, 5)
    trans_y = np.random.uniform(-5, 5)

    augmented = []
    for stroke in strokes:
        if not stroke:
            continue
        new_stroke = []
        for pt in stroke:
            x, y = float(pt[0]), float(pt[1])
            # Center to (50, 50)
            cx = (x - 50.0) * scale_x
            cy = (y - 50.0) * scale_y

            # Slant
            cx = cx + slant * cy

            # Rotate
            rx = cx * cos_a - cy * sin_a
            ry = cx * sin_a + cy * cos_a

            # Translate back with noise
            nx = rx + 50.0 + trans_x + np.random.normal(0, 1.0)
            ny = ry + 50.0 + trans_y + np.random.normal(0, 1.0)
            new_stroke.append((int(round(nx)), int(round(ny))))
        augmented.append(new_stroke)

    return augmented


def generate_baseline_dataset(samples_per_class=45):
    """Generates synthetic dataset covering all A-Z uppercase letters."""
    templates = get_canonical_templates()
    dataset_mgr = DatasetManager()
    extractor = StrokeFeatureExtractor()

    X = []
    y = []

    print(f"[Dataset] Generating {samples_per_class} augmented samples per character (A-Z)...")

    for label, variations in templates.items():
        count_for_label = 0
        for i in range(samples_per_class):
            base_strokes = variations[i % len(variations)]
            aug_strokes = augment_stroke_sample(base_strokes)

            feat = extractor.extract_feature_vector(aug_strokes)
            X.append(feat)
            y.append(label)
            count_for_label += 1

            # Save first 8 samples of each letter to disk in data/dataset/
            if i < 8:
                dataset_mgr.save_sample(label, aug_strokes)

    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    print(f"[Dataset] Total generated samples: {len(X)} across {len(set(y))} character classes.")
    return X, y


def train_and_save_baseline():
    X, y = generate_baseline_dataset(samples_per_class=50)

    print("[Model] Training RBF Support Vector Machine with probability calibration...")
    recognizer = CharacterRecognizer()

    from sklearn.svm import SVC
    clf = SVC(kernel="rbf", C=12.0, gamma="scale", probability=True)
    clf.fit(X, y)

    recognizer.model = clf
    recognizer.classes_ = list(clf.classes_)
    recognizer.save_model()

    train_score = clf.score(X, y)
    print(f"[Model] Training Accuracy: {train_score * 100:.1f}%")
    print(f"[Model] Classes registered: {', '.join(clf.classes_)}")
    print(f"[Model] Model saved successfully to {recognizer.model_path}!")

    # Verify predictions on test words: MONU and HOME
    extractor = StrokeFeatureExtractor()
    templates = get_canonical_templates()

    print("\n--- Verifying Recognition on Target Words ---")
    for test_word in ["MONU", "HOME", "AIR", "HELLO"]:
        predicted_word = ""
        for ch in test_word:
            if ch in templates:
                sample_strokes = templates[ch][0]
                pred_char, conf, stat = recognizer.predict_strokes(sample_strokes)
                predicted_word += (pred_char or "?")
        print(f"Target: {test_word}  -->  Recognized: {predicted_word}  {'[PASS]' if predicted_word == test_word else '[CHECK]'}")

    return recognizer


if __name__ == "__main__":
    train_and_save_baseline()
