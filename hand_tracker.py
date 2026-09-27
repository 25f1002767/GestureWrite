"""
Hand Tracker Module (Ultra-Fast & Stable with Hold-to-Switch State Machine)
Tracks 21 hand landmarks using optimized downscaled inference,
3D-aware knuckle distance ratios, and hold-to-switch dwell time.
Default mode is always DRAWING (uninterrupted, fluid handwriting).
Switching to ERASE or SELECT requires holding the gesture steadily.
Automatically reverts to DRAW mode when the special gesture ends.
"""

import os
import urllib.request
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "hand_landmarker.task")


class HandTracker:
    def __init__(self, model_path: str = None, num_hands: int = 1, min_confidence: float = 0.4):
        if model_path is None:
            model_path = MODEL_PATH

        self._ensure_model_exists(model_path)

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=num_hands,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        # Skeleton bone connections
        self.connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17)
        ]

        # Hold-to-switch state machine:
        # Default mode is DRAW (writing)
        self.active_mode = "DRAW"
        self.hold_threshold = 14  # ~0.45s at 30 FPS

        # Hold counters for activating non-default modes
        self.erase_hold_count = 0
        self.select_hold_count = 0
        self.pause_hold_count = 0

        # Auto-exit counters when in non-default modes
        self.exit_erase_count = 0
        self.exit_select_count = 0

        # Current transition visual info: {'target': 'ERASE', 'progress': float, 'pt': (x, y)}
        self.transition_info = None

        # Pinch tracking
        self.is_pinching = False

    def _ensure_model_exists(self, path: str):
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            print(f"[HandTracker] Downloading model bundle to {path}...")
            urllib.request.urlretrieve(MODEL_URL, path)
            print("[HandTracker] Download completed.")

    def find_hand(self, bgr_frame: np.ndarray):
        """
        Processes frame using downscaled buffer for fast 60+ FPS inference
        and projects landmarks back to original frame dimensions.
        """
        orig_h, orig_w, _ = bgr_frame.shape

        scale_w = min(640, orig_w)
        scale_h = int(orig_h * (scale_w / orig_w))
        resized = cv2.resize(bgr_frame, (scale_w, scale_h), interpolation=cv2.INTER_LINEAR)
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.detector.detect(mp_image)

        if not result.hand_landmarks or len(result.hand_landmarks) == 0:
            return None, None

        hand_lms = result.hand_landmarks[0]
        landmarks_px = []
        for lm in hand_lms:
            cx = int(np.clip(lm.x * orig_w, 0, orig_w - 1))
            cy = int(np.clip(lm.y * orig_h, 0, orig_h - 1))
            landmarks_px.append((cx, cy))

        return landmarks_px, hand_lms

    def get_hand_scale(self, landmarks_px: list) -> float:
        """Returns pixel distance between wrist (0) and middle MCP (9)."""
        if not landmarks_px:
            return 100.0
        p0 = np.array(landmarks_px[0], dtype=float)
        p9 = np.array(landmarks_px[9], dtype=float)
        return float(np.linalg.norm(p0 - p9))

    def get_fingers_up(self, landmarks_px: list) -> list:
        """
        Determines finger extension using 3D-aware knuckle distance ratios.
        Returns: [thumb_up, index_up, middle_up, ring_up, pinky_up]
        """
        if not landmarks_px or len(landmarks_px) < 21:
            return [False] * 5

        wrist = np.array(landmarks_px[0], dtype=float)
        hand_size = self.get_hand_scale(landmarks_px)
        fingers = []

        # 1. Thumb
        thumb_tip = np.array(landmarks_px[4], dtype=float)
        thumb_ip = np.array(landmarks_px[3], dtype=float)
        pinky_mcp = np.array(landmarks_px[17], dtype=float)
        thumb_dist_tip = np.linalg.norm(thumb_tip - pinky_mcp)
        thumb_dist_ip = np.linalg.norm(thumb_ip - pinky_mcp)
        thumb_up = thumb_dist_tip > (thumb_dist_ip * 1.1)
        fingers.append(bool(thumb_up))

        # Check Pinch
        index_tip = np.array(landmarks_px[8], dtype=float)
        pinch_dist = np.linalg.norm(thumb_tip - index_tip)
        self.is_pinching = bool(pinch_dist < (hand_size * 0.18))

        # 2. Four Fingers (Index, Middle, Ring, Pinky)
        finger_indices = [
            (8, 6, 5),   # Index
            (12, 10, 9), # Middle
            (16, 14, 13),# Ring
            (20, 18, 17) # Pinky
        ]

        for tip_idx, pip_idx, mcp_idx in finger_indices:
            tip = np.array(landmarks_px[tip_idx], dtype=float)
            pip = np.array(landmarks_px[pip_idx], dtype=float)
            mcp = np.array(landmarks_px[mcp_idx], dtype=float)

            dist_tip_wrist = float(np.linalg.norm(tip - wrist))
            dist_pip_wrist = float(np.linalg.norm(pip - wrist))

            dist_tip_mcp = float(np.linalg.norm(tip - mcp))
            dist_pip_mcp = float(np.linalg.norm(pip - mcp))

            dist_ratio_mcp = dist_tip_mcp / max(1.0, dist_pip_mcp)
            dist_ratio_wrist = dist_tip_wrist / max(1.0, dist_pip_wrist)

            is_extended = (dist_ratio_mcp >= 1.25 and dist_ratio_wrist >= 1.05)
            fingers.append(bool(is_extended))

        return fingers

    def get_gesture_mode(self, fingers: list, landmarks_px: list = None) -> tuple:
        """
        Hold-to-Switch Gesture State Machine:
        - Default mode is always 'DRAW' (uninterrupted handwriting).
        - Switching to 'ERASE' requires holding all 5 fingers open for hold_threshold frames.
        - Switching to 'SELECT' requires holding index + middle for hold_threshold frames.
        - Once done erasing/selecting, automatically reverts back to default 'DRAW' mode.
        Returns:
            (active_mode: str, transition_info: dict or None)
        """
        if not fingers or len(fingers) != 5:
            self.transition_info = None
            return self.active_mode, None

        thumb, index, middle, ring, pinky = fingers
        hand_pt = landmarks_px[9] if (landmarks_px and len(landmarks_px) > 9) else (landmarks_px[8] if landmarks_px else None)

        is_all_five = index and middle and ring and pinky
        is_two_fingers = index and middle and not ring and not pinky
        is_fist = not index and not middle and not ring and not pinky

        self.transition_info = None

        if self.active_mode == "DRAW":
            # 1. Check if holding 5 fingers to activate ERASE
            if is_all_five:
                self.erase_hold_count += 1
                self.select_hold_count = 0
                self.pause_hold_count = 0
                prog = min(1.0, self.erase_hold_count / float(self.hold_threshold))
                self.transition_info = {"target": "ERASE", "progress": prog, "pt": hand_pt}
                if self.erase_hold_count >= self.hold_threshold:
                    self.active_mode = "ERASE"
                    self.erase_hold_count = 0
                    self.exit_erase_count = 0
                    self.transition_info = None

            # 2. Check if holding 2 fingers to activate SELECT
            elif is_two_fingers:
                self.select_hold_count += 1
                self.erase_hold_count = 0
                self.pause_hold_count = 0
                prog = min(1.0, self.select_hold_count / float(self.hold_threshold))
                self.transition_info = {"target": "SELECT", "progress": prog, "pt": hand_pt}
                if self.select_hold_count >= self.hold_threshold:
                    self.active_mode = "SELECT"
                    self.select_hold_count = 0
                    self.exit_select_count = 0
                    self.transition_info = None

            # 3. Check if holding fist to pause
            elif is_fist:
                self.pause_hold_count += 1
                self.erase_hold_count = 0
                self.select_hold_count = 0
                if self.pause_hold_count >= (self.hold_threshold // 2):
                    self.active_mode = "IDLE"
                    self.pause_hold_count = 0
            else:
                # Normal writing or moving -> decay hold counters
                self.erase_hold_count = max(0, self.erase_hold_count - 1)
                self.select_hold_count = max(0, self.select_hold_count - 1)
                self.pause_hold_count = 0

            return self.active_mode, self.transition_info

        elif self.active_mode == "ERASE":
            # In ERASE mode: user erases with open palm
            if is_all_five:
                self.exit_erase_count = 0
            else:
                self.exit_erase_count += 1
                if self.exit_erase_count >= 5:  # ~0.15s of stopping eraser
                    # Automatically revert to default DRAW mode!
                    self.active_mode = "DRAW"
                    self.exit_erase_count = 0

            return self.active_mode, None

        elif self.active_mode == "SELECT":
            # In SELECT mode: user navigates toolbar
            if is_two_fingers:
                self.exit_select_count = 0
            else:
                self.exit_select_count += 1
                if self.exit_select_count >= 6:
                    # Automatically revert to default DRAW mode!
                    self.active_mode = "DRAW"
                    self.exit_select_count = 0

            return self.active_mode, None

        elif self.active_mode == "IDLE":
            # In IDLE / pause mode: as soon as hand opens or index points, resume DRAW
            if not is_fist:
                self.active_mode = "DRAW"

            return self.active_mode, None

        return self.active_mode, None

    def draw_hand_skeleton(self, frame: np.ndarray, landmarks_px: list, mode: str = "DRAW") -> np.ndarray:
        """Renders stylized cyber skeleton overlay."""
        if not landmarks_px:
            return frame

        if mode == "DRAW":
            bone_color = (0, 255, 128)      # Neon Green
            joint_color = (0, 255, 255)     # Yellow
        elif mode == "SELECT":
            bone_color = (255, 200, 0)      # Neon Cyan
            joint_color = (255, 255, 255)   # White
        elif mode == "ERASE":
            bone_color = (0, 100, 255)      # Bright Orange
            joint_color = (0, 0, 255)       # Red
        else:
            bone_color = (160, 160, 170)    # Subdued Gray
            joint_color = (200, 200, 210)

        for p1_idx, p2_idx in self.connections:
            p1 = landmarks_px[p1_idx]
            p2 = landmarks_px[p2_idx]
            cv2.line(frame, p1, p2, bone_color, 2, cv2.LINE_AA)

        for idx, pt in enumerate(landmarks_px):
            if idx == 8:  # Index fingertip
                cv2.circle(frame, pt, 8, joint_color, -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 13, (255, 255, 255), 2, cv2.LINE_AA)
            elif idx == 4 and self.is_pinching:
                cv2.circle(frame, pt, 7, (0, 255, 255), -1, cv2.LINE_AA)
            else:
                cv2.circle(frame, pt, 4, joint_color, -1, cv2.LINE_AA)

        return frame
