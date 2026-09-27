"""
AirWrite Dataset Collection & Model Training Studio
Allows users to select a character (A-Z, a-z), write it in the air,
save normalized stroke data (JSON) and canonical images (PNG) per Prompt Section 18,
and train the SVM character classifier on demand.
"""

import sys
import os
import time
import cv2
import numpy as np

from hand_tracker import HandTracker
from canvas_manager import CanvasManager
from recognition.dataset_manager import DatasetManager
from recognition.recognizer import CharacterRecognizer
from main import open_webcam


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    print("=" * 65)
    print("  * AIRWRITE - DATASET COLLECTION & ML TRAINING STUDIO *")
    print("=" * 65)
    print("  Instructions:")
    print("    1. Write the target character in the air with index finger.")
    print("    2. Press [Space] or [Enter] to SAVE the sample & clear for next.")
    print("    3. Press [N] or [P] to change target character (A -> B -> C...).")
    print("    4. Press [T] to TRAIN the ML classifier on all collected data.")
    print("    5. Press [C] to clear current drawing without saving.")
    print("    6. Press [Q] / [ESC] to quit.")
    print("=" * 65)

    dataset_mgr = DatasetManager()
    recognizer = CharacterRecognizer()

    # Target characters list: uppercase A-Z then lowercase a-z
    classes = [chr(c) for c in range(ord('A'), ord('Z') + 1)] + [chr(c) for c in range(ord('a'), ord('z') + 1)]
    current_char_idx = 0

    cap, actual_w, actual_h = open_webcam()
    if cap is None:
        print("[Error] Could not open webcam.")
        return

    hand_tracker = HandTracker()
    canvas_mgr = CanvasManager(width=actual_w, height=actual_h)

    window_name = "AirWrite - Dataset Collector"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, actual_w, actual_h)

    notification = "Write the target letter, then press [Space] to save sample."
    notification_time = time.time() + 4.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        landmarks_px, _ = hand_tracker.find_hand(frame)
        mode = "IDLE"

        if landmarks_px:
            fingers = hand_tracker.get_fingers_up(landmarks_px)
            mode = hand_tracker.get_gesture_mode(fingers)

            if mode == "DRAW":
                pt = canvas_mgr.smooth_point(landmarks_px[8])
                canvas_mgr.draw_to(pt)
            else:
                canvas_mgr.end_stroke()
        else:
            if hand_tracker.draw_grace_counter > 0 and hand_tracker.current_mode == "DRAW":
                hand_tracker.draw_grace_counter -= 1
            else:
                canvas_mgr.end_stroke()

        # Composite frame
        out = canvas_mgr.composite(frame, mode="overlaid")
        if landmarks_px:
            hand_tracker.draw_hand_skeleton(out, landmarks_px, mode=mode)

        # Draw Studio HUD Overlay
        target_char = classes[current_char_idx]
        stats = dataset_mgr.get_dataset_stats()
        key_folder = f"upper_{target_char}" if target_char.isupper() else f"lower_{target_char}"
        sample_count = stats.get(key_folder, 0)

        # Top banner: Target Character
        cv2.rectangle(out, (16, 12), (w - 16, 80), (25, 25, 30), -1)
        cv2.rectangle(out, (16, 12), (w - 16, 80), (120, 45, 185), 2, cv2.LINE_AA)

        char_text = f"TARGET: '{target_char}'"
        count_text = f"Saved Samples: {sample_count}"
        cv2.putText(out, char_text, (32, 54), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, count_text, (260, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 180), 2, cv2.LINE_AA)

        # Bottom banner: Shortcuts
        bot_y1, bot_y2 = h - 50, h - 14
        cv2.rectangle(out, (16, bot_y1), (w - 16, bot_y2), (25, 25, 30), -1)
        cv2.rectangle(out, (16, bot_y1), (w - 16, bot_y2), (70, 70, 85), 1, cv2.LINE_AA)
        help_str = "[Space] Save Sample | [N/P] Next/Prev Letter | [T] Train SVM | [C] Clear | [Q] Quit"
        cv2.putText(out, help_str, (28, bot_y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 230), 1, cv2.LINE_AA)

        # Toast
        if notification and time.time() < notification_time:
            cv2.rectangle(out, (w // 2 - 250, 95), (w // 2 + 250, 135), (20, 25, 30), -1)
            cv2.rectangle(out, (w // 2 - 250, 95), (w // 2 + 250, 135), (0, 220, 120), 2, cv2.LINE_AA)
            cv2.putText(out, notification, (w // 2 - 235, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(window_name, out)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == 32 or key == 13:  # Space or Enter: Save sample
            canvas_mgr.end_stroke()
            raw_strokes = canvas_mgr.get_raw_strokes()
            if raw_strokes:
                saved_path = dataset_mgr.save_sample(target_char, raw_strokes)
                canvas_mgr.clear()
                notification = f"Saved sample for '{target_char}'! Total: {sample_count + 1}"
                notification_time = time.time() + 2.5
                print(f"[Dataset] Saved: {saved_path}")
            else:
                notification = "Canvas is empty! Draw character first."
                notification_time = time.time() + 2.0
        elif key == ord('n') or key == ord('N'):
            current_char_idx = (current_char_idx + 1) % len(classes)
            canvas_mgr.clear()
        elif key == ord('p') or key == ord('P'):
            current_char_idx = (current_char_idx - 1) % len(classes)
            canvas_mgr.clear()
        elif key == ord('c') or key == ord('C'):
            canvas_mgr.clear()
            notification = "Canvas Cleared"
            notification_time = time.time() + 1.5
        elif key == ord('t') or key == ord('T'):
            notification = "Training SVM Classifier..."
            notification_time = time.time() + 3.0
            cv2.imshow(window_name, out)
            cv2.waitKey(10)
            res = recognizer.train_from_dataset(dataset_mgr)
            if res.get("status") == "success":
                acc = res.get("train_accuracy", 0.0) * 100.0
                notification = f"Model Trained! Classes: {res['classes']}, Accuracy: {acc:.1f}%"
                print(f"[Training] Success! Classes: {res['classes']}, Accuracy: {acc:.1f}%")
            else:
                notification = res.get("message", "Training failed.")
                print(f"[Training] {notification}")
            notification_time = time.time() + 4.0

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
