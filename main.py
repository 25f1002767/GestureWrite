"""
AirWrite (Virtual Air Canvas & Touch-Free Handwriting Recognition)
Features:
- DirectShow robust webcam streaming with resolution fallback
- High-speed 3D knuckle-based hand tracking with gesture hysteresis
- Mask-based virtual canvas with lossless stroke-level undo
- Letter segmentation (temporal pause & spatial clustering)
- Dual recognition: Machine Learning (SVM/KNN) + Windows Native OCR
- Automatic clipboard synchronization
- Adaptive HUD overlay fitting any resolution (640x480 to 4K)
"""

import sys
import time
import threading
import cv2
import numpy as np

from hand_tracker import HandTracker
from canvas_manager import CanvasManager
from ui_overlay import UIOverlay
from ocr_engine import OCREngine
from recognition.letter_segmenter import LetterSegmenter
from recognition.recognizer import CharacterRecognizer


def open_webcam(preferred_indices=(0, 1), req_width=1280, req_height=720):
    cap = None
    for idx in preferred_indices:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap is not None and cap.isOpened():
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                print(f"[Camera] Connected to camera index {idx} via DirectShow.")
                break
            cap.release()
            cap = None

        cap = cv2.VideoCapture(idx)
        if cap is not None and cap.isOpened():
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                print(f"[Camera] Connected to camera index {idx} via default backend.")
                break
            cap.release()
            cap = None

    if cap is None or not cap.isOpened():
        return None, 0, 0

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, req_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, req_height)
    ret, frame = cap.read()
    if ret and frame is not None:
        actual_h, actual_w = frame.shape[:2]
    else:
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or req_width
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or req_height

    return cap, actual_w, actual_h


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    print("=" * 68)
    print("  * AIRWRITE - TOUCH-FREE HANDWRITING TO DIGITAL TEXT *")
    print("=" * 68)
    print("  How to Write:")
    print("    1. Raise Index Finger to draw letters in the air (M, O, N, U, etc.)")
    print("    2. Pause ~0.8s after a letter: automatically converts to digital text!")
    print("    3. Multi-stroke letters (A, E, H, T, X): pause briefly between strokes.")
    print("    4. Digital text automatically syncs to your Windows Clipboard!")
    print()
    print("  Hand Gestures:")
    print("    [1] Index Finger UP        -> Write / Draw")
    print("    [2] Index + Middle (Hold)  -> Select menu / toolbar buttons")
    print("    [5] Open Palm (Hold)       -> Erase with palm")
    print("    [0] Fist / Curl finger     -> Pause / Hover")
    print()
    print("  Controls & Shortcuts:")
    print("    [Space] / [_]   Insert SPACE into digital text")
    print("    [Backspace]     Delete last character (BKSP)")
    print("    [C]             Clear Canvas (keeps text buffer intact!)")
    print("    [X]             Clear Text Buffer")
    print("    [R] / [Enter]   Convert Canvas to Text via OCR")
    print("    [U]             Undo last air stroke")
    print("    [S]             Save Canvas Snapshot")
    print("    [M]             Toggle View (Overlaid / Side-by-Side / Canvas)")
    print("    [+] / [-]       Brush Size")
    print("    [Q] / [ESC]     Quit")
    print("=" * 68)

    # Initialize Camera
    cap, actual_w, actual_h = open_webcam()
    if cap is None:
        print("[Error] No active camera could be found. Please check your webcam connection and permissions.")
        print("Exiting...")
        return

    print(f"[Camera] Initialized with resolution: {actual_w}x{actual_h}")

    # Initialize modules
    hand_tracker = HandTracker()
    canvas_mgr = CanvasManager(width=actual_w, height=actual_h)
    ui = UIOverlay(width=actual_w, height=actual_h)
    ocr_engine = OCREngine()
    segmenter = LetterSegmenter(pause_timeout=0.85)
    recognizer = CharacterRecognizer()

    display_modes = ["overlaid", "side_by_side", "canvas_only"]
    mode_index = 0

    window_name = "AirWrite - Touch-Free Handwriting Recognition"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, actual_w, actual_h)

    prev_time = time.time()
    fps = 30.0

    # Recognition state
    is_recognizing = False
    ml_word_buffer = ""
    prev_mode = "DRAW"
    last_move_time = time.time()
    last_cursor_pos = None

    def handle_action(action: str):
        nonlocal mode_index, ml_word_buffer
        if action == "action_ocr":
            trigger_recognition()
        elif action == "action_space":
            ml_word_buffer += " "
            ui.set_recognized_text(ml_word_buffer)
            ocr_engine.copy_to_clipboard(ml_word_buffer)
            ui.show_notification(f"SPACE -> \"{ml_word_buffer}\"")
        elif action == "action_backspace":
            if ml_word_buffer:
                ml_word_buffer = ml_word_buffer[:-1]
                ui.set_recognized_text(ml_word_buffer)
                ocr_engine.copy_to_clipboard(ml_word_buffer)
                ui.show_notification(f"BKSP -> \"{ml_word_buffer}\"")
        elif action == "action_clear_text":
            ml_word_buffer = ""
            ui.set_recognized_text("")
            ocr_engine.copy_to_clipboard("")
            segmenter.clear()
            ui.show_notification("Text Buffer Cleared")
        elif action == "action_clear_canvas":
            canvas_mgr.clear()
            segmenter.clear()
            ui.show_notification("Canvas Cleared (Text Preserved)")
        elif action.startswith("color_"):
            cname = action.replace("color_", "")
            canvas_mgr.set_color(cname)
            ui.show_notification(f"Selected Color: {cname}")
        elif action == "tool_eraser":
            canvas_mgr.toggle_eraser_tool()
            st = "ON" if canvas_mgr.is_eraser_tool else "OFF"
            ui.show_notification(f"Eraser Tool: {st}")
        elif action == "size_inc":
            canvas_mgr.set_brush_size(canvas_mgr.brush_size + 2)
            ui.show_notification(f"Brush Size: {canvas_mgr.brush_size}px")
        elif action == "size_dec":
            canvas_mgr.set_brush_size(canvas_mgr.brush_size - 2)
            ui.show_notification(f"Brush Size: {canvas_mgr.brush_size}px")
        elif action == "action_undo":
            if canvas_mgr.undo():
                ui.show_notification("Undone last stroke")
            else:
                ui.show_notification("Nothing to undo")
        elif action == "action_save":
            path = canvas_mgr.save_snapshot()
            ui.show_notification("Snapshot Saved!")
        elif action == "action_mode":
            mode_index = (mode_index + 1) % len(display_modes)
            ui.show_notification(f"View: {display_modes[mode_index].replace('_', ' ').title()}")

    def trigger_recognition():
        nonlocal is_recognizing, ml_word_buffer
        if is_recognizing:
            return
        if not canvas_mgr.has_content():
            ui.show_notification("Nothing drawn to recognize!")
            return

        def _worker():
            nonlocal is_recognizing, ml_word_buffer
            is_recognizing = True
            ui.show_notification("Recognizing text...")

            # 1. Try ML character recognition if ML model is trained
            recognized_text = ""
            if recognizer.is_trained():
                forced_strokes = segmenter.force_commit()
                if forced_strokes:
                    pred_char, conf, status = recognizer.predict_strokes(forced_strokes)
                    if status == "accepted":
                        ml_word_buffer += pred_char
                if ml_word_buffer:
                    recognized_text = ml_word_buffer

            # 2. If no ML text or ML untrained, run full-canvas Windows OCR
            if not recognized_text:
                recognized_text = ocr_engine.recognize_canvas(canvas_mgr.canvas)

            if recognized_text:
                ui.set_recognized_text(recognized_text)
                ocr_engine.copy_to_clipboard(recognized_text)
                ui.show_notification(f"Copied: \"{recognized_text}\"", duration=3.5)
                print(f"[AirWrite] Recognized & Copied to Clipboard: \"{recognized_text}\"")
            else:
                ui.show_notification("Could not detect clear letters. Try writing bigger!")

            is_recognizing = False

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    ui.show_notification("Ready! Air-write letters (M, O, N, U...) - Pause ~0.8s to convert!", duration=4.5)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Warning] Failed to grab camera frame.")
            break

        # Mirror horizontally
        frame = cv2.flip(frame, 1)

        # Calculate FPS
        curr_time = time.time()
        dt = curr_time - prev_time
        prev_time = curr_time
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        # Detect Hand & Landmarks
        landmarks_px, _ = hand_tracker.find_hand(frame)

        mode = "DRAW"
        cursor_pt = None
        transition_info = None

        if landmarks_px:
            fingers = hand_tracker.get_fingers_up(landmarks_px)
            mode, transition_info = hand_tracker.get_gesture_mode(fingers, landmarks_px)

            # Detect mode transition to cleanly manage stroke buffers and user alerts
            if mode != prev_mode:
                if prev_mode == "DRAW":
                    st = canvas_mgr.end_stroke()
                    if st and len(st.get('points', [])) >= 3:
                        segmenter.add_stroke(st['points'])
                if mode == "ERASE":
                    ui.show_notification("ERASER ACTIVE (Open Palm)")
                elif mode == "SELECT":
                    ui.show_notification("MENU SELECTION ACTIVE (2 Fingers)")
                elif mode == "DRAW":
                    ui.show_notification("WRITING ACTIVE")
                prev_mode = mode

            if mode == "DRAW":
                raw_cursor = landmarks_px[8]
                cursor_pt = canvas_mgr.smooth_point(raw_cursor)

                # Dwell / motion pause detection (allows multi-stroke letter formation)
                if last_cursor_pos is not None:
                    d_move = float(np.hypot(cursor_pt[0] - last_cursor_pos[0], cursor_pt[1] - last_cursor_pos[1]))
                    if d_move > 8.0:
                        last_move_time = time.time()
                        last_cursor_pos = cursor_pt
                    elif (time.time() - last_move_time) > 0.40 and canvas_mgr.current_stroke:
                        st = canvas_mgr.end_stroke()
                        if st and len(st.get('points', [])) >= 3:
                            segmenter.add_stroke(st['points'])
                        last_move_time = time.time()
                else:
                    last_cursor_pos = cursor_pt
                    last_move_time = time.time()

                canvas_mgr.draw_to(cursor_pt)

            elif mode == "SELECT":
                idx_tip = np.array(landmarks_px[8])
                mid_tip = np.array(landmarks_px[12])
                raw_cursor = tuple(((idx_tip + mid_tip) / 2).astype(int))
                cursor_pt = canvas_mgr.smooth_point(raw_cursor)
                st = canvas_mgr.end_stroke()
                if st and len(st.get('points', [])) >= 3:
                    segmenter.add_stroke(st['points'])
                last_cursor_pos = None

                # Check toolbar button click
                action = ui.check_interaction(cursor_pt, is_selection_mode=True)
                if action:
                    handle_action(action)

            elif mode == "ERASE":
                st = canvas_mgr.end_stroke()
                if st and len(st.get('points', [])) >= 3:
                    segmenter.add_stroke(st['points'])
                raw_cursor = landmarks_px[9]
                cursor_pt = canvas_mgr.smooth_point(raw_cursor)
                canvas_mgr.erase_at(cursor_pt, radius=canvas_mgr.eraser_radius * 2)
                last_cursor_pos = None

            else:  # IDLE
                st = canvas_mgr.end_stroke()
                if st and len(st.get('points', [])) >= 3:
                    segmenter.add_stroke(st['points'])
                cursor_pt = landmarks_px[8]
                last_cursor_pos = None
        else:
            st = canvas_mgr.end_stroke()
            if st and len(st.get('points', [])) >= 3:
                segmenter.add_stroke(st['points'])
            ui.check_interaction(None, is_selection_mode=False)
            last_cursor_pos = None

        # Check temporal letter commit for ML recognition
        committed = segmenter.check_temporal_commit()
        if committed and recognizer.is_trained():
            p_char, conf, stat = recognizer.predict_strokes(committed)
            if stat == "accepted":
                ml_word_buffer += p_char
                ui.set_recognized_text(ml_word_buffer)
                ocr_engine.copy_to_clipboard(ml_word_buffer)
                ui.show_notification(f"Recognized: '{p_char}' ({int(conf*100)}%) -> \"{ml_word_buffer}\"", duration=3.0)
                print(f"[AirWrite Realtime] Added '{p_char}' ({int(conf*100)}%) | Digital Text: \"{ml_word_buffer}\"")

        # Composite output
        current_display_mode = display_modes[mode_index]
        output_frame = canvas_mgr.composite(frame, mode=current_display_mode)

        if current_display_mode == "overlaid" and landmarks_px:
            hand_tracker.draw_hand_skeleton(output_frame, landmarks_px, mode=mode)

        output_frame = ui.render(output_frame, canvas_mgr, mode=mode, fps=fps, cursor_pt=cursor_pt, transition_info=transition_info)

        cv2.imshow(window_name, output_frame)

        # Keyboard Controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("[Info] Exiting AirWrite...")
            break
        elif key == ord(' ') or key == ord('_'):
            handle_action("action_space")
        elif key == 8 or key == 127:  # Backspace / Delete
            handle_action("action_backspace")
        elif key == ord('x') or key == ord('X'):
            handle_action("action_clear_text")
        elif key == ord('c') or key == ord('C'):
            handle_action("action_clear_canvas")
        elif key == ord('r') or key == ord('R') or key == 13:
            handle_action("action_ocr")
        elif key == ord('u') or key == ord('U'):
            handle_action("action_undo")
        elif key == ord('s') or key == ord('S'):
            handle_action("action_save")
        elif key == ord('m') or key == ord('M') or key == 9:
            handle_action("action_mode")
        elif key == ord('e') or key == ord('E'):
            handle_action("tool_eraser")
        elif key == ord('+') or key == ord('='):
            handle_action("size_inc")
        elif key == ord('-'):
            handle_action("size_dec")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
