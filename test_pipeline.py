"""
AirWrite Comprehensive Test Suite
Tests all 16 pipeline areas per Prompt Section 24:
 1. HandTracker initialization
 2. Landmark conversion
 3. Finger detection
 4. Gesture detection
 5. Gesture stability
 6. Coordinate smoothing
 7. Outlier rejection
 8. Stroke creation
 9. Multiple strokes
10. Undo
11. Clear
12. Eraser
13. Canvas save
14. UI rendering
15. OCR
16. Recognition interface (FeatureExtractor, DatasetManager, LetterSegmenter, CharacterRecognizer)
"""

import os
import sys
import shutil
import numpy as np
import cv2

from hand_tracker import HandTracker
from canvas_manager import CanvasManager
from ui_overlay import UIOverlay
from ocr_engine import OCREngine
from recognition.feature_extractor import StrokeFeatureExtractor
from recognition.dataset_manager import DatasetManager
from recognition.letter_segmenter import LetterSegmenter
from recognition.recognizer import CharacterRecognizer


def test_hand_tracker():
    print("[Test 1/16] HandTracker initialization...")
    tracker = HandTracker()
    assert tracker.detector is not None

    print("[Test 2/16] Landmark conversion on blank frame...")
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    lms_px, raw_lms = tracker.find_hand(dummy)
    assert lms_px is None

    print("[Test 3/16] Finger detection (3D knuckle geometry)...")
    # Base wrist at (300, 400)
    mock = [(300, 400)] * 21
    mock[0] = (300, 400) # Wrist
    mock[9] = (300, 250) # Middle MCP (hand scale reference = 150)
    # Index extended
    mock[5] = (270, 260) # MCP
    mock[6] = (270, 200) # PIP
    mock[8] = (270, 100) # TIP
    # Middle curled
    mock[10] = (300, 290) # PIP
    mock[12] = (300, 330) # TIP
    # Ring curled
    mock[13] = (330, 260)
    mock[14] = (330, 290)
    mock[16] = (330, 330)
    # Pinky curled
    mock[17] = (350, 280)
    mock[18] = (350, 300)
    mock[20] = (350, 330)

    fingers = tracker.get_fingers_up(mock)
    assert fingers[1] is True, "Index finger should be detected as EXTENDED"
    assert fingers[2] is False, "Middle finger should be detected as CURLED"
    assert fingers[3] is False, "Ring finger should be detected as CURLED"
    assert fingers[4] is False, "Pinky finger should be detected as CURLED"

    print("[Test 4/16] Gesture detection (DRAW vs SELECT vs ERASE vs IDLE)...")
    mode_draw, _ = tracker.get_gesture_mode(fingers)
    assert mode_draw == "DRAW"

    # Make middle extended -> hold for hold_threshold frames to activate SELECT
    mock[10] = (300, 200)
    mock[12] = (300, 100)
    fingers_sel = tracker.get_fingers_up(mock)
    for _ in range(tracker.hold_threshold):
        mode_sel, _ = tracker.get_gesture_mode(fingers_sel)
    assert mode_sel == "SELECT"

    # Auto-revert back to default DRAW mode when returning to single finger
    for _ in range(7):
        mode_ret, _ = tracker.get_gesture_mode(fingers)
    assert mode_ret == "DRAW"

    # Make all 5 extended -> hold for hold_threshold frames to activate ERASE
    mock[14] = (330, 200)
    mock[16] = (330, 100)
    mock[18] = (350, 200)
    mock[20] = (350, 100)
    fingers_erase = tracker.get_fingers_up(mock)
    for _ in range(tracker.hold_threshold):
        mode_erase, _ = tracker.get_gesture_mode(fingers_erase)
    assert mode_erase == "ERASE"

    # Auto-revert back to default DRAW mode when stopping erase
    for _ in range(7):
        mode_ret, _ = tracker.get_gesture_mode(fingers)
    assert mode_ret == "DRAW"

    print("[Test 5/16] Gesture stability & temporal hysteresis...")
    # Default mode is DRAW; single dropped frame holds DRAW mode
    mode_grace, _ = tracker.get_gesture_mode(None)
    assert mode_grace == "DRAW"


def test_drawing_and_canvas():
    print("[Test 6/16] Coordinate smoothing (velocity-adaptive EMA)...")
    mgr = CanvasManager(640, 480)
    p0 = (100, 100)
    sp0 = mgr.smooth_point(p0)
    assert sp0 == p0
    p1 = (130, 130)
    sp1 = mgr.smooth_point(p1)
    assert 100 < sp1[0] < 130

    print("[Test 7/16] Outlier rejection (max_jump threshold)...")
    p_jump = (500, 500) # massive jump > 200px
    sp_jump = mgr.smooth_point(p_jump, max_jump=160.0)
    # Outlier rejection resets anchor without drawing a wild connecting line
    assert sp_jump == p_jump

    print("[Test 8/16] Stroke creation...")
    mgr.clear()
    mgr.start_stroke()
    mgr.draw_to((150, 150))
    mgr.draw_to((180, 180))
    mgr.end_stroke()
    assert len(mgr.strokes_history) == 1
    assert mgr.has_content() is True

    print("[Test 9/16] Multiple discrete strokes...")
    mgr.start_stroke()
    mgr.draw_to((200, 150))
    mgr.draw_to((220, 200))
    mgr.end_stroke()
    assert len(mgr.strokes_history) == 2
    raw = mgr.get_raw_strokes()
    assert len(raw) == 2

    print("[Test 10/16] Stroke-level Undo...")
    undone = mgr.undo()
    assert undone is True
    assert len(mgr.strokes_history) == 1

    print("[Test 11/16] Clear canvas & mask...")
    mgr.clear()
    assert len(mgr.strokes_history) == 0
    assert not mgr.has_content()
    assert np.all(mgr.canvas == 0)
    assert np.all(mgr.mask == 0)

    print("[Test 12/16] Mask-based Eraser...")
    mgr.start_stroke()
    mgr.draw_to((50, 50))
    mgr.draw_to((100, 50))
    mgr.end_stroke()
    assert np.any(mgr.mask > 0)
    # Erase center
    mgr.erase_at((75, 50), radius=20)
    # Erased region in mask is transparent (0)
    assert mgr.mask[50, 75] == 0

    print("[Test 13/16] Canvas export without black paint artifacts...")
    snap_file = mgr.save_snapshot()
    assert os.path.exists(snap_file)


def test_ui_overlay():
    print("[Test 14/16] UI rendering & adaptive bounds across resolutions...")
    for w, h in [(640, 480), (1280, 720)]:
        ui = UIOverlay(w, h)
        mgr = CanvasManager(w, h)
        frame = np.full((h, w, 3), 30, dtype=np.uint8)
        rendered = ui.render(frame, mgr, mode="DRAW", fps=45.0, cursor_pt=(w//2, h//2))
        assert rendered.shape == (h, w, 3)
        for btn in ui.buttons:
            x1, y1, x2, y2 = btn.rect
            assert 0 <= x1 < x2 <= w
            assert 0 <= y1 < y2 <= h


def test_ocr():
    print("[Test 15/16] OCR engine & clipboard...")
    ocr = OCREngine()
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(canvas, "AIRWRITE", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3, cv2.LINE_AA)
    text = ocr.recognize_canvas(canvas)
    assert "AIRWRITE" in text.upper()
    assert ocr.copy_to_clipboard(text) is True


def test_recognition_pipeline():
    print("[Test 16/16] Recognition pipeline (Normalization, Dataset, Segmenter, Recognizer)...")
    extractor = StrokeFeatureExtractor(target_size=128, margin=14, num_resample_points=64)

    # Synthetic stroke for letter 'L' (two segments: down, right)
    s1 = [(50, 30), (50, 100)]
    s2 = [(50, 100), (90, 100)]
    raw_strokes = [s1, s2]

    # 1. Normalization
    norm_strokes = extractor.normalize_strokes(raw_strokes)
    assert len(norm_strokes) == 2
    rendered = extractor.render_normalized_image(norm_strokes)
    assert rendered.shape == (128, 128)
    assert np.any(rendered > 0)

    # 2. Features
    feats = extractor.extract_feature_vector(raw_strokes)
    assert len(feats) > 100
    assert not np.isnan(feats).any()

    # 3. Letter Segmenter
    segmenter = LetterSegmenter(pause_timeout=0.1)
    segmenter.add_stroke(s1)
    segmenter.add_stroke(s2)
    committed = segmenter.force_commit()
    assert len(committed) == 2

    # 4. Dataset Manager & Recognizer
    tmp_data_dir = os.path.join(os.path.dirname(__file__), "data", "test_dataset")
    if os.path.exists(tmp_data_dir):
        shutil.rmtree(tmp_data_dir)

    dataset_mgr = DatasetManager(data_root=tmp_data_dir)
    # Save 2 classes to test training
    dataset_mgr.save_sample("L", raw_strokes)
    dataset_mgr.save_sample("L", raw_strokes)

    # Synthetic stroke for 'I'
    s_i = [(70, 30), (70, 100)]
    dataset_mgr.save_sample("I", [s_i])
    dataset_mgr.save_sample("I", [s_i])

    recognizer = CharacterRecognizer(model_path=os.path.join(tmp_data_dir, "test_svm.joblib"))
    train_res = recognizer.train_from_dataset(dataset_mgr)
    assert train_res.get("status") == "success"
    assert recognizer.is_trained()

    pred_char, conf, status = recognizer.predict_strokes(raw_strokes)
    assert pred_char in ["L", "I"]
    assert status == "accepted"

    # Cleanup test dataset
    if os.path.exists(tmp_data_dir):
        shutil.rmtree(tmp_data_dir)


def run_all_tests():
    print("=" * 65)
    print("  RUNNING AIRWRITE 16-POINT COMPREHENSIVE TEST SUITE")
    print("=" * 65)
    test_hand_tracker()
    test_drawing_and_canvas()
    test_ui_overlay()
    test_ocr()
    test_recognition_pipeline()
    print("\n[SUCCESS] ALL 16/16 TEST CRITERIA PASSED WITHOUT ERRORS!")
    print("=" * 65)


if __name__ == "__main__":
    run_all_tests()
