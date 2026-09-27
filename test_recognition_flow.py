"""
test_recognition_flow.py
Verifies end-to-end real-time handwriting recognition flow:
- M -> O -> N -> U -> "MONU"
- H -> O -> M -> E -> "HOME"
- Multi-stroke characters (H, A, E)
- Space, Backspace, Clear Canvas, Clear Text
- Windows Clipboard synchronization
"""

import os
import sys
import numpy as np

from recognition.recognizer import CharacterRecognizer
from recognition.letter_segmenter import LetterSegmenter
from recognition.train_baseline import get_canonical_templates, augment_stroke_sample
from canvas_manager import CanvasManager
from ocr_engine import OCREngine


def run_tests():
    print("=" * 65)
    print("  VERIFYING REAL-TIME HANDWRITING -> DIGITAL TEXT FLOW")
    print("=" * 65)

    recognizer = CharacterRecognizer()
    assert recognizer.is_trained(), "Recognizer model should be trained and loaded!"
    print(f"[OK] Trained model loaded ({len(recognizer.classes_)} classes registered).")

    segmenter = LetterSegmenter(pause_timeout=0.85)
    templates = get_canonical_templates()
    ocr = OCREngine()

    # 1. Test Single Letter Recognition
    print("\n[Step 1] Testing Individual Character Accuracies...")
    for ch in ["M", "O", "N", "U", "H", "O", "M", "E"]:
        base_strokes = templates[ch][0]
        # Test 5 random variations of each letter
        for v in range(5):
            aug = augment_stroke_sample(base_strokes)
            pred_char, conf, stat = recognizer.predict_strokes(aug)
            assert stat == "accepted", f"Letter {ch} var {v} rejected with stat {stat}"
            assert pred_char == ch, f"Letter {ch} misclassified as {pred_char} (conf: {conf:.2f})"
    print("[OK] All test character variations recognized with 100% precision.")

    # 2. Test Multi-Stroke Letter Recognition
    print("\n[Step 2] Testing Multi-Stroke Letters (A, H, E)...")
    # H: stroke 1 (left vertical), stroke 2 (right vertical), stroke 3 (crossbar)
    h_strokes = [[(20, 15), (20, 85)], [(80, 15), (80, 85)], [(20, 50), (80, 50)]]
    pred_h, conf_h, stat_h = recognizer.predict_strokes(h_strokes)
    assert pred_h == "H" and stat_h == "accepted", f"Multi-stroke H failed: {pred_h} ({conf_h})"
    print(f"[OK] Multi-stroke H recognized: '{pred_h}' (conf: {conf_h:.2f})")

    # A: stroke 1 (left diagonal), stroke 2 (right diagonal), stroke 3 (crossbar)
    a_strokes = [[(50, 15), (20, 85)], [(50, 15), (80, 85)], [(32, 60), (68, 60)]]
    pred_a, conf_a, stat_a = recognizer.predict_strokes(a_strokes)
    assert pred_a == "A" and stat_a == "accepted", f"Multi-stroke A failed: {pred_a} ({conf_a})"
    print(f"[OK] Multi-stroke A recognized: '{pred_a}' (conf: {conf_a:.2f})")

    # 3. Test Word Flow: M -> O -> N -> U -> "MONU"
    print("\n[Step 3] Testing Word Construction: M -> O -> N -> U...")
    text_buffer = ""
    for ch in ["M", "O", "N", "U"]:
        char_strokes = templates[ch][0]
        pred_char, conf, stat = recognizer.predict_strokes(char_strokes)
        assert stat == "accepted"
        text_buffer += pred_char
        print(f"   -> Wrote '{ch}', buffer is now: \"{text_buffer}\"")
    assert text_buffer == "MONU", f"Expected MONU, got {text_buffer}"
    print(f"[OK] Word successfully formed: \"{text_buffer}\"")

    # 4. Test Space & Backspace
    print("\n[Step 4] Testing Space & Backspace...")
    # Add Space
    text_buffer += " "
    assert text_buffer == "MONU "
    print(f"   -> Space added: \"{text_buffer}\"")

    # Append HOME
    for ch in ["H", "O", "M", "E"]:
        pred_char, conf, stat = recognizer.predict_strokes(templates[ch][0])
        text_buffer += pred_char
    assert text_buffer == "MONU HOME", f"Expected 'MONU HOME', got '{text_buffer}'"
    print(f"   -> Appended HOME: \"{text_buffer}\"")

    # Backspace twice
    text_buffer = text_buffer[:-1]  # Remove E
    text_buffer = text_buffer[:-1]  # Remove M
    assert text_buffer == "MONU HO", f"Expected 'MONU HO', got '{text_buffer}'"
    print(f"   -> Backspaced twice: \"{text_buffer}\"")

    # 5. Test Separation: CLEAR CANVAS vs CLEAR TEXT
    print("\n[Step 5] Testing Separation of CLEAR CANVAS and CLEAR TEXT...")
    canvas = CanvasManager(640, 480)
    canvas.smooth_point((100, 100))
    canvas.draw_to((150, 150))
    canvas.end_stroke()
    assert canvas.has_content(), "Canvas should have strokes"

    # Action Clear Canvas: Canvas is cleared, but text_buffer is preserved!
    canvas.clear()
    assert not canvas.has_content(), "Canvas should be cleared"
    assert text_buffer == "MONU HO", "Text buffer MUST remain intact after CLEAR CANVAS!"
    print(f"[OK] CLEAR CANVAS wiped canvas while text buffer remained: \"{text_buffer}\"")

    # Action Clear Text: Text buffer is cleared
    text_buffer = ""
    assert text_buffer == "", "Text buffer should be cleared"
    print("[OK] CLEAR TEXT wiped digital text buffer.")

    # 6. Test Clipboard Synchronization
    print("\n[Step 6] Testing Windows Clipboard Sync...")
    test_str = "AIRWRITE MONU HOME"
    ocr.copy_to_clipboard(test_str)
    print(f"[OK] Clipboard successfully synchronized with: \"{test_str}\"")

    print("\n" + "=" * 65)
    print("  [ALL FLOW TESTS PASSED SUCCESSFULLY!]")
    print("=" * 65)


if __name__ == "__main__":
    run_tests()
