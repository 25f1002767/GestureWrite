"""
OCR & Handwriting Recognition Engine
Extracts written text from canvas using Windows native OCR (winocr)
and automatically copies recognized text to the Windows clipboard.
"""

import os
import re
from datetime import datetime
import numpy as np
import cv2
import winocr
import win32clipboard


class OCREngine:
    def __init__(self):
        self.history_file = os.path.join(os.path.dirname(__file__), "snapshots", "recognized_text.txt")
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        self.last_recognized_text = ""

    def copy_to_clipboard(self, text: str) -> bool:
        """Copies text to Windows system clipboard using native win32 API."""
        if not text:
            return False
        try:
            win32clipboard.OpenClipboard()
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardText(text, win32clipboard.CF_UNICODETEXT)
            win32clipboard.CloseClipboard()
            return True
        except Exception as e:
            print(f"[OCR] Clipboard error: {e}")
            return False

    def preprocess_canvas_for_ocr(self, canvas: np.ndarray):
        """
        Cuts out the active drawing region and creates a high-contrast
        black-on-white image optimal for Windows OCR engine.
        """
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

        # Find bounding box of drawing
        points = cv2.findNonZero(thresh)
        if points is None:
            return None

        x, y, w, h = cv2.boundingRect(points)
        if w < 10 or h < 10:
            return None

        # Add generous padding around the drawing
        pad = 40
        h_img, w_img = gray.shape
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w_img, x + w + pad)
        y2 = min(h_img, y + h + pad)

        cropped = thresh[y1:y2, x1:x2]

        # Invert to black text on white background (standard for OCR)
        ocr_input = cv2.bitwise_not(cropped)

        # Optional: dilate slightly if strokes are too thin
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        ocr_input = cv2.erode(ocr_input, kernel, iterations=1)

        # Convert back to 3-channel BGR for winocr
        ocr_bgr = cv2.cvtColor(ocr_input, cv2.COLOR_GRAY2BGR)
        return ocr_bgr

    def recognize_canvas(self, canvas: np.ndarray, lang: str = "en") -> str:
        """
        Recognizes handwritten text on the canvas and copies it to clipboard.
        Returns recognized string or empty string.
        """
        preprocessed = self.preprocess_canvas_for_ocr(canvas)
        if preprocessed is None:
            return ""

        try:
            res = winocr.recognize_cv2_sync(preprocessed, lang)
            raw_text = res.get("text", "")
            if not raw_text:
                # Try with lines
                lines = res.get("lines", [])
                raw_text = " ".join([l.get("text", "") for l in lines if l.get("text")])

            # Clean text
            cleaned = raw_text.strip()
            if cleaned:
                self.last_recognized_text = cleaned
                self.copy_to_clipboard(cleaned)
                self._log_to_history(cleaned)

            return cleaned
        except Exception as e:
            print(f"[OCR] Recognition error: {e}")
            return ""

    def _log_to_history(self, text: str):
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.history_file, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] {text}\n")
        except Exception as e:
            print(f"[OCR] Failed to log text: {e}")
