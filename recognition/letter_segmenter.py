"""
Letter Segmentation Module
Segments air-drawn strokes into individual characters based on temporal pause,
spatial bounding box separation, and gesture transitions per Prompt Section 15.
"""

import time
import numpy as np


class LetterSegmenter:
    def __init__(self, pause_timeout: float = 0.9, spatial_gap_ratio: float = 0.45):
        """
        pause_timeout: seconds of inactivity after writing before finalizing a letter.
        spatial_gap_ratio: horizontal distance threshold relative to character width to trigger a new letter.
        """
        self.pause_timeout = pause_timeout
        self.spatial_gap_ratio = spatial_gap_ratio

        self.current_letter_strokes = []
        self.last_stroke_time = 0
        self.completed_letters = []  # list of [strokes]
        self.is_active = False

    def add_stroke(self, stroke_points: list):
        """Adds a completed stroke to the active letter buffer."""
        if not stroke_points or len(stroke_points) < 2:
            return

        now = time.time()

        # Check spatial separation if we already have strokes
        if self.current_letter_strokes:
            curr_pts = np.vstack(self.current_letter_strokes)
            curr_max_x = np.max(curr_pts[:, 0])
            curr_w = max(40.0, float(curr_max_x - np.min(curr_pts[:, 0])))

            new_pts = np.array(stroke_points)
            new_min_x = np.min(new_pts[:, 0])

            # If new stroke starts far to the right of current letter
            if (new_min_x - curr_max_x) > (curr_w * self.spatial_gap_ratio):
                self._commit_current_letter()

        self.current_letter_strokes.append(list(stroke_points))
        self.last_stroke_time = now
        self.is_active = True

    def check_temporal_commit(self) -> list:
        """
        Called in the main loop to check if the pause timeout has elapsed.
        Returns the finalized letter's strokes if committed, else None.
        """
        if not self.is_active or not self.current_letter_strokes:
            return None

        if (time.time() - self.last_stroke_time) >= self.pause_timeout:
            return self._commit_current_letter()

        return None

    def force_commit(self) -> list:
        """Manually finalizes whatever is in the letter buffer."""
        if self.current_letter_strokes:
            return self._commit_current_letter()
        return None

    def _commit_current_letter(self) -> list:
        letter_strokes = list(self.current_letter_strokes)
        self.completed_letters.append(letter_strokes)
        self.current_letter_strokes = []
        self.is_active = False
        return letter_strokes

    def clear(self):
        self.current_letter_strokes = []
        self.completed_letters = []
        self.is_active = False
        self.last_stroke_time = 0
