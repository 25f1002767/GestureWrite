"""
UI Overlay Module
Renders an adaptive, responsive interactive toolbar, OCR conversion buttons,
copyable text banners, floating cursors, mode badges, and status notifications.
Automatically adjusts layout for any camera resolution (640x480, 1280x720, 1920x1080).
"""

import time
import numpy as np
import cv2


class UIButton:
    def __init__(self, action_id: str, label: str, rect: tuple, color: tuple = None, bg_color: tuple = None):
        self.action_id = action_id
        self.label = label
        self.rect = rect  # (x1, y1, x2, y2)
        self.color = color or (255, 255, 255)
        self.bg_color = bg_color or (40, 40, 40)
        self.is_hovered = False

    def contains(self, pt: tuple) -> bool:
        if pt is None:
            return False
        x, y = pt
        x1, y1, x2, y2 = self.rect
        return x1 <= x <= x2 and y1 <= y <= y2


class UIOverlay:
    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height

        self.notification_text = ""
        self.notification_expiry = 0

        # Recognized text display
        self.recognized_text = ""
        self.recognized_timestamp = 0

        # Debounce for gesture clicking
        self.last_click_time = 0
        self.click_cooldown = 0.45  # seconds

        self.buttons = []
        self._init_toolbar()

    def update_resolution(self, width: int, height: int):
        if self.width != width or self.height != height:
            self.width = width
            self.height = height
            self._init_toolbar()

    def _init_toolbar(self):
        self.buttons = []
        w, h = self.width, self.height

        colors = [
            ("Cyan", (255, 230, 0)),
            ("Magenta", (255, 0, 230)),
            ("Lime", (50, 255, 50)),
            ("Yellow", (0, 235, 255)),
            ("Orange", (0, 140, 255)),
            ("White", (255, 255, 255)),
        ]

        if w >= 900:
            # Single-row layout for wide resolutions (720p, 1080p)
            bar_y1, bar_y2 = 10, min(68, int(h * 0.1))
            curr_x = 16
            btn_w = max(42, min(56, int(w * 0.042)))
            btn_gap = max(6, min(10, int(w * 0.007)))

            # 1. Colors
            for name, col in colors:
                self.buttons.append(
                    UIButton(
                        action_id=f"color_{name}",
                        label="",
                        rect=(curr_x, bar_y1, curr_x + btn_w, bar_y2),
                        color=col,
                        bg_color=col
                    )
                )
                curr_x += btn_w + btn_gap

            curr_x += 10

            # 2. Text Controls & Actions
            text_actions = [
                ("action_space", "SPACE", (60, 140, 60)),
                ("action_backspace", "BKSP", (140, 60, 60)),
                ("action_clear_text", "CLR TXT", (150, 80, 50)),
                ("action_clear_canvas", "CLR DRAW", (50, 60, 160)),
                ("action_ocr", "TO TEXT", (125, 45, 185)),
            ]
            for act_id, lbl, bg_col in text_actions:
                act_w = max(58, min(80, int(w * 0.058)))
                self.buttons.append(
                    UIButton(
                        action_id=act_id,
                        label=lbl,
                        rect=(curr_x, bar_y1, curr_x + act_w, bar_y2),
                        color=(255, 255, 255),
                        bg_color=bg_col
                    )
                )
                curr_x += act_w + btn_gap

            curr_x += 6

            # 3. Canvas Tools (Eraser & Brush Size)
            erase_w = max(50, min(65, int(w * 0.048)))
            self.buttons.append(
                UIButton(
                    action_id="tool_eraser",
                    label="ERASE",
                    rect=(curr_x, bar_y1, curr_x + erase_w, bar_y2),
                    color=(255, 255, 255),
                    bg_color=(50, 50, 60)
                )
            )
            curr_x += erase_w + btn_gap

            sz_w = max(42, min(56, int(w * 0.042)))
            self.buttons.append(
                UIButton(
                    action_id="size_dec",
                    label="- SZ",
                    rect=(curr_x, bar_y1, curr_x + sz_w, bar_y2),
                    color=(255, 255, 255),
                    bg_color=(50, 50, 60)
                )
            )
            curr_x += sz_w + btn_gap

            self.buttons.append(
                UIButton(
                    action_id="size_inc",
                    label="+ SZ",
                    rect=(curr_x, bar_y1, curr_x + sz_w, bar_y2),
                    color=(255, 255, 255),
                    bg_color=(50, 50, 60)
                )
            )
            curr_x += sz_w + btn_gap + 6

            # 4. Utilities (Undo & View)
            util_actions = [
                ("action_undo", "UNDO", (45, 55, 70)),
                ("action_mode", "VIEW", (45, 55, 70)),
            ]
            for act_id, lbl, bg_col in util_actions:
                u_w = max(48, min(62, int(w * 0.046)))
                self.buttons.append(
                    UIButton(
                        action_id=act_id,
                        label=lbl,
                        rect=(curr_x, bar_y1, min(w - 12, curr_x + u_w), bar_y2),
                        color=(255, 255, 255),
                        bg_color=bg_col
                    )
                )
                curr_x += u_w + btn_gap
        else:
            # Two-row compact layout for 640x480 and narrower screens
            # Row 1: Colors + Eraser + Sizes + View (y: 6 to 38)
            r1_y1, r1_y2 = 6, 38
            curr_x = 12
            c_w = 34
            c_gap = 5
            for name, col in colors:
                self.buttons.append(
                    UIButton(
                        action_id=f"color_{name}",
                        label="",
                        rect=(curr_x, r1_y1, curr_x + c_w, r1_y2),
                        color=col,
                        bg_color=col
                    )
                )
                curr_x += c_w + c_gap

            curr_x += 6

            self.buttons.append(
                UIButton(
                    action_id="tool_eraser",
                    label="ERASE",
                    rect=(curr_x, r1_y1, curr_x + 58, r1_y2),
                    color=(255, 255, 255),
                    bg_color=(50, 50, 60)
                )
            )
            curr_x += 58 + c_gap

            self.buttons.append(
                UIButton(
                    action_id="size_dec",
                    label="- SZ",
                    rect=(curr_x, r1_y1, curr_x + 46, r1_y2),
                    color=(255, 255, 255),
                    bg_color=(50, 50, 60)
                )
            )
            curr_x += 46 + c_gap

            self.buttons.append(
                UIButton(
                    action_id="size_inc",
                    label="+ SZ",
                    rect=(curr_x, r1_y1, curr_x + 46, r1_y2),
                    color=(255, 255, 255),
                    bg_color=(50, 50, 60)
                )
            )
            curr_x += 46 + c_gap

            self.buttons.append(
                UIButton(
                    action_id="action_mode",
                    label="VIEW",
                    rect=(curr_x, r1_y1, min(w - 12, curr_x + 52), r1_y2),
                    color=(255, 255, 255),
                    bg_color=(45, 55, 70)
                )
            )

            # Row 2: SPACE + BKSP + CLR TXT + CLR DRAW + TO TEXT + UNDO (y: 42 to 74)
            r2_y1, r2_y2 = 42, 74
            curr_x = 10
            r2_actions = [
                ("action_space", "SPACE", 54, (60, 140, 60)),
                ("action_backspace", "BKSP", 50, (140, 60, 60)),
                ("action_clear_text", "CLR TXT", 64, (150, 80, 50)),
                ("action_clear_canvas", "CLR DRAW", 70, (50, 60, 160)),
                ("action_ocr", "TO TEXT", 68, (125, 45, 185)),
                ("action_undo", "UNDO", 52, (45, 55, 70)),
            ]
            for act_id, lbl, bw, bg_col in r2_actions:
                self.buttons.append(
                    UIButton(
                        action_id=act_id,
                        label=lbl,
                        rect=(curr_x, r2_y1, min(w - 6, curr_x + bw), r2_y2),
                        color=(255, 255, 255),
                        bg_color=bg_col
                    )
                )
                curr_x += bw + 5

    def show_notification(self, text: str, duration: float = 2.5):
        self.notification_text = text
        self.notification_expiry = time.time() + duration

    def set_recognized_text(self, text: str):
        self.recognized_text = text
        self.recognized_timestamp = time.time()

    def check_interaction(self, cursor_pt: tuple, is_selection_mode: bool) -> str:
        if cursor_pt is None:
            for b in self.buttons:
                b.is_hovered = False
            return None

        triggered_action = None
        now = time.time()

        for btn in self.buttons:
            inside = btn.contains(cursor_pt)
            btn.is_hovered = inside

            if inside and is_selection_mode:
                if (now - self.last_click_time) > self.click_cooldown:
                    self.last_click_time = now
                    triggered_action = btn.action_id

        return triggered_action

    def render(self, frame: np.ndarray, canvas_mgr, mode: str, fps: float, cursor_pt: tuple, transition_info: dict = None) -> np.ndarray:
        h, w, _ = frame.shape
        if (w, h) != (self.width, self.height):
            self.update_resolution(w, h)

        # 1. Top Bar Backdrop
        max_bar_y = max(btn.rect[3] for btn in self.buttons) + 4
        sub_bar = frame[4:max_bar_y, 8:w-8]
        dark_bar = np.full_like(sub_bar, (20, 20, 25))
        cv2.addWeighted(dark_bar, 0.78, sub_bar, 0.22, 0, sub_bar)
        cv2.rectangle(frame, (8, 4), (w - 8, max_bar_y), (60, 60, 75), 1, cv2.LINE_AA)

        # 2. Render Buttons
        for btn in self.buttons:
            x1, y1, x2, y2 = btn.rect
            border_color = (0, 255, 255) if btn.is_hovered else (80, 80, 95)
            thickness = 2 if btn.is_hovered else 1

            is_active = False
            if btn.action_id.startswith("color_"):
                col_name = btn.action_id.replace("color_", "")
                if col_name == canvas_mgr.color_name and not canvas_mgr.is_eraser_tool:
                    is_active = True
            elif btn.action_id == "tool_eraser" and canvas_mgr.is_eraser_tool:
                is_active = True

            if is_active:
                border_color = (255, 255, 255)
                thickness = 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), btn.bg_color, -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, thickness, cv2.LINE_AA)

            if btn.label:
                font_scale = 0.38 if w < 900 else 0.42
                (tw, th), _ = cv2.getTextSize(btn.label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                tx = x1 + max(2, (x2 - x1 - tw) // 2)
                ty = y1 + (y2 - y1 + th) // 2
                cv2.putText(frame, btn.label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, btn.color, 1, cv2.LINE_AA)

            if is_active and btn.action_id.startswith("color_"):
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                cv2.circle(frame, (cx, cy), 5, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 3, (255, 255, 255), -1, cv2.LINE_AA)

        # 3. Persistent Recognized Text Banner (Centered below top bar)
        text_disp = f"TEXT: \"{self.recognized_text}\"" if self.recognized_text else "TEXT: [ Air-write letters... ]"
        clip_disp = "[ COPIED TO CLIPBOARD ]" if self.recognized_text else "[ Hold 0.8s pause to complete letter ]"
        font_scale = 0.52 if w < 900 else 0.62
        (tw1, th1), _ = cv2.getTextSize(text_disp, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        (tw2, th2), _ = cv2.getTextSize(clip_disp, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)

        card_w = min(w - 32, max(tw1 + 30, tw2 + 40, 310))
        card_h = 48
        cx = w // 2
        x1 = cx - card_w // 2
        y1 = max_bar_y + 6
        x2 = cx + card_w // 2
        y2 = y1 + card_h

        sub_card = frame[y1:y2, x1:x2]
        card_bg = np.full_like(sub_card, (15, 20, 30))
        cv2.addWeighted(card_bg, 0.88, sub_card, 0.12, 0, sub_card)
        border_col = (200, 100, 255) if self.recognized_text else (100, 100, 120)
        cv2.rectangle(frame, (x1, y1), (x2, y2), border_col, 2, cv2.LINE_AA)

        text_col = (255, 255, 255) if self.recognized_text else (180, 180, 190)
        cv2.putText(frame, text_disp, (x1 + 12, y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_col, 2 if self.recognized_text else 1, cv2.LINE_AA)
        sub_col = (0, 255, 150) if self.recognized_text else (140, 160, 180)
        cv2.putText(frame, clip_disp, (x1 + 12, y1 + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.36, sub_col, 1, cv2.LINE_AA)

        # 4. Status Badges (Bottom HUD)
        mode_badges = {
            "DRAW": ("DRAWING", (0, 220, 100)),
            "SELECT": ("SELECTION", (255, 200, 0)),
            "ERASE": ("ERASER", (0, 100, 255)),
            "IDLE": ("HOVER", (180, 180, 180)),
        }
        badge_label, badge_color = mode_badges.get(mode, ("HOVER", (180, 180, 180)))

        badge_h1, badge_h2 = h - 42, h - 14

        # Mode Badge
        m_w = min(140, int(w * 0.22))
        cv2.rectangle(frame, (12, badge_h1), (12 + m_w, badge_h2), (25, 25, 30), -1)
        cv2.rectangle(frame, (12, badge_h1), (12 + m_w, badge_h2), badge_color, 2, cv2.LINE_AA)
        cv2.circle(frame, (24, (badge_h1 + badge_h2) // 2), 4, badge_color, -1, cv2.LINE_AA)
        cv2.putText(frame, badge_label, (34, badge_h2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        # Brush size indicator
        sz_x1 = 12 + m_w + 8
        sz_w = min(100, int(w * 0.16))
        size_str = f"Size: {canvas_mgr.brush_size}px"
        cv2.rectangle(frame, (sz_x1, badge_h1), (sz_x1 + sz_w, badge_h2), (25, 25, 30), -1)
        cv2.rectangle(frame, (sz_x1, badge_h1), (sz_x1 + sz_w, badge_h2), (80, 80, 95), 1, cv2.LINE_AA)
        cv2.putText(frame, size_str, (sz_x1 + 10, badge_h2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1, cv2.LINE_AA)

        # FPS indicator
        fps_w = 80
        fps_x1 = w - fps_w - 12
        fps_text = f"{int(fps)} FPS"
        cv2.rectangle(frame, (fps_x1, badge_h1), (w - 12, badge_h2), (25, 25, 30), -1)
        cv2.rectangle(frame, (fps_x1, badge_h1), (w - 12, badge_h2), (80, 80, 95), 1, cv2.LINE_AA)
        cv2.putText(frame, fps_text, (fps_x1 + 10, badge_h2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 180), 1, cv2.LINE_AA)

        # Shortcuts helper (Only if space allows)
        help_space = fps_x1 - (sz_x1 + sz_w + 16)
        if help_space > 260:
            if w >= 950:
                help_text = "[R/Enter] OCR -> Copy | [C] Clear | [U] Undo | [S] Save | [M] View | [Q] Quit"
            else:
                help_text = "[R] Text | [C] Clear | [U] Undo | [Q] Quit"
            cv2.putText(frame, help_text, (sz_x1 + sz_w + 14, badge_h2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 165), 1, cv2.LINE_AA)

        # 5. Toast Notification
        if self.notification_text and time.time() < self.notification_expiry:
            (ntw, nth), _ = cv2.getTextSize(self.notification_text, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
            ncx, ncy = w // 2, max_bar_y + 60
            nx1 = max(12, ncx - (ntw // 2) - 14)
            ny1 = ncy - (nth // 2) - 8
            nx2 = min(w - 12, ncx + (ntw // 2) + 14)
            ny2 = ncy + (nth // 2) + 8

            sub_toast = frame[ny1:ny2, nx1:nx2]
            toast_bg = np.full_like(sub_toast, (20, 25, 30))
            cv2.addWeighted(toast_bg, 0.85, sub_toast, 0.15, 0, sub_toast)
            cv2.rectangle(frame, (nx1, ny1), (nx2, ny2), (0, 220, 120), 2, cv2.LINE_AA)
            cv2.putText(frame, self.notification_text, (nx1 + 14, ncy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)

        # 6. Reticle / Cursor
        if cursor_pt is not None:
            cx, cy = cursor_pt
            if mode == "DRAW":
                pen_col = (0, 0, 0) if canvas_mgr.is_eraser_tool else canvas_mgr.color
                cv2.circle(frame, (cx, cy), canvas_mgr.brush_size, pen_col, -1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), canvas_mgr.brush_size + 3, (255, 255, 255), 1, cv2.LINE_AA)
            elif mode == "SELECT":
                cv2.circle(frame, (cx, cy), 14, (255, 200, 0), 2, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.line(frame, (cx - 18, cy), (cx - 8, cy), (255, 200, 0), 2)
                cv2.line(frame, (cx + 8, cy), (cx + 18, cy), (255, 200, 0), 2)
                cv2.line(frame, (cx, cy - 18), (cx, cy - 8), (255, 200, 0), 2)
                cv2.line(frame, (cx, cy + 8), (cx, cy + 18), (255, 200, 0), 2)
            elif mode == "ERASE":
                cv2.circle(frame, (cx, cy), canvas_mgr.eraser_radius, (0, 100, 255), 2, cv2.LINE_AA)

        # 7. Render Hold-to-Switch Progress Ring
        if transition_info and transition_info.get("pt"):
            pt = transition_info["pt"]
            prog = transition_info.get("progress", 0.0)
            target = transition_info.get("target", "")

            cx, cy = int(pt[0]), int(pt[1])
            radius = 34

            if target == "ERASE":
                ring_col = (0, 140, 255)
                label = f"HOLD 5 FINGERS: {int(prog * 100)}%"
            elif target == "SELECT":
                ring_col = (255, 200, 0)
                label = f"HOLD 2 FINGERS: {int(prog * 100)}%"
            else:
                ring_col = (220, 220, 220)
                label = f"{int(prog * 100)}%"

            # Background circle
            cv2.circle(frame, (cx, cy), radius, (40, 40, 50), 4, cv2.LINE_AA)
            # Progress arc
            start_ang = -90
            end_ang = -90 + int(prog * 360)
            if end_ang > start_ang:
                cv2.ellipse(frame, (cx, cy), (radius, radius), 0, start_ang, end_ang, ring_col, 5, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 5, ring_col, -1, cv2.LINE_AA)

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            tx = max(10, cx - tw // 2)
            ty = max(25, cy - radius - 10)
            cv2.rectangle(frame, (tx - 6, ty - th - 4), (tx + tw + 6, ty + 4), (20, 20, 25), -1)
            cv2.rectangle(frame, (tx - 6, ty - th - 4), (tx + tw + 6, ty + 4), ring_col, 1, cv2.LINE_AA)
            cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        return frame
