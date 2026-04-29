from __future__ import annotations

import math
import time
from bisect import bisect_left
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageSequence
import pygame

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
AUDIO_FILE = "kicau-mania.mp3"
GIF_FILE = "kicau-mania.gif"

CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAME_FPS = 30
MIRROR_VIEW = False

NOSE_LANDMARK_INDEX = 1
SMOOTHING_ALPHA = 0.6

COVER_DISTANCE_NORM = 0.05
COVER_BBOX_MARGIN_NORM = 0.01

WAVE_WINDOW_FRAMES = 18
WAVE_MIN_MOVE_NORM = 0.008
WAVE_MIN_AMPLITUDE_NORM = 0.12
WAVE_MIN_SWINGS = 2
WAVE_ACTIVE_SEC = 0.7

GIF_WIDTH_RATIO = 0.28
GIF_MARGIN = 20

PALM_LANDMARKS = [0, 5, 9, 13, 17]


class AudioController:
    def __init__(self, audio_path: Path) -> None:
        self.audio_path = audio_path
        self.ready = False
        self.playing = False
        self.error: str | None = None

    def setup(self) -> bool:
        if not self.audio_path.exists():
            self.error = f"Missing audio: {self.audio_path.name}"
            return False
        try:
            pygame.mixer.init()
            pygame.mixer.music.load(str(self.audio_path))
            self.ready = True
            return True
        except Exception as exc:  # noqa: BLE001
            self.error = f"Audio init failed: {exc}"
            return False

    def start(self) -> None:
        if not self.ready or self.playing:
            return
        pygame.mixer.music.play(-1)
        self.playing = True

    def stop(self) -> None:
        if not self.ready or not self.playing:
            return
        pygame.mixer.music.stop()
        self.playing = False

    def close(self) -> None:
        if self.ready:
            pygame.mixer.quit()


class GifOverlay:
    def __init__(self, gif_path: Path, width_ratio: float) -> None:
        self.gif_path = gif_path
        self.width_ratio = width_ratio
        self.frames: list[Image.Image] = []
        self.durations: list[float] = []
        self.frame_cum: list[float] = []
        self.prepared_frames: list[np.ndarray] = []
        self.total_duration = 0.0
        self.error: str | None = None

    def load(self) -> bool:
        if not self.gif_path.exists():
            self.error = f"Missing GIF: {self.gif_path.name}"
            return False
        try:
            gif = Image.open(self.gif_path)
            for frame in ImageSequence.Iterator(gif):
                rgba = frame.convert("RGBA")
                duration_ms = frame.info.get("duration", 100)
                self.frames.append(rgba)
                self.durations.append(max(duration_ms, 20) / 1000.0)
            if not self.frames:
                self.error = "GIF has no frames"
                return False
            self.total_duration = sum(self.durations)
            running = 0.0
            for duration in self.durations:
                running += duration
                self.frame_cum.append(running)
            return True
        except Exception as exc:  # noqa: BLE001
            self.error = f"GIF load failed: {exc}"
            return False

    def prepare(self, frame_width: int) -> bool:
        if not self.frames or frame_width <= 0:
            return False
        target_w = max(64, int(frame_width * self.width_ratio))
        first = self.frames[0]
        aspect = first.height / first.width
        target_h = max(64, int(target_w * aspect))
        self.prepared_frames = []
        for frame in self.frames:
            resized = frame.resize((target_w, target_h), Image.LANCZOS)
            self.prepared_frames.append(np.array(resized))
        return True

    def get_frame(self, t: float) -> np.ndarray | None:
        if not self.prepared_frames:
            return None
        if self.total_duration <= 0:
            idx = int(t * 10) % len(self.prepared_frames)
            return self.prepared_frames[idx]
        t_mod = t % self.total_duration
        idx = bisect_left(self.frame_cum, t_mod)
        if idx >= len(self.prepared_frames):
            idx = -1
        return self.prepared_frames[idx]

    @property
    def ready(self) -> bool:
        return bool(self.prepared_frames)

    def draw(self, frame_bgr: np.ndarray, t: float, x: int, y: int) -> None:
        rgba = self.get_frame(t)
        if rgba is None:
            return
        overlay_rgba(frame_bgr, rgba, x, y)


class WaveDetector:
    def __init__(
        self,
        window_frames: int,
        min_move: float,
        min_amplitude: float,
        min_swings: int,
        active_sec: float,
    ) -> None:
        self.x_hist: deque[float] = deque(maxlen=window_frames)
        self.last_wave_time: float | None = None
        self.min_move = min_move
        self.min_amplitude = min_amplitude
        self.min_swings = min_swings
        self.active_sec = active_sec

    def reset(self) -> None:
        self.x_hist.clear()
        self.last_wave_time = None

    def update(self, x: float, t: float) -> None:
        self.x_hist.append(x)
        if len(self.x_hist) < 4:
            return

        amp = max(self.x_hist) - min(self.x_hist)
        if amp < self.min_amplitude:
            return

        direction = 0
        swings = 0
        prev_x = self.x_hist[0]
        for value in list(self.x_hist)[1:]:
            dx = value - prev_x
            prev_x = value
            if abs(dx) < self.min_move:
                continue
            new_dir = 1 if dx > 0 else -1
            if direction == 0:
                direction = new_dir
            elif new_dir != direction:
                swings += 1
                direction = new_dir

        # A wave is detected if the hand changes direction enough within the window.
        if swings >= self.min_swings:
            self.last_wave_time = t

    def is_active(self, t: float) -> bool:
        if self.last_wave_time is None:
            return False
        return (t - self.last_wave_time) <= self.active_sec


def smooth_point(prev: tuple[float, float] | None, new: tuple[float, float] | None, alpha: float) -> tuple[float, float] | None:
    if new is None:
        return None
    if prev is None:
        return new
    return (alpha * new[0] + (1 - alpha) * prev[0], alpha * new[1] + (1 - alpha) * prev[1])


def normalized_to_pixel(point: tuple[float, float] | None, width: int, height: int) -> tuple[int, int] | None:
    if point is None:
        return None
    x = int(point[0] * width)
    y = int(point[1] * height)
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    return (x, y)


def palm_center_norm(hand_landmarks: list) -> tuple[float, float] | None:
    if not hand_landmarks:
        return None
    xs = [hand_landmarks[i].x for i in PALM_LANDMARKS]
    ys = [hand_landmarks[i].y for i in PALM_LANDMARKS]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def hand_bbox_norm(hand_landmarks: list) -> tuple[float, float, float, float] | None:
    if not hand_landmarks:
        return None
    xs = [lm.x for lm in hand_landmarks]
    ys = [lm.y for lm in hand_landmarks]
    return (min(xs), min(ys), max(xs), max(ys))


def overlay_rgba(frame_bgr: np.ndarray, rgba: np.ndarray, x: int, y: int) -> None:
    fh, fw = frame_bgr.shape[:2]
    h, w = rgba.shape[:2]
    if x >= fw or y >= fh:
        return

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(fw, x + w)
    y2 = min(fh, y + h)

    if x1 >= x2 or y1 >= y2:
        return

    roi = frame_bgr[y1:y2, x1:x2].astype(np.float32)
    rgba_crop = rgba[y1 - y : y2 - y, x1 - x : x2 - x].astype(np.float32)
    bgr = rgba_crop[..., :3][..., ::-1]
    alpha = rgba_crop[..., 3:4] / 255.0

    blended = roi * (1 - alpha) + bgr * alpha
    frame_bgr[y1:y2, x1:x2] = blended.astype(np.uint8)


def draw_status(frame: np.ndarray, lines: list[str]) -> None:
    y = 24
    for line in lines:
        cv2.putText(
            frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 22


def main() -> None:
    audio_path = ASSETS_DIR / AUDIO_FILE
    gif_path = ASSETS_DIR / GIF_FILE

    audio = AudioController(audio_path)
    audio_ok = audio.setup()

    gif_overlay = GifOverlay(gif_path, GIF_WIDTH_RATIO)
    gif_ok = gif_overlay.load()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FRAME_FPS)

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    wave_detector = WaveDetector(
        window_frames=WAVE_WINDOW_FRAMES,
        min_move=WAVE_MIN_MOVE_NORM,
        min_amplitude=WAVE_MIN_AMPLITUDE_NORM,
        min_swings=WAVE_MIN_SWINGS,
        active_sec=WAVE_ACTIVE_SEC,
    )

    nose_smoothed: tuple[float, float] | None = None
    left_palm_smoothed: tuple[float, float] | None = None
    right_wrist_smoothed: tuple[float, float] | None = None

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            now = time.time()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = holistic.process(frame_rgb)
            frame_rgb.flags.writeable = True

            output = frame.copy()
            h, w = output.shape[:2]

            left_lms = results.left_hand_landmarks.landmark if results.left_hand_landmarks else None
            right_lms = results.right_hand_landmarks.landmark if results.right_hand_landmarks else None

            if results.face_landmarks and len(results.face_landmarks.landmark) > NOSE_LANDMARK_INDEX:
                nose_lm = results.face_landmarks.landmark[NOSE_LANDMARK_INDEX]
                nose_smoothed = smooth_point(nose_smoothed, (nose_lm.x, nose_lm.y), SMOOTHING_ALPHA)
            else:
                nose_smoothed = None

            if left_lms:
                palm_raw = palm_center_norm(left_lms)
                left_palm_smoothed = smooth_point(left_palm_smoothed, palm_raw, SMOOTHING_ALPHA)
            else:
                left_palm_smoothed = None

            if right_lms:
                wrist_raw = (right_lms[0].x, right_lms[0].y)
                right_wrist_smoothed = smooth_point(right_wrist_smoothed, wrist_raw, SMOOTHING_ALPHA)
                if right_wrist_smoothed is not None:
                    wave_detector.update(right_wrist_smoothed[0], now)
            else:
                right_wrist_smoothed = None
                wave_detector.reset()

            covering_nose = False
            if nose_smoothed and left_palm_smoothed and left_lms:
                bbox = hand_bbox_norm(left_lms)
                in_box = False
                if bbox:
                    min_x, min_y, max_x, max_y = bbox
                    margin = COVER_BBOX_MARGIN_NORM
                    in_box = (
                        (min_x - margin) <= nose_smoothed[0] <= (max_x + margin)
                        and (min_y - margin) <= nose_smoothed[1] <= (max_y + margin)
                    )
                dist = math.hypot(
                    nose_smoothed[0] - left_palm_smoothed[0],
                    nose_smoothed[1] - left_palm_smoothed[1],
                )
                covering_nose = in_box or (dist <= COVER_DISTANCE_NORM)

            wave_active = wave_detector.is_active(now)
            trigger_active = covering_nose and wave_active

            if trigger_active:
                audio.start()
            else:
                audio.stop()

            if gif_ok and not gif_overlay.ready:
                gif_overlay.prepare(w)

            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    output,
                    results.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                )
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    output,
                    results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                )

            nose_px = normalized_to_pixel(nose_smoothed, w, h)
            if nose_px:
                cv2.circle(output, nose_px, 6, (0, 255, 255), -1)

            if trigger_active and gif_overlay.ready:
                gif_overlay.draw(output, now, GIF_MARGIN, GIF_MARGIN)

            status_lines = [
                f"Left cover: {covering_nose}",
                f"Right wave: {wave_active}",
                f"Trigger: {trigger_active}",
                "Press Q to quit",
            ]
            if not results.face_landmarks:
                status_lines.append("Face: not detected")
            if not results.left_hand_landmarks:
                status_lines.append("Left hand: not detected")
            if not results.right_hand_landmarks:
                status_lines.append("Right hand: not detected")
            if not audio_ok:
                status_lines.append(audio.error or "Audio: unavailable")
            if not gif_ok:
                status_lines.append(gif_overlay.error or "GIF: unavailable")

            draw_status(output, status_lines)

            if MIRROR_VIEW:
                output = cv2.flip(output, 1)

            cv2.imshow("Kicau Mania", output)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    audio.close()


if __name__ == "__main__":
    main()
