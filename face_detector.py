#!/usr/bin/env python3
"""Real-time face detection from your webcam with OpenCV.

Features
--------
- Opens the webcam (or a video/image source) in real time
- Draws a green rectangle around every detected face
- Shows the current FPS and face count on the frame
- Press [s] (or Space) to save the current frame as a screenshot
- Press [q] or [ESC] to quit

Usage
-----
    python face_detector.py                # use webcam #0
    python face_detector.py --source 1     # use webcam #1
    python face_detector.py --source clip.mp4   # use a video file instead
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

# Haar cascade bundled with OpenCV — no extra downloads needed.
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
DEFAULT_SCREENSHOTS_DIR = Path(__file__).resolve().parent / "screenshots"

GREEN = (0, 255, 0)   # BGR colour for the face rectangles
WHITE = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX

EXIT_KEYS = (ord("q"), 27)          # q or ESC
SAVE_KEYS = (ord("s"), ord(" "))    # s or Space


def create_detector() -> cv2.CascadeClassifier:
    """Load the built-in Haar cascade for frontal faces."""
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    if detector.empty():
        raise RuntimeError(f"Could not load Haar cascade from: {CASCADE_PATH}")
    return detector


def smoothed_fps(prev_fps: float, dt: float, alpha: float = 0.1) -> float:
    """Exponentially weighted average so the FPS reading is stable."""
    if dt <= 0:
        return prev_fps
    return alpha * (1.0 / dt) + (1.0 - alpha) * prev_fps


def save_screenshot(frame, directory: Path) -> Path:
    """Write the current frame to ``directory`` with a timestamped name."""
    directory.mkdir(parents=True, exist_ok=True)
    # Milliseconds suffix so two quick saves can never overwrite each other.
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    path = directory / f"face_{stamp}.jpg"
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"Could not write screenshot to {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time face detection with OpenCV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index (e.g. 0) or a path to a video/image file",
    )
    parser.add_argument("--scale-factor", type=float, default=1.1,
                        help="How much the image size is reduced each pass")
    parser.add_argument("--min-neighbors", type=int, default=5,
                        help="Min. rectangles around a face to keep the match")
    parser.add_argument(        "--min-size", type=int, default=30,
        help="Smallest face size (pixels) worth detecting")  # clamped to >= 1 below
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Stop after N frames (0 = run until quit)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_SCREENSHOTS_DIR,
                        help="Folder where screenshots are saved")
    parser.add_argument("--headless", action="store_true",
                        help="Process frames without showing a window (for testing)")
    return parser.parse_args()


def main(args: argparse.Namespace) -> int:
    detector = create_detector()

    # ``--source`` may be a camera index ("0") or a file path.
    if args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))
    else:
        cap = cv2.VideoCapture(args.source)

    if not cap.isOpened():
        print(f"[ERROR] Could not open source: {args.source!r}")
        print("        If it is a webcam index, try another: --source 1")
        return 1

    if args.headless:
        print("Running headless (no display window).")
    else:
        print("Controls:  [s] save screenshot   [q / ESC] quit")

    min_size = max(1, args.min_size)   # OpenCV rejects a minSize of 0

    fps = 0.0
    last_time = time.perf_counter()
    notice = None          # name of the last screenshot, shown briefly
    notice_until = 0.0
    frames = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Could not read a frame — stopping.")
                break

            # Convert to grayscale: Haar cascades work on intensity only.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = detector.detectMultiScale(
                gray,
                scaleFactor=args.scale_factor,
                minNeighbors=args.min_neighbors,
                minSize=(min_size, min_size),
            )

            # Draw a green rectangle around every detected face.
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), GREEN, 2)

            # FPS + face count + "saved" notice.
            now = time.perf_counter()
            fps = smoothed_fps(fps, now - last_time)
            last_time = now

            cv2.putText(frame, f"FPS: {fps:5.1f}", (10, 30), FONT, 0.7, GREEN, 2)
            cv2.putText(frame, f"Faces: {len(faces)}", (10, 60), FONT, 0.6, WHITE, 2)
            if notice is not None and now < notice_until:
                cv2.putText(frame, f"Saved: {notice}", (10, 90), FONT, 0.6, WHITE, 2)

            if not args.headless:
                cv2.imshow("Face Detection", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in EXIT_KEYS:
                    break
                if key in SAVE_KEYS:
                    path = save_screenshot(frame, args.out_dir)
                    notice = path.name
                    notice_until = time.perf_counter() + 2.0
                    print(f"[SAVED] {path}")

            frames += 1
            if args.max_frames and frames >= args.max_frames:
                break
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()

    print(f"Done - processed {frames} frames.")
    return 0


if __name__ == "__main__":
    sys.exit(main(parse_args()))
