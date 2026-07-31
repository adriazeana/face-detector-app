# Face Detection

Real-time face detection from your webcam, built with OpenCV (Python).

![tech](https://img.shields.io/badge/OpenCV-4.8+-green) ![lang](https://img.shields.io/badge/Python-3.9+-blue)

## Features

- ✅ Live webcam feed with real-time face detection
- ✅ Green rectangle drawn around every detected face
- ✅ On-screen **FPS** counter (and face count)
- ✅ Save screenshots with a single keypress
- ✅ Works with video files too (`--source clip.mp4`)

## Installation

```bash
cd face_detection
pip install -r requirements.txt
```

## Run

```bash
python face_detector.py
```

Point the camera at yourself (or hold up a photo) and watch the green boxes appear.

### Controls

| Key          | Action                    |
|--------------|---------------------------|
| `s` or `Space` | Save a screenshot        |
| `q` or `Esc`   | Quit                     |

Screenshots are saved to `./screenshots/` as timestamped `.jpg` files.

### Options

```bash
python face_detector.py --source 1            # use webcam #1 instead of #0
python face_detector.py --source video.mp4    # detect faces in a video file
python face_detector.py --min-neighbors 8     # fewer false positives
python face_detector.py --min-size 100        # ignore tiny faces
```

Run `python face_detector.py --help` for the full list.

## How it works

1. **OpenCV** grabs frames from your webcam via `VideoCapture`.
2. Each frame is converted to **grayscale** and passed to the bundled **Haar cascade** (`haarcascade_frontalface_default.xml`).
3. `detectMultiScale` returns bounding boxes for any faces found — we draw them in **green**.
4. A rolling average of frame times gives a stable **FPS** readout.
5. Pressing `s` writes the current frame to the `screenshots/` folder.

## Troubleshooting

- **"Could not open source"** — the webcam is busy or the index is wrong. Try `--source 1`, `--source 2`, etc. On laptops the built-in camera is usually `0`.
- **No faces detected** — make sure your face is well-lit and facing the camera. Haar cascades miss profile/side views; those would need a DNN model (e.g. `res10_300x300_ssd`).
- **Slow FPS** — reduce the capture resolution in the script or raise `--min-size` so the detector searches less.
