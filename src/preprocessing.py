"""Extract face-cropped frames from raw videos for ResNet18 fine-tuning.

For each video in data/raw_videos/<class>/<video>.MOV, samples frames from
only the last DURATION_SEC seconds at FPS_EXTRACT frames per second, detects
the face with MediaPipe, crops a square region around it (with padding),
resizes to OUTPUT_SIZE x OUTPUT_SIZE, and saves to
data/frames/<class>/<video_stem>/frame_####.jpg.

If a frame has no detected face, the last known face box is reused. Frames
before the first successful detection in a video are skipped.
"""

import os

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions

RAW_VIDEOS_DIR = "data/raw_videos"
FRAMES_DIR = "data/frames"
FACE_MODEL_PATH = "models/blaze_face_short_range.tflite"

DURATION_SEC = 5
FPS_EXTRACT = 20
PADDING_RATIO = 0.4
OUTPUT_SIZE = 224

VIDEO_EXTENSIONS = (".mov", ".mp4", ".avi", ".mkv")


def _square_and_pad_box(x1, y1, x2, y2, frame_w, frame_h, padding_ratio):
    box_w = x2 - x1
    box_h = y2 - y1

    cx = x1 + box_w / 2
    cy = y1 + box_h / 2

    side = max(box_w, box_h) * (1 + 2 * padding_ratio)

    half = side / 2
    new_x1 = int(round(cx - half))
    new_y1 = int(round(cy - half))
    new_x2 = int(round(cx + half))
    new_y2 = int(round(cy + half))

    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    new_x2 = min(frame_w, new_x2)
    new_y2 = min(frame_h, new_y2)

    return new_x1, new_y1, new_x2, new_y2


def _detect_face_box(face_detector, frame_bgr):
    frame_h, frame_w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = face_detector.detect(mp_image)

    if not result.detections:
        return None

    best_detection = max(
        result.detections,
        key=lambda d: d.bounding_box.width * d.bounding_box.height,
    )
    box = best_detection.bounding_box

    x1 = box.origin_x
    y1 = box.origin_y
    x2 = box.origin_x + box.width
    y2 = box.origin_y + box.height

    return _square_and_pad_box(x1, y1, x2, y2, frame_w, frame_h, PADDING_RATIO)


def _sample_timestamps_ms(video_duration_sec):
    start_sec = max(0.0, video_duration_sec - DURATION_SEC)
    window_sec = video_duration_sec - start_sec

    num_frames = int(round(window_sec * FPS_EXTRACT))
    if num_frames <= 0:
        return []

    step_ms = 1000.0 / FPS_EXTRACT
    return [start_sec * 1000.0 + i * step_ms for i in range(num_frames)]


def process_video(video_path, output_dir, face_detector):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if not fps or fps <= 0 or frame_count <= 0:
        print(f"  Invalid video metadata, skipping: {video_path}")
        cap.release()
        return

    duration_sec = frame_count / fps
    timestamps_ms = _sample_timestamps_ms(duration_sec)

    os.makedirs(output_dir, exist_ok=True)

    last_box = None
    saved_count = 0

    for timestamp_ms in timestamps_ms:
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
        success, frame = cap.read()
        if not success:
            continue

        box = _detect_face_box(face_detector, frame)
        if box is None:
            box = last_box
        else:
            last_box = box

        if box is None:
            continue

        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            continue

        face_crop = frame[y1:y2, x1:x2]
        face_resized = cv2.resize(
            face_crop, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_AREA
        )

        saved_count += 1
        out_path = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
        cv2.imwrite(out_path, face_resized)

    cap.release()
    print(f"  Saved {saved_count} frames to {output_dir}")


def main():
    options = mp_vision.FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH),
        running_mode=mp_vision.RunningMode.IMAGE,
        min_detection_confidence=0.5,
    )

    with mp_vision.FaceDetector.create_from_options(options) as face_detector:
        for class_name in sorted(os.listdir(RAW_VIDEOS_DIR)):
            class_dir = os.path.join(RAW_VIDEOS_DIR, class_name)
            if not os.path.isdir(class_dir):
                continue

            for video_name in sorted(os.listdir(class_dir)):
                if not video_name.lower().endswith(VIDEO_EXTENSIONS):
                    continue

                video_path = os.path.join(class_dir, video_name)
                video_stem = os.path.splitext(video_name)[0]
                output_dir = os.path.join(FRAMES_DIR, class_name, video_stem)

                if os.path.isdir(output_dir) and os.listdir(output_dir):
                    print(f"Skipping {video_path} (frames already exist)")
                    continue

                print(f"Processing {video_path}")
                process_video(video_path, output_dir, face_detector)


if __name__ == "__main__":
    main()
