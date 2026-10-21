"""Face detection and video export, independent of the desktop interface."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import ExitStack
import hashlib
import math
import os
from pathlib import Path
import subprocess
import tempfile
import threading
from typing import Callable

import cv2
import imageio_ffmpeg
import numpy as np

MODEL_PATH = Path(__file__).parent / "assets" / "face_detection_yunet_2023mar.onnx"
MODEL_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
Box = tuple[int, int, int, int]  # x, y, width, height
Region = tuple[float, float, float, float]  # normalized x, y, width, height


class VideoError(RuntimeError):
    """An input, detector, or encoder could not complete the operation."""


class Cancelled(Exception):
    """The user cancelled export; no final output is produced."""


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frames: int

    @property
    def duration(self) -> float:
        return self.frames / self.fps


@dataclass(frozen=True)
class Settings:
    confidence: float = 0.6
    detection_size: int = 960
    strength: int = 80
    margin: float = 0.3
    keep_audio: bool = True
    manual_regions: tuple[Region, ...] = ()
    effect: str = "blur"


@dataclass(frozen=True)
class ExportResult:
    path: Path
    frames: int
    frames_with_faces: int


def open_video(path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise VideoError("Cannot open this video. Try converting it to MP4 (H.264) first.")
    return capture


def video_info(capture: cv2.VideoCapture) -> VideoInfo:
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if width < 1 or height < 1 or not math.isfinite(fps) or not 0 < fps <= 1000:
        raise VideoError("The video has invalid dimensions or frame rate.")
    return VideoInfo(width, height, fps, max(0, frames))


class FaceDetector:
    def __init__(self, confidence: float = 0.6, max_size: int = 960):
        if not MODEL_PATH.exists() or hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() != MODEL_SHA256:
            raise VideoError("The bundled face detector is missing or damaged. Reinstall Blur All.")
        self.max_size = max_size
        self.detector = cv2.FaceDetectorYN.create(
            str(MODEL_PATH), "", (320, 320), confidence, 0.3, 5000
        )

    def detect(self, frame: np.ndarray) -> list[Box]:
        height, width = frame.shape[:2]
        scale = min(1.0, self.max_size / max(width, height))
        small = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))))
        sh, sw = small.shape[:2]
        # Pad to strides of 32; keep coordinates relative to the unpadded image.
        padded = cv2.copyMakeBorder(small, 0, (-sh) % 32, 0, (-sw) % 32, cv2.BORDER_CONSTANT)
        self.detector.setInputSize((padded.shape[1], padded.shape[0]))
        _, faces = self.detector.detect(padded)
        if faces is None:
            return []
        result = []
        for face in faces:
            x, y, w, h = face[:4]
            left, top = max(0, int(x / scale)), max(0, int(y / scale))
            right, bottom = min(width, math.ceil((x + w) / scale)), min(height, math.ceil((y + h) / scale))
            if right > left and bottom > top:
                result.append((left, top, right - left, bottom - top))
        return result


def overlap(a: Box, b: Box) -> float:
    x, y = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    area = max(0, right - x) * max(0, bottom - y)
    return area / max(1, a[2] * a[3] + b[2] * b[3] - area)


class FaceHold:
    """Follow recent boxes with optical flow during short detection dropouts."""

    def __init__(self, frames: int):
        self.frames = frames
        self.previous: list[tuple[Box, int]] = []
        self.last_thumbnail: np.ndarray | None = None
        self.last_gray: np.ndarray | None = None

    def _move_boxes(self, gray: np.ndarray, scale: float, width: int, height: int):
        if self.last_gray is None or self.last_gray.shape != gray.shape or not self.previous:
            return self.previous
        mask = np.zeros_like(gray)
        for (x, y, w, h), _ in self.previous:
            left, top = max(0, round(x * scale)), max(0, round(y * scale))
            right, bottom = min(gray.shape[1], round((x + w) * scale)), min(gray.shape[0], round((y + h) * scale))
            mask[top:bottom, left:right] = 255
        points = cv2.goodFeaturesToTrack(self.last_gray, 200, 0.01, 3, mask=mask)
        if points is None:
            return self.previous
        moved, status, _ = cv2.calcOpticalFlowPyrLK(self.last_gray, gray, points, None)
        if moved is None or status is None:
            return self.previous
        returned, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, self.last_gray, moved, None)
        if returned is None or back_status is None:
            return self.previous
        before, after = points.reshape(-1, 2) / scale, moved.reshape(-1, 2) / scale
        valid = (status.ravel() == 1) & (back_status.ravel() == 1)
        valid &= np.linalg.norm(points.reshape(-1, 2) - returned.reshape(-1, 2), axis=1) < 1.5
        result = []
        for box, ttl in self.previous:
            x, y, w, h = box
            inside = valid & (before[:, 0] >= x) & (before[:, 0] <= x + w)
            inside &= (before[:, 1] >= y) & (before[:, 1] <= y + h)
            if inside.sum() >= 4:
                shifts = after[inside] - before[inside]
                dx, dy = np.median(shifts, axis=0)
                # Reject implausible jumps; short-term flow is only a fallback.
                if abs(dx) <= max(20, w * 0.75) and abs(dy) <= max(20, h * 0.75):
                    left, top = max(0, round(x + dx)), max(0, round(y + dy))
                    right, bottom = min(width, round(x + w + dx)), min(height, round(y + h + dy))
                    if right > left and bottom > top:
                        box = (left, top, right - left, bottom - top)
            result.append((box, ttl))
        return result

    def update(self, boxes: list[Box], frame: np.ndarray) -> list[Box]:
        height, width = frame.shape[:2]
        scale = min(1.0, 640 / max(width, height))
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (max(1, round(width * scale)), max(1, round(height * scale))))
        thumbnail = cv2.resize(gray, (32, 32))
        if self.last_thumbnail is not None:
            change = np.abs(thumbnail.astype(float) - self.last_thumbnail).mean()
            if change > 35:  # Avoid carrying old boxes over a hard scene cut.
                self.previous.clear()
        self.last_thumbnail = thumbnail
        predicted = self._move_boxes(gray, scale, width, height)
        self.last_gray = gray
        retained = [
            (old, ttl - 1) for old, ttl in predicted
            if ttl > 1 and not any(overlap(old, new) > 0.2 for new in boxes)
        ]
        self.previous = [(box, self.frames) for box in boxes] + retained
        return [box for box, _ in self.previous]


def blur_frame(frame: np.ndarray, boxes: list[Box], settings: Settings) -> np.ndarray:
    result = frame.copy()
    height, width = frame.shape[:2]
    areas = []
    for x, y, w, h in boxes:
        dx, dy = round(w * settings.margin), round(h * settings.margin)
        areas.append((x - dx, y - dy, x + w + dx, y + h + dy))
    for x, y, w, h in settings.manual_regions:
        areas.append((round(x * width), round(y * height), round((x + w) * width), round((y + h) * height)))
    for left, top, right, bottom in areas:
        left, top = max(0, left), max(0, top)
        right, bottom = min(width, right), min(height, bottom)
        if right <= left or bottom <= top:
            continue
        roi = result[top:bottom, left:right]
        if settings.effect == "solid":
            result[top:bottom, left:right] = (18, 18, 18)
            continue
        # Destroy detail before smoothing; a larger strength retains fewer samples.
        samples = max(2, round(22 - settings.strength * 0.2))
        tiny = cv2.resize(roi, (min(samples, right - left), min(samples, bottom - top)), interpolation=cv2.INTER_AREA)
        interpolation = cv2.INTER_NEAREST if settings.effect == "pixelate" else cv2.INTER_LINEAR
        obscured = cv2.resize(tiny, (right - left, bottom - top), interpolation=interpolation)
        sigma = max(3, min(right - left, bottom - top) * 0.12)
        result[top:bottom, left:right] = obscured if settings.effect == "pixelate" else cv2.GaussianBlur(obscured, (0, 0), sigma)
    return result


def ffmpeg_executable() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def _spawn(command: list[str], **kwargs) -> subprocess.Popen:
    return subprocess.Popen(
        command, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0, **kwargs
    )


def _check_cancel(cancel: threading.Event) -> None:
    if cancel.is_set():
        raise Cancelled()


def _stop_process(process: subprocess.Popen | None) -> None:
    if process is None:
        return
    if process.poll() is None:
        process.kill()
    process.wait()
    if process.stdin is not None and not process.stdin.closed:
        try:
            process.stdin.close()
        except BrokenPipeError:
            pass


def export_video(
    source: Path,
    destination: Path,
    settings: Settings,
    cancel: threading.Event | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> ExportResult:
    source, destination = source.resolve(), destination.resolve()
    cancel = cancel if cancel is not None else threading.Event()
    report = progress or (lambda current, total, stage: None)
    if source == destination:
        raise VideoError("Choose a different output file to keep your original safe.")
    if destination.exists():
        raise VideoError("The output file already exists. Choose a new filename.")
    if destination.suffix.lower() != ".mp4":
        raise VideoError("Choose an output filename ending in .mp4.")
    if not destination.parent.is_dir():
        raise VideoError("The output folder does not exist.")
    _check_cancel(cancel)
    detector = FaceDetector(settings.confidence, settings.detection_size)
    capture = open_video(source)
    encoder = None
    count = face_frames = 0
    try:
        info = video_info(capture)
        tracker = FaceHold(max(1, round(info.fps * 0.35)))
        ffmpeg = ffmpeg_executable()
        with ExitStack() as stack:
            directory = stack.enter_context(tempfile.TemporaryDirectory(prefix=".blur-all-", dir=destination.parent))
            # Kill children before removing their temporary files (required on Windows).
            stack.callback(lambda: _stop_process(encoder))
            temp = Path(directory)
            silent, final = temp / "video.mp4", temp / "final.mp4"
            with (temp / "encoder.log").open("w+b") as log:
                encoder = _spawn([
                    ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
                    "-pix_fmt", "bgr24", "-s", f"{info.width}x{info.height}",
                    "-r", str(info.fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
                    "-preset", "fast", "-crf", "18", "-threads", "2",
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(silent)
                ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
                try:
                    while True:
                        _check_cancel(cancel)
                        ok, frame = capture.read()
                        if not ok:
                            break
                        if frame.shape[:2] != (info.height, info.width):
                            raise VideoError("The video changes resolution midstream. Convert it to a fixed resolution first.")
                        boxes = detector.detect(frame)
                        face_frames += bool(boxes)
                        blurred = blur_frame(frame, tracker.update(boxes, frame), settings)
                        encoder.stdin.write(blurred.tobytes())
                        count += 1
                        if count == 1 or count % max(1, round(info.fps / 5)) == 0:
                            report(count, info.frames, "Blurring faces")
                    encoder.stdin.close()
                    while encoder.poll() is None:
                        _check_cancel(cancel)
                        cancel.wait(0.05)
                except BrokenPipeError as exc:
                    log.flush()
                    log.seek(0)
                    raise VideoError("Video encoding failed: " + log.read().decode(errors="replace")[-2000:]) from exc
                if encoder.returncode:
                    log.seek(0)
                    raise VideoError("Video encoding failed: " + log.read().decode(errors="replace")[-2000:])
            if count == 0:
                raise VideoError("No video frames could be decoded.")
            if info.frames and count < info.frames - max(2, round(info.frames * 0.01)):
                raise VideoError(f"Decoding stopped early ({count} of {info.frames} frames). No incomplete video was saved.")
            _check_cancel(cancel)
            report(count, max(count, info.frames), "Finishing audio and MP4")
            command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(silent)]
            if settings.keep_audio:
                command += ["-i", str(source), "-map", "0:v:0", "-map", "1:a?", "-c:a", "aac", "-b:a", "192k"]
            else:
                command += ["-map", "0:v:0", "-an"]
            command += ["-c:v", "copy", "-map_metadata", "-1", "-map_chapters", "-1", "-t", str(count / info.fps), "-movflags", "+faststart", str(final)]
            with (temp / "mux.log").open("w+b") as log:
                encoder = _spawn(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=log)
                while encoder.poll() is None:
                    _check_cancel(cancel)
                    cancel.wait(0.05)
                if encoder.returncode:
                    log.seek(0)
                    raise VideoError("Audio export failed: " + log.read().decode(errors="replace")[-2000:])
            _check_cancel(cancel)
            # Reserve without overwriting, then replace atomically on the same filesystem.
            with destination.open("xb"):
                pass
            try:
                os.replace(final, destination)
            except BaseException:
                destination.unlink(missing_ok=True)
                raise
            report(count, count, "Complete")
            return ExportResult(destination, count, face_frames)
    finally:
        capture.release()
        _stop_process(encoder)
