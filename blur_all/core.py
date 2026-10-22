"""Face detection and video export, independent of the desktop interface."""

from __future__ import annotations

from dataclasses import dataclass, field
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

from blur_all.tracking import FaceTracker, overlap
from blur_all.masks import FaceRegion, face_polygon, polygon_mask

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
    margin: float = 0.15
    keep_audio: bool = True
    manual_regions: tuple[Region, ...] = ()
    effect: str = "blur"
    detection_interval: int = 1
    mask_shape: str = "face"
    edge_softness: float = 0.08
    output_format: str = "mp4"
    export_quality: str = "high"
    resolution: str = "source"
    audio_bitrate: int = 192


@dataclass(frozen=True)
class ExportResult:
    path: Path
    frames: int
    frames_with_faces: int
    detector_frames: int = 0


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
        self.last_landmarks = []
        self.detector = cv2.FaceDetectorYN.create(
            str(MODEL_PATH), "", (320, 320), confidence, 0.3, 5000
        )

    def detect(self, frame: np.ndarray) -> list[Box]:
        self.last_landmarks = []
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
                self.last_landmarks.append((face[4:14].reshape(5, 2) / scale).astype(np.float32))
        return result


# Kept for callers using the original tracker name.
FaceHold = FaceTracker


@dataclass(frozen=True)
class FrameAnalysis:
    boxes: list[Box]
    did_detect: bool
    detected_faces: int
    faces: list[FaceRegion] = field(default_factory=list)


class FaceAnalyzer:
    """Shared preview/export detection scheduling with same-frame recovery."""

    def __init__(self, confidence=0.6, max_size=960, fps=30, detection_interval=1):
        if detection_interval not in (1, 2, 4):
            raise ValueError("Detection interval must be 1, 2, or 4 frames.")
        self.detector = FaceDetector(confidence, max_size)
        self.tracker = FaceTracker(max(2, round(fps * 0.5)))
        self.interval = detection_interval
        self.index = 0
        self.detector_frames = 0

    def analyze(self, frame: np.ndarray) -> FrameAnalysis:
        prepared = self.tracker.prepare(frame)
        should_detect = self.index % self.interval == 0 or not self.tracker.tracks or prepared.scene_cut
        detections = self.detector.detect(frame) if should_detect else None
        landmarks = getattr(self.detector, "last_landmarks", None) if should_detect else None
        boxes = self.tracker.update(detections, prepared=prepared, landmarks=landmarks)
        if not should_detect and self.tracker.needs_detection:
            # Do not wait for the scheduled check when motion cannot be trusted.
            detections = self.detector.detect(frame)
            boxes = self.tracker.correct(detections, getattr(self.detector, "last_landmarks", None))
            should_detect = True
        self.index += 1
        self.detector_frames += int(should_detect)
        return FrameAnalysis(boxes, should_detect, len(detections) if detections is not None else 0, self.tracker.regions())


def apply_effect(roi: np.ndarray, settings: Settings) -> np.ndarray:
    if settings.effect == "solid":
        return np.full_like(roi, 18)
    if settings.effect == "distort":
        h, w = roi.shape[:2]
        x, y = np.meshgrid(np.linspace(-1, 1, w, dtype=np.float32), np.linspace(-1, 1, h, dtype=np.float32))
        falloff = np.maximum(0, 1 - x * x - y * y) ** 2
        amount = settings.strength / 100
        angle = 0.45 * amount * falloff
        cosine, sine = np.cos(angle), np.sin(angle)
        warped_x = (x * cosine - y * sine) * (1 + 0.35 * amount * falloff)
        warped_y = (x * sine + y * cosine) * (1 - 0.15 * amount * falloff)
        map_x = ((warped_x + 1) * (w - 1) / 2).astype(np.float32)
        map_y = ((warped_y + 1) * (h - 1) / 2).astype(np.float32)
        return cv2.remap(roi, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    if settings.effect == "pixelate":
        h, w = roi.shape[:2]
        block = max(3, round(min(w, h) * (0.04 + settings.strength * 0.003)))
        tiny = cv2.resize(roi, (max(1, math.ceil(w / block)), max(1, math.ceil(h / block))), interpolation=cv2.INTER_AREA)
        return cv2.resize(tiny, (w, h), interpolation=cv2.INTER_NEAREST)
    # Destroy detail before smoothing; a larger strength retains fewer samples.
    samples = max(2, round(22 - settings.strength * 0.2))
    tiny = cv2.resize(roi, (min(samples, roi.shape[1]), min(samples, roi.shape[0])), interpolation=cv2.INTER_AREA)
    obscured = cv2.resize(tiny, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_LINEAR)
    sigma = max(3, min(roi.shape[1], roi.shape[0]) * 0.12)
    return cv2.GaussianBlur(obscured, (0, 0), sigma)



def blur_frame(frame: np.ndarray, boxes: list[Box], settings: Settings, *, faces=None) -> np.ndarray:
    result = frame.copy()
    height, width = frame.shape[:2]
    regions = faces if faces is not None else [FaceRegion(box) for box in boxes]
    for face in regions:
        polygon = face_polygon(face, settings.margin, settings.mask_shape)
        feather = min(face.box[2:]) * settings.edge_softness
        masked = polygon_mask(polygon, width, height, feather)
        if masked is None:
            continue
        (left, top, right, bottom), alpha = masked
        roi = result[top:bottom, left:right]
        treated = apply_effect(roi, settings)
        alpha = alpha[..., None]
        result[top:bottom, left:right] = np.clip(treated.astype(np.float32) * alpha + roi.astype(np.float32) * (1 - alpha), 0, 255).astype(np.uint8)
    # Drawn areas retain their exact rectangular bounds; contour fitting is for
    # automatic faces. They can also cover non-face objects such as name badges.
    for x, y, w, h in settings.manual_regions:
        left, top = max(0, round(x * width)), max(0, round(y * height))
        right, bottom = min(width, round((x + w) * width)), min(height, round((y + h) * height))
        if right > left and bottom > top:
            result[top:bottom, left:right] = apply_effect(result[top:bottom, left:right], settings)
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


QUALITY_PRESETS = {"high": 18, "balanced": 22, "small": 26}
OUTPUT_FORMATS = ("mp4", "mov", "mkv")


def export_dimensions(width: int, height: int, resolution: str) -> tuple[int, int]:
    presets = {"source": None, "1080p": (1920, 1080), "720p": (1280, 720), "480p": (854, 480)}
    if resolution not in presets:
        raise VideoError("Choose a supported output resolution.")
    size = presets[resolution]
    scale = 1.0
    if size is not None:
        max_w, max_h = size if width >= height else size[::-1]
        scale = min(1.0, max_w / width, max_h / height)
    w, h = max(1, round(width * scale)), max(1, round(height * scale))
    if scale < 1:
        w, h = max(2, w - w % 2), max(2, h - h % 2)
    return w + w % 2, h + h % 2


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
    if settings.output_format not in OUTPUT_FORMATS or destination.suffix.lower() != "." + settings.output_format:
        raise VideoError("Choose an output filename matching the selected MP4, MOV, or MKV format.")
    if settings.export_quality not in QUALITY_PRESETS:
        raise VideoError("Choose a supported quality preset.")
    if settings.audio_bitrate not in (128, 192, 256):
        raise VideoError("Choose a supported AAC audio bitrate.")
    if not destination.parent.is_dir():
        raise VideoError("The output folder does not exist.")
    _check_cancel(cancel)
    capture = open_video(source)
    encoder = None
    count = face_frames = 0
    try:
        info = video_info(capture)
        output_w, output_h = export_dimensions(info.width, info.height, settings.resolution)
        video_filter = "pad=ceil(iw/2)*2:ceil(ih/2)*2"
        if (output_w, output_h) != (info.width + info.width % 2, info.height + info.height % 2):
            video_filter = f"scale={output_w}:{output_h}:flags=lanczos,setsar=1"
        analyzer = FaceAnalyzer(settings.confidence, settings.detection_size, info.fps, settings.detection_interval)
        ffmpeg = ffmpeg_executable()
        with ExitStack() as stack:
            directory = stack.enter_context(tempfile.TemporaryDirectory(prefix=".blur-all-", dir=destination.parent))
            # Kill children before removing their temporary files (required on Windows).
            stack.callback(lambda: _stop_process(encoder))
            temp = Path(directory)
            silent, final = temp / "video.mp4", temp / ("final." + settings.output_format)
            with (temp / "encoder.log").open("w+b") as log:
                encoder = _spawn([
                    ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
                    "-pix_fmt", "bgr24", "-s", f"{info.width}x{info.height}",
                    "-r", str(info.fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
                    "-preset", "fast", "-crf", str(QUALITY_PRESETS[settings.export_quality]), "-threads", "2",
                    "-vf", video_filter, "-pix_fmt", "yuv420p",
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
                        analysis = analyzer.analyze(frame)
                        face_frames += bool(analysis.boxes)
                        blurred = blur_frame(frame, analysis.boxes, settings, faces=analysis.faces)
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
            report(count, max(count, info.frames), "Finishing audio and " + settings.output_format.upper())
            command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(silent)]
            if settings.keep_audio:
                command += ["-i", str(source), "-map", "0:v:0", "-map", "1:a?", "-c:a", "aac", "-b:a", f"{settings.audio_bitrate}k"]
            else:
                command += ["-map", "0:v:0", "-an"]
            command += ["-c:v", "copy", "-map_metadata", "-1", "-map_chapters", "-1", "-t", str(count / info.fps)]
            if settings.output_format in ("mp4", "mov"):
                command += ["-movflags", "+faststart"]
            command.append(str(final))
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
            return ExportResult(destination, count, face_frames, analyzer.detector_frames)
    finally:
        capture.release()
        _stop_process(encoder)
