"""A bounded background preview queue: only the newest pending frame is decoded."""

from dataclasses import dataclass
from pathlib import Path
import threading

import cv2
from PySide6.QtCore import QThread, Signal

from blur_all.core import FaceDetector, FaceHold, Settings, VideoError, blur_frame, open_video


@dataclass
class PreviewFrame:
    index: int
    generation: int
    original: object
    processed: object
    boxes: list
    regions: list


class PreviewWorker(QThread):
    ready = Signal(object)
    failed = Signal(int, str)

    def __init__(self, source: Path, fps: float):
        super().__init__()
        self.source, self.fps = source, fps
        self._condition = threading.Condition()
        self._pending = None
        self._stopping = False

    def request(self, index: int, settings: Settings, generation: int):
        with self._condition:
            self._pending = (index, settings, generation)
            self._condition.notify()

    def stop(self):
        with self._condition:
            self._stopping = True
            self._pending = None
            self._condition.notify()

    def run(self):
        capture = None
        generation = 0
        try:
            capture = open_video(self.source)
            detector = None
            detector_key = None
            cached_index, frame, boxes = -1, None, []
            hold = FaceHold(max(1, round(self.fps * 0.35)))
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._stopping or self._pending is not None)
                    if self._stopping:
                        return
                    index, settings, generation = self._pending
                    self._pending = None
                try:
                    key = (settings.confidence, settings.detection_size)
                    new_detector = key != detector_key
                    if new_detector:
                        detector = FaceDetector(*key)
                        detector_key = key
                    if index != cached_index:
                        if index != cached_index + 1:
                            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                            hold = FaceHold(max(1, round(self.fps * 0.35)))
                        ok, frame = capture.read()
                        if not ok:
                            raise VideoError("Could not decode this frame. Try another position.")
                        boxes = hold.update(detector.detect(frame), frame)
                        cached_index = index
                    elif new_detector:
                        hold = FaceHold(max(1, round(self.fps * 0.35)))
                        boxes = hold.update(detector.detect(frame), frame)
                    processed = blur_frame(frame, boxes, settings)
                    height, width = frame.shape[:2]
                    scale = min(1.0, 1280 / max(width, height))
                    size = (max(1, round(width * scale)), max(1, round(height * scale)))
                    self.ready.emit(PreviewFrame(
                        index, generation, cv2.resize(frame, size), cv2.resize(processed, size),
                        boxes, [(x / width, y / height, w / width, h / height) for x, y, w, h in boxes],
                    ))
                except Exception as exc:
                    cached_index, frame = -1, None
                    detector_key = None
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    hold = FaceHold(max(1, round(self.fps * 0.35)))
                    self.failed.emit(generation, str(exc))
        except Exception as exc:
            self.failed.emit(generation, str(exc))
        finally:
            if capture is not None:
                capture.release()
