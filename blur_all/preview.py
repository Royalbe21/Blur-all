"""A bounded background preview queue: only the newest pending frame is decoded."""

from dataclasses import dataclass
from pathlib import Path
import threading

import cv2
from PySide6.QtCore import QThread, Signal

from blur_all.core import FaceAnalyzer, Settings, VideoError, blur_frame, open_video


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
            analyzer = None
            analyzer_key = None
            cached_index, frame, boxes = -1, None, []
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._stopping or self._pending is not None)
                    if self._stopping:
                        return
                    index, settings, generation = self._pending
                    self._pending = None
                try:
                    key = (settings.confidence, settings.detection_size, settings.detection_interval)
                    reset = key != analyzer_key or index not in (cached_index, cached_index + 1)
                    if reset:
                        analyzer = FaceAnalyzer(key[0], key[1], self.fps, key[2])
                        analyzer_key = key
                    if index != cached_index:
                        if index != cached_index + 1:
                            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                        ok, frame = capture.read()
                        if not ok:
                            raise VideoError("Could not decode this frame. Try another position.")
                        boxes = analyzer.analyze(frame).boxes
                        cached_index = index
                    elif reset:
                        boxes = analyzer.analyze(frame).boxes
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
                    analyzer_key = None
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.failed.emit(generation, str(exc))
        except Exception as exc:
            self.failed.emit(generation, str(exc))
        finally:
            if capture is not None:
                capture.release()
