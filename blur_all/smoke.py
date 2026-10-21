"""A packaged application self-check that does not need network or user input."""

from pathlib import Path
import tempfile
import time

import cv2
import numpy as np


def smoke_test(window, app):
    with tempfile.TemporaryDirectory(prefix="blur-all-smoke-") as directory:
        directory = Path(directory)
        source, output = directory / "source.avi", directory / "out.mp4"
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 120))
        if not writer.isOpened():
            raise RuntimeError("Smoke test cannot create video")
        frame = np.random.default_rng(7).integers(0, 256, (120, 160, 3), dtype=np.uint8)
        for _ in range(3):
            writer.write(frame)
        writer.release()
        if not window.load_video(source):
            raise RuntimeError("Desktop cannot load video")
        window.preview.regions.append((0.1, 0.1, 0.5, 0.5))
        window.start_export(output)
        deadline = time.monotonic() + 60
        while window.worker is not None:
            app.processEvents()
            if time.monotonic() > deadline:
                window.cancel_export()
                # Keep the worker alive until it has cleaned up its child processes.
                while window.worker is not None:
                    app.processEvents()
                    time.sleep(0.01)
                raise RuntimeError("Desktop export timed out")
            time.sleep(0.01)
        if not output.exists() or window.progress.value() != 100:
            raise RuntimeError("Desktop export failed")
        capture = cv2.VideoCapture(str(output))
        try:
            count = 0
            while True:
                ok, result = capture.read()
                if not ok:
                    break
                count += 1
                if result.shape != frame.shape:
                    raise RuntimeError("Unexpected output dimensions")
            if count != 3:
                raise RuntimeError("Incomplete output")
        finally:
            capture.release()
        # Release the source before Windows tries to remove its temporary directory.
        if window.capture is not None:
            window.capture.release()
            window.capture = None
