import os
from pathlib import Path
import sys

import cv2
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def face():
    image = cv2.imread(str(Path(__file__).parent / "assets" / "astronaut.jpg"))
    assert image is not None
    return image


@pytest.fixture
def video(tmp_path, face):
    path = tmp_path / "source.avi"
    height, width = face.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (width, height))
    assert writer.isOpened()
    for _ in range(12):
        writer.write(face)
    writer.release()
    return path
