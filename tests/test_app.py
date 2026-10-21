import time

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from blur_all.app import MainWindow, STYLE


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    application.setStyleSheet(STYLE)
    return application


@pytest.fixture(autouse=True)
def no_blocking_dialogs(monkeypatch):
    def unexpected_dialog(*args):
        pytest.fail(f"Unexpected dialog: {args[1:]}")

    monkeypatch.setattr(QMessageBox, "warning", unexpected_dialog)


def test_desktop_load_draw_seek_and_export(app, tmp_path):
    source = tmp_path / "input.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 240))
    assert writer.isOpened()
    for index in range(6):
        frame = np.random.default_rng(index).integers(0, 256, (240, 320, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    window = MainWindow()
    window.show()
    app.processEvents()
    try:
        assert not window.export_button.isEnabled()
        assert window.load_video(source)
        assert window.export_button.isEnabled()
        window.scrubber.setValue(3)
        assert window.scrubber.value() == 3
        window.draw.setChecked(True)
        area = window.preview.image_rect()
        start = QPoint(int(area.left() + area.width() * 0.2), int(area.top() + area.height() * 0.2))
        end = QPoint(int(area.left() + area.width() * 0.6), int(area.top() + area.height() * 0.6))
        QTest.mousePress(window.preview, Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseMove(window.preview, end)
        QTest.mouseRelease(window.preview, Qt.MouseButton.LeftButton, pos=end)
        assert len(window.settings().manual_regions) == 1
        window.original.setChecked(True)
        window.original.setChecked(False)
        output = tmp_path / "desktop.mp4"
        window.start_export(output)
        assert not window.browse.isEnabled()
        deadline = time.monotonic() + 30
        while window.worker is not None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        assert window.worker is None, "Export did not finish"
        assert output.is_file()
        assert window.progress.value() == 100
        assert window.last_output == output.resolve()
        assert window.browse.isEnabled()
        assert window.open_output.isVisible()
        window.clear_regions()
        assert window.settings().manual_regions == ()
    finally:
        if window.worker is not None:
            window.worker.cancel_event.set()
            while window.worker is not None:
                app.processEvents()
                time.sleep(0.01)
        window.close()


def test_packaged_smoke_routine(app):
    from blur_all.smoke import smoke_test

    window = MainWindow()
    window.show()
    app.processEvents()
    try:
        smoke_test(window, app)
    finally:
        window.close()
