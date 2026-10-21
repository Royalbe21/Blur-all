import time
import subprocess

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
        wait_until(app, lambda: window.frame is not None and window.displayed_index == 3)
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


def wait_until(app, condition, timeout=5):
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)
    assert condition(), "The expected desktop state did not arrive"


def make_clip(path, frames=30):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20, (160, 120))
    assert writer.isOpened()
    for index in range(frames):
        writer.write(np.full((120, 160, 3), index * 5 % 256, np.uint8))
    writer.release()


def test_play_pause_step_speed_and_end_of_video(app, tmp_path):
    source = tmp_path / "playback.avi"
    make_clip(source, 12)
    window = MainWindow()
    window.show()
    try:
        assert window.load_video(source)
        wait_until(app, lambda: window.frame is not None)
        QTest.mouseClick(window.play_button, Qt.MouseButton.LeftButton)
        wait_until(app, lambda: window.displayed_index >= 2)
        QTest.mouseClick(window.play_button, Qt.MouseButton.LeftButton)
        assert not window.timer.isActive()
        stopped = window.scrubber.value()
        window.step(1)
        wait_until(app, lambda: window.displayed_index == stopped + 1)
        window.step(-1)
        wait_until(app, lambda: window.displayed_index == stopped)
        window.speed.setCurrentIndex(4)
        assert window.player.playbackRate() == 2.0
        window.toggle_playback()
        wait_until(app, lambda: window.displayed_index == 11 and not window.timer.isActive())
        # Restarting from the end starts playback at the beginning.
        window.toggle_playback()
        assert window.timer.isActive()
        assert window.playback_base == 0
        window.pause()
    finally:
        window.close()


def test_preview_options_and_undo_are_applied_to_export_settings(app, tmp_path):
    source = tmp_path / "options.avi"
    make_clip(source)
    window = MainWindow()
    window.show()
    try:
        assert window.load_video(source)
        wait_until(app, lambda: window.frame is not None)
        window.preview.regions.extend([(0.1, 0.1, 0.3, 0.3), (0.5, 0.5, 0.2, 0.2)])
        window.regions_changed()
        window.effect.setCurrentIndex(2)
        window.margin.setValue(50)
        window.mute.setChecked(True)
        assert window.audio.isChecked(), "Muting preview must not remove exported audio"
        assert window.audio_output.isMuted()
        assert window.settings().effect == "solid"
        assert window.settings().margin == 0.5
        assert not window.strength.isEnabled()
        wait_until(app, lambda: window.processed_frame is not None and np.all(window.processed_frame[20:40, 20:40] == 18))
        window.undo_region()
        assert len(window.settings().manual_regions) == 1
        assert window.region_count.text() == "1"
        window.clear_regions()
        assert not window.undo.isEnabled()
    finally:
        window.close()


def test_seeking_ignores_old_frames_and_keeps_ui_responsive(app, tmp_path, monkeypatch):
    from PySide6.QtCore import QTimer
    import blur_all.preview as preview_module

    original_detector = preview_module.FaceDetector

    class SlowDetector(original_detector):
        def detect(self, frame):
            time.sleep(0.1)
            return super().detect(frame)

    monkeypatch.setattr(preview_module, "FaceDetector", SlowDetector)
    source = tmp_path / "seeking.avi"
    make_clip(source, 30)
    window = MainWindow()
    window.show()
    beats = []
    heartbeat = QTimer()
    heartbeat.setInterval(10)
    heartbeat.timeout.connect(lambda: beats.append(1))
    heartbeat.start()
    try:
        assert window.load_video(source)
        window.scrubber.setValue(4)
        window.scrubber.setValue(15)
        window.scrubber.setValue(25)
        wait_until(app, lambda: window.displayed_index == 25)
        assert len(beats) >= 4, "Detection must not block the UI event loop"
        assert float(window.frame.mean()) == pytest.approx(125, abs=3)
        window.scrubber.setValue(3)
        wait_until(app, lambda: window.displayed_index == 3)
        assert float(window.frame.mean()) == pytest.approx(15, abs=3)
    finally:
        heartbeat.stop()
        window.close()


def test_preview_decodes_real_audio_without_changing_export_audio(app, tmp_path):
    from PySide6.QtCore import QObject, Slot
    from PySide6.QtMultimedia import QAudioBuffer, QAudioBufferOutput
    from blur_all.core import ffmpeg_executable

    silent, source = tmp_path / "silent.avi", tmp_path / "with-audio.mp4"
    make_clip(silent, 20)
    subprocess.run([
        ffmpeg_executable(), "-v", "error", "-i", str(silent), "-f", "lavfi",
        "-i", "sine=frequency=440:duration=1", "-c:v", "libx264", "-c:a", "aac",
        "-shortest", str(source),
    ], check=True)
    window = MainWindow()
    window.show()
    buffers = []

    class Receiver(QObject):
        @Slot(QAudioBuffer)
        def receive(self, buffer):
            buffers.append(buffer.sampleCount())

    receiver = Receiver(window)
    audio_tap = QAudioBufferOutput(window)
    audio_tap.audioBufferReceived.connect(receiver.receive, Qt.ConnectionType.QueuedConnection)
    window.player.setAudioBufferOutput(audio_tap)
    try:
        assert window.load_video(source)
        wait_until(app, lambda: window.frame is not None and window.player.hasAudio())
        window.toggle_playback()
        wait_until(app, lambda: sum(buffers) > 1000)
        window.pause()
        window.mute.setChecked(True)
        assert window.audio.isChecked()
        assert window.audio_output.isMuted()
    finally:
        window.close()


def test_preview_recovers_after_an_undecodable_frame(app, tmp_path):
    from blur_all.core import Settings
    from blur_all.preview import PreviewWorker

    source = tmp_path / "recover.avi"
    make_clip(source, 5)
    worker = PreviewWorker(source, 20)
    failures, frames = [], []
    worker.failed.connect(lambda generation, message: failures.append(message))
    worker.ready.connect(lambda packet: frames.append(packet))
    worker.start()
    try:
        worker.request(500, Settings(), 1)
        wait_until(app, lambda: bool(failures))
        worker.request(2, Settings(), 2)
        wait_until(app, lambda: bool(frames))
        assert frames[-1].index == 2
        assert frames[-1].generation == 2
    finally:
        worker.stop()
        worker.wait()


def test_packaged_smoke_routine(app):
    from blur_all.smoke import smoke_test

    window = MainWindow()
    window.show()
    app.processEvents()
    try:
        smoke_test(window, app)
    finally:
        window.close()
