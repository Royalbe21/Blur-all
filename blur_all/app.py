"""Windows-friendly Qt desktop interface."""

from __future__ import annotations

from pathlib import Path
import sys
import threading

import cv2
from PySide6.QtCore import QPoint, QRectF, Qt, QThread, Signal
from PySide6.QtGui import QColor, QDesktopServices, QImage, QPainter, QPen, QPixmap
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QGridLayout,
    QHBoxLayout, QLabel, QMainWindow, QMessageBox, QProgressBar, QPushButton,
    QSlider, QVBoxLayout, QWidget,
)

from blur_all.core import (
    Cancelled, FaceDetector, Settings, VideoError, blur_frame,
    export_video, open_video, video_info,
)


STYLE = """
QWidget { background: #11181d; color: #ecf1f4; font-family: 'Segoe UI', sans-serif; font-size: 13px; }
QMainWindow { background: #11181d; }
QLabel#title { font-size: 30px; font-weight: 700; }
QLabel#muted { color: #9aadb9; }
QLabel#section { color: #76e4c5; font-size: 12px; font-weight: 700; }
QLabel#preview { background: #0b1115; border: 1px solid #2a3841; border-radius: 12px; }
QPushButton { background: #25343e; border: 1px solid #3a4b56; border-radius: 7px; padding: 10px 16px; }
QPushButton:hover { background: #344852; }
QPushButton:disabled { background: #1b252c; color: #667780; border-color: #26323a; }
QPushButton#primary { background: #76e4c5; color: #10251e; border: none; font-weight: 700; }
QPushButton#primary:hover { background: #9bf1d9; }
QPushButton#primary:disabled { background: #2e5149; color: #88a59c; }
QComboBox { background: #25343e; border: 1px solid #3a4b56; border-radius: 6px; padding: 7px; }
QSlider::groove:horizontal { height: 5px; background: #31434e; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #76e4c5; border-radius: 2px; }
QSlider::handle:horizontal { background: #d6fff0; width: 14px; margin: -5px 0; border-radius: 7px; }
QCheckBox { spacing: 8px; }
QProgressBar { background: #25343e; border: none; border-radius: 5px; height: 10px; text-align: center; }
QProgressBar::chunk { background: #76e4c5; border-radius: 5px; }
"""


def label(text: str, name: str | None = None) -> QLabel:
    widget = QLabel(text)
    if name:
        widget.setObjectName(name)
    widget.setWordWrap(True)
    return widget


def timestamp(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


class Preview(QLabel):
    region_added = Signal(tuple)

    def __init__(self):
        super().__init__("Choose a video to get started\n\nYour videos stay on this computer")
        self.setObjectName("preview")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(400, 260)
        self.setMouseTracking(True)
        self.picture: QPixmap | None = None
        self.draw_mode = False
        self.start: QPoint | None = None
        self.end: QPoint | None = None
        self.regions: list[tuple] = []

    def image_rect(self) -> QRectF:
        if self.picture is None:
            return QRectF()
        size = self.picture.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        return QRectF((self.width() - size.width()) / 2, (self.height() - size.height()) / 2, size.width(), size.height())

    def set_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        self.picture = QPixmap.fromImage(QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy())
        self.update()

    def paintEvent(self, event):
        if self.picture is None:
            return super().paintEvent(event)
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0b1115"))
        area = self.image_rect()
        painter.drawPixmap(area.toRect(), self.picture)
        painter.setPen(QPen(QColor("#76e4c5"), 2, Qt.PenStyle.DashLine))
        for x, y, w, h in self.regions:
            painter.drawRect(QRectF(area.x() + x * area.width(), area.y() + y * area.height(), w * area.width(), h * area.height()))
        if self.start is not None and self.end is not None:
            painter.drawRect(QRectF(self.start, self.end).normalized().intersected(area))

    def mousePressEvent(self, event):
        if self.draw_mode and self.picture is not None and event.button() == Qt.MouseButton.LeftButton:
            point = event.position().toPoint()
            if self.image_rect().contains(point):
                self.start = self.end = point

    def mouseMoveEvent(self, event):
        if self.start is not None:
            self.end = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event):
        if self.start is None:
            return
        area = self.image_rect()
        rect = QRectF(self.start, event.position().toPoint()).normalized().intersected(area)
        self.start = self.end = None
        if rect.width() >= 5 and rect.height() >= 5:
            region = ((rect.x() - area.x()) / area.width(), (rect.y() - area.y()) / area.height(), rect.width() / area.width(), rect.height() / area.height())
            self.regions.append(region)
            self.region_added.emit(region)
        self.update()


class ExportWorker(QThread):
    progress = Signal(int, int, str)
    succeeded = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, source: Path, output: Path, settings: Settings):
        super().__init__()
        self.source, self.output, self.settings = source, output, settings
        self.cancel_event = threading.Event()

    def run(self):
        try:
            result = export_video(self.source, self.output, self.settings, self.cancel_event, self.progress.emit)
            self.succeeded.emit(result)
        except Cancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Blur All · Face blur for video")
        self.resize(1120, 830)
        self.source: Path | None = None
        self.capture = None
        self.info = None
        self.frame = None
        self.boxes = []
        self.detector = None
        self.worker: ExportWorker | None = None
        self.last_output: Path | None = None
        self.close_after_cancel = False
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(30, 24, 30, 24)
        layout.setSpacing(14)
        header = QHBoxLayout()
        titles = QVBoxLayout()
        titles.addWidget(label("Blur All", "title"))
        titles.addWidget(label("Face blur for video. Private by design.", "muted"))
        header.addLayout(titles, 1)
        self.browse = QPushButton("Open video…")
        self.browse.clicked.connect(self.choose_video)
        header.addWidget(self.browse)
        layout.addLayout(header)
        self.file_label = label("01  /  CHOOSE A VIDEO", "section")
        layout.addWidget(self.file_label)
        self.preview = Preview()
        self.preview.region_added.connect(lambda _: self.render_preview())
        layout.addWidget(self.preview, 1)
        timeline = QHBoxLayout()
        self.scrubber = QSlider(Qt.Orientation.Horizontal)
        self.scrubber.setEnabled(False)
        self.scrubber.sliderReleased.connect(self.seek)
        self.scrubber.valueChanged.connect(self.scrub_changed)
        timeline.addWidget(self.scrubber, 1)
        self.time_label = label("00:00:00 / 00:00:00", "muted")
        timeline.addWidget(self.time_label)
        layout.addLayout(timeline)

        self.controls = QWidget()
        controls = QGridLayout(self.controls)
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setHorizontalSpacing(22)
        for column in range(3):
            controls.setColumnStretch(column, 1)
        controls.addWidget(label("02  /  ADJUST & REVIEW", "section"), 0, 0, 1, 3)
        self.strength = QSlider(Qt.Orientation.Horizontal)
        self.strength.setRange(30, 100)
        self.strength.setValue(80)
        self.strength.valueChanged.connect(self.render_preview)
        controls.addWidget(label("Blur strength"), 1, 0)
        controls.addWidget(self.strength, 2, 0)
        self.sensitivity = QComboBox()
        for text, confidence in [("Balanced", 0.6), ("Sensitive · more detections", 0.35), ("Strict · fewer detections", 0.8)]:
            self.sensitivity.addItem(text, confidence)
        self.sensitivity.currentIndexChanged.connect(self.refresh_detector)
        controls.addWidget(label("Detection sensitivity"), 1, 1)
        controls.addWidget(self.sensitivity, 2, 1)
        self.detail = QComboBox()
        for text, size in [("Standard · 960 px", 960), ("Small faces · 1280 px", 1280), ("Fast · 640 px", 640)]:
            self.detail.addItem(text, size)
        self.detail.currentIndexChanged.connect(self.refresh_detector)
        controls.addWidget(label("Detection detail"), 1, 2)
        controls.addWidget(self.detail, 2, 2)
        self.audio = QCheckBox("Keep audio")
        self.audio.setChecked(True)
        self.original = QCheckBox("Show original")
        self.original.toggled.connect(self.render_preview)
        self.draw = QCheckBox("Draw extra blur areas")
        self.draw.toggled.connect(self.toggle_draw)
        controls.addWidget(self.audio, 3, 0)
        controls.addWidget(self.original, 3, 1)
        controls.addWidget(self.draw, 3, 2)
        self.clear = QPushButton("Clear drawn areas")
        self.clear.clicked.connect(self.clear_regions)
        controls.addWidget(self.clear, 4, 2)
        controls.addWidget(label("Drawn areas stay in the same position for the entire video.", "muted"), 4, 0, 1, 2)
        self.controls.setEnabled(False)
        layout.addWidget(self.controls)
        layout.addWidget(label("Review before sharing: automatic detection can miss small, sideways, or obscured faces. This tool does not guarantee anonymity.", "muted"))
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        layout.addWidget(self.progress)
        footer = QHBoxLayout()
        self.status = label("Ready when you are. Exports are saved as a new MP4.", "muted")
        footer.addWidget(self.status, 1)
        self.open_output = QPushButton("Open output folder")
        self.open_output.hide()
        self.open_output.clicked.connect(self.open_folder)
        footer.addWidget(self.open_output)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_export)
        self.cancel_button.hide()
        footer.addWidget(self.cancel_button)
        self.export_button = QPushButton("Export blurred video…")
        self.export_button.setObjectName("primary")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.choose_export)
        footer.addWidget(self.export_button)
        layout.addLayout(footer)

    def settings(self) -> Settings:
        return Settings(
            strength=self.strength.value(), confidence=self.sensitivity.currentData(),
            detection_size=self.detail.currentData(), keep_audio=self.audio.isChecked(),
            manual_regions=tuple(self.preview.regions),
        )

    def choose_video(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open video", "", "Videos (*.mp4 *.mov *.mkv *.avi *.webm *.m4v *.wmv *.mpeg *.mpg *.mts *.m2ts);;All files (*)")
        if filename:
            self.load_video(Path(filename))

    def load_video(self, path: Path) -> bool:
        new_capture = None
        try:
            new_capture = open_video(path)
            info = video_info(new_capture)
            ok, frame = new_capture.read()
            if not ok:
                raise VideoError("Cannot read the first video frame.")
            detector = FaceDetector(self.sensitivity.currentData(), self.detail.currentData())
            boxes = detector.detect(frame)
        except Exception as exc:
            if new_capture is not None:
                new_capture.release()
            QMessageBox.warning(self, "Could not open video", str(exc))
            return False
        if self.capture is not None:
            self.capture.release()
        self.capture, self.source, self.info = new_capture, path, info
        self.frame, self.detector, self.boxes = frame, detector, boxes
        self.preview.regions.clear()
        self.preview.start = self.preview.end = None
        self.scrubber.setRange(0, max(0, info.frames - 1))
        self.scrubber.setValue(0)
        self.scrubber.setEnabled(info.frames > 1)
        self.controls.setEnabled(True)
        self.export_button.setEnabled(True)
        self.open_output.hide()
        self.progress.setValue(0)
        self.file_label.setText(f"{path.name}  ·  {info.width} × {info.height}  ·  {info.fps:.2f} fps")
        self.file_label.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setText("Scrub through the video to review detection, then export.")
        self.scrub_changed(0)
        self.render_preview()
        return True

    def scrub_changed(self, value):
        if self.info:
            self.time_label.setText(f"{timestamp(value / self.info.fps)} / {timestamp(self.info.duration)}")
        if not self.scrubber.isSliderDown():
            self.seek()

    def seek(self):
        if self.capture is None:
            return
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.scrubber.value())
        ok, frame = self.capture.read()
        if ok:
            self.frame = frame
            try:
                self.boxes = self.detector.detect(frame)
                self.render_preview()
            except Exception as exc:
                self.status.setText(f"Preview failed: {exc}")
        else:
            self.status.setText("Could not decode this frame. Try another position.")

    def refresh_detector(self):
        if self.frame is None:
            return
        try:
            self.detector = FaceDetector(self.sensitivity.currentData(), self.detail.currentData())
            self.boxes = self.detector.detect(self.frame)
            self.render_preview()
        except Exception as exc:
            QMessageBox.warning(self, "Detection failed", str(exc))

    def render_preview(self, *_):
        if self.frame is None:
            return
        frame = self.frame if self.original.isChecked() else blur_frame(self.frame, self.boxes, self.settings())
        self.preview.set_frame(frame)

    def toggle_draw(self, enabled):
        self.preview.draw_mode = enabled
        self.preview.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)

    def clear_regions(self):
        self.preview.regions.clear()
        self.render_preview()

    def choose_export(self):
        if self.source is None:
            return
        candidate = self.source.with_name(self.source.stem + "_blurred.mp4")
        suffix = 2
        while candidate.exists():
            candidate = self.source.with_name(f"{self.source.stem}_blurred_{suffix}.mp4")
            suffix += 1
        filename, _ = QFileDialog.getSaveFileName(self, "Save blurred video as a new file", str(candidate), "MP4 video (*.mp4)", options=QFileDialog.Option.DontConfirmOverwrite)
        if not filename:
            return
        output = Path(filename)
        if output.suffix.lower() != ".mp4":
            output = output.with_suffix(".mp4")
        if output.exists() or output.resolve() == self.source.resolve():
            QMessageBox.warning(self, "Choose a new file", "Choose an unused filename. Existing videos are never overwritten.")
            return
        self.start_export(output)

    def start_export(self, output: Path):
        self.set_busy(True)
        self.progress.setValue(0)
        self.status.setText("Starting local export…")
        self.worker = ExportWorker(self.source, output, self.settings())
        self.worker.progress.connect(self.on_progress)
        self.worker.succeeded.connect(self.on_success)
        self.worker.failed.connect(self.on_failure)
        self.worker.cancelled.connect(self.on_cancelled)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

    def set_busy(self, busy: bool):
        self.browse.setEnabled(not busy)
        self.controls.setEnabled(not busy)
        self.preview.setEnabled(not busy)
        self.scrubber.setEnabled(not busy and self.info is not None and self.info.frames > 1)
        self.export_button.setEnabled(not busy and self.source is not None)
        self.cancel_button.setVisible(busy)
        self.cancel_button.setEnabled(busy)
        if busy:
            self.open_output.hide()

    def on_progress(self, current, total, stage):
        if total <= 0:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, 100)
            self.progress.setValue(min(99, round(current / total * 97)))
        self.status.setText(f"{stage} · {current:,}" + (f" / {total:,} frames" if total else " frames"))

    def on_success(self, result):
        self.last_output = result.path
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        self.status.setText(f"Saved {result.path.name}. Faces detected in {result.frames_with_faces:,} of {result.frames:,} frames. Review the output before sharing.")
        self.open_output.show()

    def on_failure(self, message):
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.status.setText("Export failed. Your original was not changed.")
        if "--smoke-test" not in sys.argv:
            QMessageBox.warning(self, "Export failed", message)

    def on_cancelled(self):
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.status.setText("Export cancelled. No output video was saved.")

    def on_worker_finished(self):
        self.set_busy(False)
        if self.worker:
            self.worker.deleteLater()
            self.worker = None
        if self.close_after_cancel:
            self.close()

    def cancel_export(self):
        if self.worker:
            self.worker.cancel_event.set()
            self.cancel_button.setEnabled(False)
            self.status.setText("Cancelling and cleaning up…")

    def open_folder(self):
        if self.last_output:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.last_output.parent)))

    def closeEvent(self, event):
        if self.worker is not None:
            answer = QMessageBox.question(self, "Cancel export?", "An export is running. Cancel it and close Blur All?")
            if answer == QMessageBox.StandardButton.Yes:
                self.close_after_cancel = True
                self.cancel_export()
            event.ignore()
            return
        if self.capture is not None:
            self.capture.release()
        event.accept()


def main():
    cv2.setNumThreads(4)
    app = QApplication(sys.argv)
    app.setApplicationName("Blur All")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    window = MainWindow()
    window.show()
    if "--smoke-test" in sys.argv:
        from blur_all.smoke import smoke_test

        app.processEvents()
        try:
            FaceDetector()  # Fail without a modal dialog if the packaged model is invalid.
            smoke_test(window, app)
        except Exception as exc:
            import tempfile

            # Windowed builds have no stderr; leave diagnostics for CI instead of a modal.
            (Path(tempfile.gettempdir()) / "blur-all-smoke-error.txt").write_text(str(exc), encoding="utf-8")
            raise SystemExit(1) from exc
        finally:
            window.close()
        return
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
