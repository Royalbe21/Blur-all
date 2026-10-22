"""Desktop editor with background video previews and local exports."""

from __future__ import annotations

from pathlib import Path
import sys
import threading
import time

import cv2
from PySide6.QtCore import QElapsedTimer, QPoint, QPointF, QRectF, QSize, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QIcon, QImage, QKeySequence, QPainter, QPen, QPixmap, QPolygonF, QShortcut
from PySide6.QtMultimedia import QAudioOutput, QMediaDevices, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFrame, QHBoxLayout,
    QLabel, QMainWindow, QMessageBox, QProgressBar, QPushButton, QScrollArea,
    QSlider, QTabWidget, QVBoxLayout, QWidget,
)

from blur_all.core import Cancelled, FaceDetector, Settings, VideoError, export_dimensions, export_video, open_video, video_info
from blur_all.preview import PreviewWorker


STYLE = """
QWidget { color: #ececf3; font-family: 'Segoe UI', sans-serif; font-size: 13px; }
QMainWindow, QWidget#root { background: #111217; }
QFrame#panel { background: #191b23; border: 1px solid #2b2d38; border-radius: 12px; }
QFrame#sidebar { background: #191b23; border: 1px solid #2b2d38; border-radius: 12px; }
QWidget#sidebarContent { background: #191b23; }
QLabel { background: transparent; border: none; }
QLabel#title { font-size: 23px; font-weight: 700; letter-spacing: -0.5px; }
QLabel#logo { background: #c5b5ff; color: #211933; border-radius: 10px; font-size: 25px; font-weight: 800; }
QLabel#muted { color: #9195a8; font-size: 12px; }
QLabel#section { color: #a9adbf; font-size: 10px; font-weight: 700; letter-spacing: 1.5px; }
QLabel#value { color: #c5b5ff; font-weight: 600; }
QLabel#badge { background: #20332e; color: #88d7ba; padding: 6px 11px; border-radius: 6px; font-size: 11px; }
QLabel#fileTitle { font-size: 14px; font-weight: 600; }
QLabel#preview { background: #0b0c10; color: #a9adbf; border-radius: 8px; }
QPushButton { background: #252834; border: 1px solid #36394a; border-radius: 7px; padding: 9px 13px; font-weight: 500; }
QPushButton:hover { background: #333749; border-color: #62637a; }
QPushButton:pressed, QPushButton:checked { background: #3d3455; border-color: #a997e0; }
QPushButton:disabled { color: #616579; background: #20222b; border-color: #2b2d38; }
QPushButton#primary { background: #c5b5ff; color: #211933; border: 1px solid #c5b5ff; font-weight: 700; }
QPushButton#primary:hover { background: #d8ceff; }
QPushButton#primary:disabled { background: #514963; color: #9e94b5; border-color: #514963; }
QPushButton#play { background: #c5b5ff; color: #211933; font-size: 15px; font-weight: 700; border: none; }
QPushButton#play:disabled { background: #332e43; color: #746b8b; }
QComboBox { background: #252834; border: 1px solid #36394a; border-radius: 7px; padding: 8px; min-height: 17px; }
QComboBox:disabled { color: #616579; }
QComboBox QAbstractItemView { background: #252834; selection-background-color: #4b4164; }
QSlider::groove:horizontal { height: 5px; background: #333645; border-radius: 2px; }
QSlider { background: transparent; min-height: 18px; }
QSlider::sub-page:horizontal { background: #b5a0ee; border-radius: 2px; }
QSlider::handle:horizontal { background: #dcd1ff; width: 13px; margin: -5px 0; border-radius: 6px; }
QSlider:disabled::sub-page:horizontal { background: #504b62; }
QCheckBox { spacing: 9px; padding: 3px 0; }
QCheckBox::indicator { width: 15px; height: 15px; border: 1px solid #5b5c71; border-radius: 4px; background: #22242e; }
QCheckBox::indicator:checked { background: #b5a0ee; border-color: #b5a0ee; image: none; }
QProgressBar { background: #2b2e3b; border: none; border-radius: 3px; height: 6px; }
QProgressBar::chunk { background: #b5a0ee; border-radius: 3px; }
QScrollArea { background: transparent; border: none; }
QScrollBar:vertical { background: #191b23; width: 7px; }
QScrollBar::handle:vertical { background: #484659; min-height: 20px; border-radius: 3px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QToolTip { background: #252834; color: #ececf3; border: 1px solid #5c5573; padding: 5px; }
QTabWidget::pane { border: none; background: #191b23; }
QTabBar::tab { background: #20222c; color: #999caf; padding: 11px 25px; border-bottom: 2px solid #20222c; }
QTabBar::tab:selected { color: #d9ceff; background: #191b23; border-bottom-color: #b5a0ee; }
"""


def label(text: str, name: str | None = None) -> QLabel:
    widget = QLabel(text)
    widget.setTextFormat(Qt.TextFormat.PlainText)
    if name:
        widget.setObjectName(name)
    widget.setWordWrap(name not in {"section", "badge", "value", "title"})
    return widget


def transport_icon(kind: str, color: str = "#ececf3") -> QIcon:
    pixmap = QPixmap(24, 24)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(color))
    if kind == "pause":
        painter.drawRoundedRect(QRectF(6, 5, 4, 14), 1, 1)
        painter.drawRoundedRect(QRectF(14, 5, 4, 14), 1, 1)
    else:
        points = [(8, 5), (19, 12), (8, 19)]
        if kind == "previous":
            points = [(16, 5), (5, 12), (16, 19)]
            painter.drawRect(QRectF(4, 5, 2, 14))
        elif kind == "next":
            points = [(7, 5), (18, 12), (7, 19)]
            painter.drawRect(QRectF(18, 5, 2, 14))
        painter.drawPolygon(QPolygonF([QPointF(x, y) for x, y in points]))
    painter.end()
    return QIcon(pixmap)


def timestamp(seconds: float, precise: bool = False) -> str:
    milliseconds = max(0, round(seconds * 1000))
    seconds = milliseconds // 1000
    value = f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"
    return value + f".{milliseconds % 1000:03d}" if precise else value


def separator() -> QFrame:
    line = QFrame()
    line.setFixedHeight(1)
    line.setStyleSheet("background: #30323e;")
    return line


class Preview(QLabel):
    region_added = Signal(tuple)
    interaction_started = Signal()

    def __init__(self):
        super().__init__("Drop a video here\n\nor choose Open video to get started\n\nMP4 · MOV · MKV · AVI · WebM")
        self.setObjectName("preview")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(360, 240)
        self.setMouseTracking(True)
        self.picture: QPixmap | None = None
        self.draw_mode = False
        self.start: QPoint | None = None
        self.end: QPoint | None = None
        self.regions: list[tuple] = []
        self.detection_regions: list[tuple] = []
        self.show_detections = False

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
        painter.fillRect(self.rect(), QColor("#0b0c10"))
        area = self.image_rect()
        painter.drawPixmap(area.toRect(), self.picture)
        if self.show_detections:
            painter.setPen(QPen(QColor("#88d7ba"), 1))
            for polygon in self.detection_regions:
                painter.drawPolygon(QPolygonF([QPointF(area.x() + x * area.width(), area.y() + y * area.height()) for x, y in polygon]))
        painter.setPen(QPen(QColor("#c5b5ff"), 2, Qt.PenStyle.DashLine))
        for x, y, w, h in self.regions:
            painter.drawRect(QRectF(area.x() + x * area.width(), area.y() + y * area.height(), w * area.width(), h * area.height()))
        if self.start is not None and self.end is not None:
            painter.drawRect(QRectF(self.start, self.end).normalized().intersected(area))

    def mousePressEvent(self, event):
        if self.draw_mode and self.picture is not None and event.button() == Qt.MouseButton.LeftButton:
            point = event.position().toPoint()
            if self.image_rect().contains(point):
                self.interaction_started.emit()
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
            self.succeeded.emit(export_video(self.source, self.output, self.settings, self.cancel_event, self.progress.emit))
        except Cancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Blur All — Video workspace")
        self.resize(1240, 820)
        self.setMinimumSize(920, 660)
        self.setAcceptDrops(True)
        self.source: Path | None = None
        self.info = None
        self.frame = self.processed_frame = None
        self.boxes = []
        self.worker: ExportWorker | None = None
        self.preview_worker: PreviewWorker | None = None
        self.last_output: Path | None = None
        self.close_after_cancel = False
        self.generation = 0
        self.displayed_index = 0
        self.requested_index = -1
        self.export_started = 0.0
        self.effect_strengths = {"blur": 80, "pixelate": 80, "distort": 25, "solid": 80}
        self.current_effect = "blur"
        self.clock = QElapsedTimer()
        self.playback_base = 0
        self.timer = QTimer(self)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self.playback_tick)
        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.audio_output.setVolume(0.7)
        self.has_audio_device = bool(QMediaDevices.audioOutputs())
        # Qt's FFmpeg audio renderer can deadlock on mute after pausing if no
        # physical audio device exists. Keep it detached on headless machines.
        if self.has_audio_device:
            self.player.setAudioOutput(self.audio_output)
        self.player.errorOccurred.connect(self.audio_error)
        self.player.hasAudioChanged.connect(self.update_audio_note)
        self.player.mediaStatusChanged.connect(self.update_audio_note)
        self._build_ui()
        for key, action in [("Space", self.toggle_playback), ("Left", lambda: self.step(-1)), ("Right", lambda: self.step(1)), ("Ctrl+O", self.choose_video), ("Ctrl+Z", self.undo_region)]:
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(action)

    def _build_ui(self):
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(24, 20, 24, 20)
        outer.setSpacing(20)
        header = QHBoxLayout()
        logo = label("B", "logo")
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo.setFixedSize(42, 42)
        header.addWidget(logo)
        names = QVBoxLayout()
        names.setSpacing(1)
        names.addWidget(label("Blur All", "title"))
        names.addWidget(label("VIDEO PRIVACY WORKSPACE", "section"))
        header.addLayout(names)
        header.addStretch()
        badge = label("●  Local processing", "badge")
        badge.setFixedHeight(30)
        header.addWidget(badge)
        self.browse = QPushButton("Open video…")
        self.browse.setToolTip("Open a video (Ctrl+O). You can also drag a file into the window.")
        self.browse.clicked.connect(self.choose_video)
        header.addWidget(self.browse)
        outer.addLayout(header)

        body = QHBoxLayout()
        body.setSpacing(18)
        editor = QFrame()
        editor.setObjectName("panel")
        workspace = QVBoxLayout(editor)
        workspace.setContentsMargins(16, 16, 16, 16)
        workspace.setSpacing(12)
        file_row = QHBoxLayout()
        file_details = QVBoxLayout()
        file_details.setSpacing(3)
        self.file_label = label("Your next video, anonymized.", "fileTitle")
        self.file_info = label("Choose a video to start your edit.", "muted")
        file_details.addWidget(self.file_label)
        file_details.addWidget(self.file_info)
        file_row.addLayout(file_details, 1)
        self.face_count = label("PREVIEW", "section")
        file_row.addWidget(self.face_count)
        workspace.addLayout(file_row)
        self.preview = Preview()
        self.preview.region_added.connect(self.regions_changed)
        self.preview.interaction_started.connect(self.pause)
        workspace.addWidget(self.preview, 1)
        view_row = QHBoxLayout()
        self.original = QCheckBox("Show original")
        self.original.setToolTip("Compare with the original. Exports always use your mask settings.")
        self.original.toggled.connect(self.show_current_frame)
        view_row.addWidget(self.original)
        self.outlines = QCheckBox("Mask outlines")
        self.outlines.toggled.connect(self.toggle_outlines)
        view_row.addWidget(self.outlines)
        view_row.addStretch()
        self.preview_note = label("Audio preview ready", "muted")
        view_row.addWidget(self.preview_note)
        workspace.addLayout(view_row)
        self.scrubber = QSlider(Qt.Orientation.Horizontal)
        self.scrubber.setEnabled(False)
        self.scrubber.sliderPressed.connect(self.pause)
        self.scrubber.sliderReleased.connect(self.seek)
        self.scrubber.valueChanged.connect(self.scrub_changed)
        workspace.addWidget(self.scrubber)
        transport = QHBoxLayout()
        self.previous_button = QPushButton()
        self.previous_button.setIcon(transport_icon("previous"))
        self.previous_button.setAccessibleName("Previous frame")
        self.previous_button.setToolTip("Previous frame (Left arrow)")
        self.previous_button.clicked.connect(lambda: self.step(-1))
        self.play_button = QPushButton()
        self.play_button.setIcon(transport_icon("play", "#211933"))
        self.play_button.setIconSize(QSize(22, 22))
        self.play_button.setAccessibleName("Play video")
        self.play_button.setObjectName("play")
        self.play_button.setFixedWidth(48)
        self.play_button.setToolTip("Play / pause (Space)")
        self.play_button.clicked.connect(self.toggle_playback)
        self.next_button = QPushButton()
        self.next_button.setIcon(transport_icon("next"))
        self.next_button.setAccessibleName("Next frame")
        self.next_button.setToolTip("Next frame (Right arrow)")
        self.next_button.clicked.connect(lambda: self.step(1))
        for button in [self.previous_button, self.play_button, self.next_button]:
            button.setEnabled(False)
            transport.addWidget(button)
        self.time_label = label("00:00:00.000 / 00:00:00", "muted")
        transport.addWidget(self.time_label, 1)
        self.mute = QPushButton("Sound on")
        self.mute.setCheckable(True)
        self.mute.setToolTip(
            "Mute preview only; use Keep audio to change the export." if self.has_audio_device
            else "No sound device is available. Connect one and reopen Blur All. Exported audio is unaffected."
        )
        self.mute.toggled.connect(self.toggle_mute)
        transport.addWidget(self.mute)
        self.speed = QComboBox()
        for rate in [0.25, 0.5, 1.0, 1.5, 2.0]:
            self.speed.addItem(f"{rate:g}×", rate)
        self.speed.setCurrentIndex(2)
        self.speed.setToolTip("Preview playback speed")
        self.speed.currentIndexChanged.connect(self.change_speed)
        transport.addWidget(self.speed)
        workspace.addLayout(transport)
        workspace.addWidget(label("Space  play / pause     ← →  step frames     Ctrl+Z  undo area", "muted"))
        body.addWidget(editor, 1)

        side = QFrame()
        side.setObjectName("sidebar")
        side.setMinimumWidth(286)
        side.setMaximumWidth(310)
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.controls = QTabWidget()
        mask_content = QWidget()
        mask_content.setObjectName("sidebarContent")
        settings = QVBoxLayout(mask_content)
        settings.setContentsMargins(18, 18, 18, 18)
        settings.setSpacing(10)
        settings.addWidget(label("MASK APPEARANCE", "section"))
        self.effect = QComboBox()
        for title, key in [("Soft blur", "blur"), ("Pixelate / mosaic", "pixelate"), ("Slight distortion", "distort"), ("Solid cover", "solid")]:
            self.effect.addItem(title, key)
        self.effect.currentIndexChanged.connect(self.effect_changed)
        settings.addWidget(self.effect)
        self.effect_hint = label("Smoothly obscure facial detail.", "muted")
        settings.addWidget(self.effect_hint)
        self.mask_shape = QComboBox()
        for title, key in [("Fit face contour", "face"), ("Oval", "oval"), ("Rectangle", "rectangle")]:
            self.mask_shape.addItem(title, key)
        self.mask_shape.setToolTip("Face contours use the eyes and mouth to orient coverage around the forehead, cheeks, and jaw. Review extreme profiles and missed faces.")
        self.mask_shape.currentIndexChanged.connect(self.render_preview)
        settings.addWidget(self.mask_shape)
        strength_row = QHBoxLayout()
        self.strength_caption = label("Strength")
        strength_row.addWidget(self.strength_caption, 1)
        self.strength_value = label("80%", "value")
        strength_row.addWidget(self.strength_value)
        settings.addLayout(strength_row)
        self.strength = QSlider(Qt.Orientation.Horizontal)
        self.strength.setRange(30, 100)
        self.strength.setValue(80)
        self.strength.valueChanged.connect(self.strength_changed)
        settings.addWidget(self.strength)
        margin_row = QHBoxLayout()
        margin_row.addWidget(label("Extra face coverage"), 1)
        self.margin_value = label("15%", "value")
        margin_row.addWidget(self.margin_value)
        settings.addLayout(margin_row)
        self.margin = QSlider(Qt.Orientation.Horizontal)
        self.margin.setRange(0, 70)
        self.margin.setValue(15)
        self.margin.setToolTip("Expand the face contour for more forehead, chin, and side coverage.")
        self.margin.valueChanged.connect(self.margin_changed)
        settings.addWidget(self.margin)
        edge_row = QHBoxLayout()
        edge_row.addWidget(label("Soft edge"), 1)
        self.edge_value = label("8%", "value")
        edge_row.addWidget(self.edge_value)
        settings.addLayout(edge_row)
        self.edge_softness = QSlider(Qt.Orientation.Horizontal)
        self.edge_softness.setRange(0, 25)
        self.edge_softness.setValue(8)
        self.edge_softness.setToolTip("Blend outside the mask contour while keeping its interior fully covered.")
        self.edge_softness.valueChanged.connect(self.edge_changed)
        settings.addWidget(self.edge_softness)
        settings.addWidget(separator())
        settings.addWidget(label("FACE DETECTION", "section"))
        self.sensitivity = QComboBox()
        for title, value in [("Balanced", 0.6), ("Sensitive · more coverage", 0.35), ("Strict · fewer detections", 0.8)]:
            self.sensitivity.addItem(title, value)
        self.sensitivity.currentIndexChanged.connect(self.render_preview)
        settings.addWidget(self.sensitivity)
        self.detail = QComboBox()
        for title, size in [("Standard detail", 960), ("Small faces · higher detail", 1280), ("Fast preview & export", 640)]:
            self.detail.addItem(title, size)
        self.detail.currentIndexChanged.connect(self.render_preview)
        settings.addWidget(self.detail)
        self.tracking_mode = QComboBox()
        for title, interval in [("Accurate · every frame", 1), ("Balanced · faster", 2), ("Fast · quickest", 4)]:
            self.tracking_mode.addItem(title, interval)
        self.tracking_mode.setToolTip("Accurate detects faces on every frame. Balanced checks every 2 frames; Fast checks every 4. Motion follows faces between checks, with an immediate check if tracking becomes unreliable. Faster modes may miss a face that appears briefly between checks.")
        self.tracking_mode.currentIndexChanged.connect(self.tracking_changed)
        settings.addWidget(self.tracking_mode)
        self.tracking_hint = label("Detect on every frame; follow faces through short gaps.", "muted")
        settings.addWidget(self.tracking_hint)
        settings.addWidget(separator())
        region_row = QHBoxLayout()
        region_row.addWidget(label("EXTRA AREAS", "section"), 1)
        self.region_count = label("0", "value")
        region_row.addWidget(self.region_count)
        settings.addLayout(region_row)
        self.draw = QCheckBox("Draw an extra area")
        self.draw.toggled.connect(self.toggle_draw)
        settings.addWidget(self.draw)
        settings.addWidget(label("Drag on the paused preview. Drawn areas stay fixed for the entire video.", "muted"))
        region_buttons = QHBoxLayout()
        self.undo = QPushButton("Undo")
        self.undo.setToolTip("Remove the last drawn area (Ctrl+Z)")
        self.undo.clicked.connect(self.undo_region)
        self.clear = QPushButton("Clear all")
        self.clear.clicked.connect(self.clear_regions)
        region_buttons.addWidget(self.undo)
        region_buttons.addWidget(self.clear)
        settings.addLayout(region_buttons)
        settings.addStretch()
        settings.addWidget(label("Review before sharing. Small or obscured faces can be missed.", "muted"))
        scroll.setWidget(mask_content)
        self.controls.addTab(scroll, "Masks")

        export_content = QWidget()
        export_content.setObjectName("sidebarContent")
        export_layout = QVBoxLayout(export_content)
        export_layout.setContentsMargins(18, 18, 18, 18)
        export_layout.setSpacing(12)
        export_layout.addWidget(label("FILE FORMAT", "section"))
        self.output_format = QComboBox()
        for title, key in [("MP4 · widest compatibility", "mp4"), ("MOV · QuickTime", "mov"), ("MKV · Matroska", "mkv")]:
            self.output_format.addItem(title, key)
        export_layout.addWidget(self.output_format)
        export_layout.addWidget(label("VIDEO QUALITY", "section"))
        self.export_quality = QComboBox()
        for title, key in [("High quality · larger files", "high"), ("Balanced quality & size", "balanced"), ("Smaller file size", "small")]:
            self.export_quality.addItem(title, key)
        export_layout.addWidget(self.export_quality)
        export_layout.addWidget(label("OUTPUT RESOLUTION", "section"))
        self.resolution = QComboBox()
        for title, key in [("Original size", "source"), ("1080p", "1080p"), ("720p", "720p"), ("480p", "480p")]:
            self.resolution.addItem(title, key)
        export_layout.addWidget(self.resolution)
        export_layout.addWidget(label("Keep aspect ratio. Portrait videos stay portrait; smaller sources are never enlarged.", "muted"))
        export_layout.addWidget(separator())
        export_layout.addWidget(label("AUDIO", "section"))
        self.audio = QCheckBox("Keep original audio")
        self.audio.setChecked(True)
        export_layout.addWidget(self.audio)
        self.audio_bitrate = QComboBox()
        for value in (128, 192, 256):
            self.audio_bitrate.addItem(f"AAC · {value} kbps", value)
        self.audio_bitrate.setCurrentIndex(1)
        export_layout.addWidget(self.audio_bitrate)
        export_layout.addWidget(label("Audio is re-encoded to AAC. Preview mute does not change the export.", "muted"))
        export_layout.addWidget(separator())
        export_layout.addWidget(label("EXPORT SUMMARY", "section"))
        self.export_summary = label("Open a video to see the output dimensions.", "muted")
        export_layout.addWidget(self.export_summary)
        export_layout.addStretch()
        export_layout.addWidget(label("A new file is created. Your original is kept intact.", "muted"))
        export_scroll = QScrollArea()
        export_scroll.setWidgetResizable(True)
        export_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        export_scroll.setWidget(export_content)
        self.controls.addTab(export_scroll, "Export")
        for control in [self.output_format, self.export_quality, self.resolution, self.audio_bitrate]:
            control.currentIndexChanged.connect(self.update_export_summary)
        self.audio.toggled.connect(self.update_export_summary)
        self.controls.setEnabled(False)
        side_layout.addWidget(self.controls)
        body.addWidget(side)
        outer.addLayout(body, 1)

        footer = QFrame()
        footer.setObjectName("panel")
        footer_layout = QVBoxLayout(footer)
        footer_layout.setContentsMargins(16, 13, 16, 13)
        footer_layout.setSpacing(10)
        footer_row = QHBoxLayout()
        details = QVBoxLayout()
        details.setSpacing(3)
        self.status_title = label("Ready to protect your video", "fileTitle")
        self.status = label("Everything stays on this computer. Your original file is kept intact.", "muted")
        details.addWidget(self.status_title)
        details.addWidget(self.status)
        footer_row.addLayout(details, 1)
        self.open_output = QPushButton("Open folder")
        self.open_output.clicked.connect(self.open_folder)
        self.open_output.hide()
        footer_row.addWidget(self.open_output)
        self.review_output = QPushButton("Review export")
        self.review_output.clicked.connect(self.review_export)
        self.review_output.hide()
        footer_row.addWidget(self.review_output)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_export)
        self.cancel_button.hide()
        footer_row.addWidget(self.cancel_button)
        self.export_button = QPushButton("Export video  ↗")
        self.export_button.setObjectName("primary")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.choose_export)
        footer_row.addWidget(self.export_button)
        footer_layout.addLayout(footer_row)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(6)
        footer_layout.addWidget(self.progress)
        outer.addWidget(footer)

    def settings(self) -> Settings:
        return Settings(
            strength=self.strength.value(), margin=self.margin.value() / 100,
            confidence=self.sensitivity.currentData(), detection_size=self.detail.currentData(),
            keep_audio=self.audio.isChecked(), manual_regions=tuple(self.preview.regions),
            effect=self.effect.currentData(),
            detection_interval=self.tracking_mode.currentData(),
            mask_shape=self.mask_shape.currentData(), edge_softness=self.edge_softness.value() / 100,
            output_format=self.output_format.currentData(), export_quality=self.export_quality.currentData(),
            resolution=self.resolution.currentData(), audio_bitrate=self.audio_bitrate.currentData(),
        )

    def choose_video(self):
        if self.worker is not None:
            return
        self.pause()
        filename, _ = QFileDialog.getOpenFileName(self, "Open video", "", "Videos (*.mp4 *.mov *.mkv *.avi *.webm *.m4v *.wmv *.mpeg *.mpg *.mts *.m2ts);;All files (*)")
        if filename:
            self.load_video(Path(filename))

    def load_video(self, path: Path) -> bool:
        if self.worker is not None:
            return False
        self.pause()
        capture = None
        try:
            capture = open_video(path)
            info = video_info(capture)
            ok, _ = capture.read()
            if not ok:
                raise VideoError("Cannot read the first video frame.")
        except Exception as exc:
            QMessageBox.warning(self, "Could not open video", str(exc))
            return False
        finally:
            if capture is not None:
                capture.release()
        self.stop_preview()
        self.player.stop()
        self.source, self.info = path.resolve(), info
        self.frame = self.processed_frame = None
        self.boxes = []
        self.displayed_index, self.requested_index = 0, -1
        self.generation += 1
        self.preview.picture = None
        self.preview.regions.clear()
        self.preview.detection_regions.clear()
        self.preview.start = self.preview.end = None
        self.preview.setText("Preparing your preview…")
        self.preview.update()
        self.scrubber.blockSignals(True)
        self.scrubber.setRange(0, max(0, info.frames - 1))
        self.scrubber.setValue(0)
        self.scrubber.blockSignals(False)
        self.file_label.setText(path.name)
        self.file_label.setToolTip(str(path))
        self.file_info.setText(f"{info.width} × {info.height}  ·  {info.fps:.2f} fps  ·  {timestamp(info.duration)}")
        self.face_count.setText("PREPARING")
        self.region_count.setText("0")
        self.original.setChecked(False)
        self.progress.setValue(0)
        self.open_output.hide()
        self.review_output.hide()
        self.status_title.setText("Preview and adjust")
        self.status.setText("Review face coverage in Masks, then choose your output in Export.")
        self.preview_note.setText("Preview audio · on")
        self.player.setSource(QUrl.fromLocalFile(str(self.source)))
        self.set_busy(False)
        self.start_preview()
        self.update_time(0)
        self.update_export_summary()
        return True

    def start_preview(self):
        if self.source is None or self.info is None or self.preview_worker is not None:
            return
        self.preview_worker = PreviewWorker(self.source, self.info.fps)
        self.preview_worker.ready.connect(self.on_preview)
        self.preview_worker.failed.connect(self.preview_failed)
        self.preview_worker.start()
        self.render_preview()

    def stop_preview(self):
        self.generation += 1
        if self.preview_worker is not None:
            self.preview_worker.stop()
            self.preview_worker.wait()
            self.preview_worker.deleteLater()
            self.preview_worker = None

    def on_preview(self, packet):
        if packet.generation != self.generation:
            return
        self.frame, self.processed_frame, self.boxes = packet.original, packet.processed, packet.boxes
        self.displayed_index = packet.index
        self.preview.detection_regions = packet.regions
        self.scrubber.blockSignals(True)
        self.scrubber.setValue(packet.index)
        self.scrubber.blockSignals(False)
        self.update_time(packet.index)
        count = len(packet.boxes)
        self.face_count.setText(f"{count} FACE{'S' if count != 1 else ''} COVERED")
        self.show_current_frame()

    def preview_failed(self, generation, message):
        if generation != self.generation:
            return
        self.pause()
        self.status_title.setText("Preview needs attention")
        self.status.setText(message)

    def render_preview(self, *_):
        if self.preview_worker is None or self.worker is not None:
            return
        self.generation += 1
        self.requested_index = self.scrubber.value()
        self.preview_worker.request(self.requested_index, self.settings(), self.generation)

    def show_current_frame(self, *_):
        frame = self.frame if self.original.isChecked() else self.processed_frame
        if frame is not None:
            self.preview.set_frame(frame)

    def update_time(self, index):
        if self.info:
            self.time_label.setText(f"{timestamp(index / self.info.fps, True)} / {timestamp(self.info.duration)}")

    def scrub_changed(self, value):
        self.update_time(value)
        if not self.scrubber.isSliderDown():
            self.seek()

    def seek(self):
        if self.info is not None and self.worker is None:
            self.pause(index=self.scrubber.value())

    def pause(self, *_, index=None):
        was_playing = self.timer.isActive()
        self.timer.stop()
        self.player.pause()
        self.play_button.setIcon(transport_icon("play", "#211933"))
        self.play_button.setAccessibleName("Play video")
        if index is not None:
            self.scrubber.blockSignals(True)
            self.scrubber.setValue(index)
            self.scrubber.blockSignals(False)
        if self.info is not None and (was_playing or index is not None):
            self.player.setPosition(round(self.scrubber.value() / self.info.fps * 1000))
            self.render_preview()

    def toggle_playback(self):
        if self.info is None or self.info.frames < 2 or self.worker is not None:
            return
        if self.timer.isActive():
            self.pause()
            return
        if self.scrubber.value() >= self.info.frames - 1:
            self.pause(index=0)
        self.draw.setChecked(False)
        self.playback_base = self.scrubber.value()
        self.clock.start()
        self.player.setPosition(round(self.playback_base / self.info.fps * 1000))
        self.player.setPlaybackRate(self.speed.currentData())
        self.player.play()
        self.timer.start()
        self.play_button.setIcon(transport_icon("pause", "#211933"))
        self.play_button.setAccessibleName("Pause video")

    def playback_tick(self):
        if self.info is None or self.preview_worker is None:
            return
        index = self.playback_base + int(self.clock.elapsed() / 1000 * self.info.fps * self.speed.currentData())
        if index >= self.info.frames - 1:
            self.pause(index=self.info.frames - 1)
            return
        if index != self.requested_index:
            self.requested_index = index
            self.preview_worker.request(index, self.settings(), self.generation)

    def step(self, offset):
        if self.info and self.worker is None:
            self.pause(index=max(0, min(self.info.frames - 1, self.scrubber.value() + offset)))

    def change_speed(self):
        playing = self.timer.isActive()
        self.pause()
        self.player.setPlaybackRate(self.speed.currentData())
        if playing:
            self.toggle_playback()

    def toggle_mute(self, muted):
        self.audio_output.setMuted(muted)
        self.mute.setText("Muted" if muted else "Sound on")
        self.update_audio_note()

    def update_audio_note(self, *_):
        if self.player.error() != QMediaPlayer.Error.NoError:
            self.preview_note.setText("Preview audio unavailable")
        elif self.player.mediaStatus() in {QMediaPlayer.MediaStatus.LoadedMedia, QMediaPlayer.MediaStatus.BufferedMedia, QMediaPlayer.MediaStatus.EndOfMedia} and not self.player.hasAudio():
            self.preview_note.setText("No audio track")
        elif not self.has_audio_device:
            self.preview_note.setText("No audio device")
        else:
            self.preview_note.setText("Preview audio · muted" if self.mute.isChecked() else "Preview audio · on")

    def audio_error(self, *_):
        self.preview_note.setText("Preview audio unavailable")
        # Visual preview and exports use independent decoders and remain available.

    def toggle_outlines(self, enabled):
        self.preview.show_detections = enabled
        self.preview.update()

    def strength_changed(self, value):
        self.strength_value.setText(f"{value}%")
        self.render_preview()

    def margin_changed(self, value):
        self.margin_value.setText(f"{value}%")
        self.render_preview()

    def edge_changed(self, value):
        self.edge_value.setText(f"{value}%")
        self.render_preview()

    def update_export_summary(self, *_):
        self.audio_bitrate.setEnabled(self.audio.isChecked())
        if self.info is None:
            return
        width, height = export_dimensions(self.info.width, self.info.height, self.resolution.currentData())
        audio = f"AAC {self.audio_bitrate.currentData()} kbps" if self.audio.isChecked() else "No audio"
        self.export_summary.setText(f"{self.output_format.currentData().upper()} · H.264\n{width} × {height} · {self.info.fps:.2f} fps\n{self.export_quality.currentText()}\n{audio}")

    def effect_changed(self):
        self.effect_strengths[self.current_effect] = self.strength.value()
        effect = self.effect.currentData()
        self.current_effect = effect
        self.strength.blockSignals(True)
        self.strength.setRange(30 if effect == "blur" else 5, 100)
        self.strength.setValue(self.effect_strengths[effect])
        self.strength.blockSignals(False)
        self.strength.setEnabled(effect != "solid")
        self.strength_caption.setText({"distort": "Distortion amount", "pixelate": "Block size"}.get(effect, "Strength"))
        self.effect_hint.setText({
            "blur": "Smoothly obscure facial detail.",
            "pixelate": "Cover faces with adjustable mosaic blocks.",
            "distort": "Subtle geometric warp. Faces may remain recognizable.",
            "solid": "Replace face regions with a solid cover.",
        }[effect])
        self.strength_value.setText(f"{self.strength.value()}%")
        self.render_preview()

    def tracking_changed(self):
        interval = self.tracking_mode.currentData()
        self.tracking_hint.setText(
            "Detect on every frame; follow faces through short gaps." if interval == 1
            else f"Check every {interval} frames. Brief appearances between checks can be missed."
        )
        self.render_preview()

    def toggle_draw(self, enabled):
        if enabled:
            self.pause()
        self.preview.draw_mode = enabled
        self.preview.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)

    def regions_changed(self, *_):
        self.region_count.setText(str(len(self.preview.regions)))
        self.undo.setEnabled(bool(self.preview.regions))
        self.clear.setEnabled(bool(self.preview.regions))
        self.render_preview()

    def undo_region(self):
        if self.worker is None and self.preview.regions:
            self.preview.regions.pop()
            self.regions_changed()

    def clear_regions(self):
        self.preview.regions.clear()
        self.regions_changed()

    def dragEnterEvent(self, event):
        urls = event.mimeData().urls()
        if self.worker is None and len(urls) == 1 and urls[0].isLocalFile() and Path(urls[0].toLocalFile()).is_file():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if self.worker is None and len(urls) == 1 and urls[0].isLocalFile():
            if self.load_video(Path(urls[0].toLocalFile())):
                event.acceptProposedAction()

    def choose_export(self):
        if self.source is None or self.worker is not None:
            return
        self.pause()
        extension = self.output_format.currentData()
        candidate = self.source.with_name(self.source.stem + "_protected." + extension)
        suffix = 2
        while candidate.exists():
            candidate = self.source.with_name(f"{self.source.stem}_protected_{suffix}.{extension}")
            suffix += 1
        filename, _ = QFileDialog.getSaveFileName(self, "Save your protected video", str(candidate), f"{extension.upper()} video (*.{extension})", options=QFileDialog.Option.DontConfirmOverwrite)
        if not filename:
            return
        output = Path(filename)
        if output.suffix.lower() != "." + extension:
            output = output.with_suffix("." + extension)
        if output.exists() or output.resolve() == self.source.resolve():
            QMessageBox.warning(self, "Choose a new file", "Choose an unused filename. Existing videos are never overwritten.")
            return
        self.start_export(output)

    def start_export(self, output: Path):
        if self.worker is not None or self.source is None:
            return
        self.pause()
        self.stop_preview()
        self.player.stop()
        self.set_busy(True)
        self.progress.setValue(0)
        self.export_started = time.monotonic()
        self.status_title.setText("Protecting your video…")
        self.status.setText("Starting local export. Your original stays intact.")
        self.worker = ExportWorker(self.source, output, self.settings())
        self.worker.progress.connect(self.on_progress)
        self.worker.succeeded.connect(self.on_success)
        self.worker.failed.connect(self.on_failure)
        self.worker.cancelled.connect(self.on_cancelled)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

    def set_busy(self, busy: bool):
        loaded = self.source is not None
        self.browse.setEnabled(not busy)
        self.controls.setEnabled(not busy and loaded)
        self.preview.setEnabled(not busy)
        self.scrubber.setEnabled(not busy and loaded and self.info.frames > 1)
        for button in [self.play_button, self.previous_button, self.next_button, self.speed, self.mute, self.original, self.outlines]:
            button.setEnabled(not busy and loaded)
        self.mute.setEnabled(not busy and loaded and self.has_audio_device)
        self.export_button.setEnabled(not busy and loaded)
        self.cancel_button.setVisible(busy)
        self.cancel_button.setEnabled(busy)
        self.undo.setEnabled(bool(self.preview.regions))
        self.clear.setEnabled(bool(self.preview.regions))
        if busy:
            self.open_output.hide()
            self.review_output.hide()

    def on_progress(self, current, total, stage):
        if total <= 0:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, 100)
            self.progress.setValue(min(99, round(current / total * 97)))
        self.status_title.setText(stage)
        elapsed = max(0.01, time.monotonic() - self.export_started)
        rate = current / elapsed
        detail = f"{current:,}" + (f" / {total:,} frames" if total else " frames")
        if elapsed > 0.5 and stage == "Blurring faces":
            detail += f"  ·  {rate:.1f} fps"
            if total > current:
                detail += f"  ·  about {timestamp((total - current) / rate)} remaining"
        self.status.setText(detail)

    def on_success(self, result):
        self.last_output = result.path
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        self.status_title.setText("Your video is ready")
        self.status.setText(f"{result.path.name} · Faces covered in {result.frames_with_faces:,} / {result.frames:,} frames. Review before sharing.")
        self.open_output.show()
        self.review_output.show()

    def on_failure(self, message):
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.status_title.setText("Export could not finish")
        self.status.setText("Your original was kept intact. Check the error and try again.")
        if "--smoke-test" not in sys.argv:
            QMessageBox.warning(self, "Export failed", message)

    def on_cancelled(self):
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.status_title.setText("Export cancelled")
        self.status.setText("Temporary files cleaned up. No output video was saved.")

    def on_worker_finished(self):
        if self.worker:
            self.worker.deleteLater()
            self.worker = None
        self.set_busy(False)
        if self.close_after_cancel:
            self.close()
        else:
            self.start_preview()

    def cancel_export(self):
        if self.worker:
            self.worker.cancel_event.set()
            self.cancel_button.setEnabled(False)
            self.status_title.setText("Cancelling export…")
            self.status.setText("Cleaning up temporary files.")

    def open_folder(self):
        if self.last_output:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.last_output.parent)))

    def review_export(self):
        if self.last_output:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.last_output)))

    def closeEvent(self, event):
        if self.worker is not None:
            answer = QMessageBox.question(self, "Cancel export?", "An export is running. Cancel it and close Blur All?")
            if answer == QMessageBox.StandardButton.Yes:
                self.close_after_cancel = True
                self.cancel_export()
            event.ignore()
            return
        self.pause()
        self.stop_preview()
        self.player.stop()
        self.player.setSource(QUrl())
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
            FaceDetector()
            smoke_test(window, app)
        except Exception as exc:
            import tempfile

            (Path(tempfile.gettempdir()) / "blur-all-smoke-error.txt").write_text(str(exc), encoding="utf-8")
            raise SystemExit(1) from exc
        finally:
            window.close()
        return
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
