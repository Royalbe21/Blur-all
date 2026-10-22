"""Short-term face association and motion recovery, without recognizing identities."""

from dataclasses import dataclass, field
import math

import cv2
import numpy as np

Box = tuple[int, int, int, int]


def overlap(a, b) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    area = max(0, right - left) * max(0, bottom - top)
    return area / max(1, a[2] * a[3] + b[2] * b[3] - area)


def corners(box):
    x, y, w, h = box
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.float32)


def bounds(quad, width, height) -> Box:
    left, top = np.maximum(0, np.floor(quad.min(axis=0))).astype(int)
    right, bottom = np.minimum([width, height], np.ceil(quad.max(axis=0))).astype(int)
    return int(left), int(top), max(0, int(right - left)), max(0, int(bottom - top))


@dataclass
class TrackingFrame:
    gray: np.ndarray
    scale_x: float
    scale_y: float
    width: int
    height: int
    scene_cut: bool


@dataclass
class Track:
    id: int
    box: Box
    ttl: int
    smooth: np.ndarray
    quad: np.ndarray
    points: np.ndarray | None = None
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, np.float32))
    uncertain: int = 0


class FaceTracker:
    """Associate detections one-to-one and recover short gaps with robust affine flow.

    Confident detections bypass optical flow. Features are allocated per face and
    reused during dropouts, then discarded on a new detection or a scene cut.
    """

    def __init__(self, frames: int):
        self.frames = max(1, frames)
        self.tracks: list[Track] = []
        self.next_id = 1
        self.last_gray = None
        self.last_thumbnail = None
        self.needs_detection = False
        self.last_frame: TrackingFrame | None = None

    def prepare(self, frame: np.ndarray) -> TrackingFrame:
        height, width = frame.shape[:2]
        scale = min(1.0, 640 / max(width, height))
        size = max(1, round(width * scale)), max(1, round(height * scale))
        # Resize color first so a 4K source does not require a full-size gray copy.
        small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA) if scale < 1 else frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        thumbnail = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
        cut = self.last_gray is not None and self.last_gray.shape != gray.shape
        if self.last_thumbnail is not None:
            delta = thumbnail - self.last_thumbnail
            raw_change = np.abs(delta).mean()
            structural_change = np.abs(delta - np.median(delta)).mean()
            cut = cut or raw_change > 65 or structural_change > 24
        if cut:
            self.tracks.clear()
        self.last_thumbnail = thumbnail
        return TrackingFrame(gray, size[0] / width, size[1] / height, width, height, bool(cut))

    def _associate(self, detections):
        candidates = []
        for i, track in enumerate(self.tracks):
            for j, box in enumerate(detections):
                iou = overlap(track.box, box)
                predicted = np.array(track.box[:2]) + np.array(track.box[2:]) / 2 + track.velocity
                center = np.array(box[:2]) + np.array(box[2:]) / 2
                distance = np.linalg.norm(predicted - center) / max(1, math.hypot(*track.box[2:]))
                size_ratio = box[2] * box[3] / max(1, track.box[2] * track.box[3])
                if 0.3 < size_ratio < 3.3 and (iou > 0.1 or distance < 0.4):
                    candidates.append((iou - 0.15 * distance, i, j))
        matched, used_tracks, used_detections = [], set(), set()
        for _, i, j in sorted(candidates, reverse=True):
            if i not in used_tracks and j not in used_detections:
                matched.append((i, j))
                used_tracks.add(i)
                used_detections.add(j)
        return matched, used_tracks, used_detections

    def _correct_track(self, track, box):
        previous = np.array(track.box[:2]) + np.array(track.box[2:]) / 2
        center = np.array(box[:2]) + np.array(box[2:]) / 2
        track.velocity = (center - previous).astype(np.float32)
        measurement = np.array(box, np.float32)
        # Suppress tiny detector jitter without lagging behind deliberate motion.
        alpha = 0.65 if np.linalg.norm(track.velocity) < min(box[2:]) * 0.08 else 1.0
        track.smooth = track.smooth * (1 - alpha) + measurement * alpha
        track.box, track.quad = box, corners(box)
        track.ttl, track.uncertain, track.points = self.frames, 0, None

    def _add(self, box):
        self.tracks.append(Track(self.next_id, box, self.frames, np.array(box, np.float32), corners(box)))
        self.next_id += 1

    def _seed(self, track, prepared):
        scale = np.array([prepared.scale_x, prepared.scale_y], np.float32)
        box = bounds(track.quad * scale, prepared.gray.shape[1], prepared.gray.shape[0])
        x, y, w, h = box
        if min(w, h) < 5:
            return None
        points = cv2.goodFeaturesToTrack(self.last_gray[y:y+h, x:x+w], 64, 0.015, 3)
        if points is not None:
            points += np.array([x, y], np.float32)
        return points

    def _propagate(self, tracks, prepared):
        if not tracks:
            return
        self.needs_detection = False
        if self.last_gray is None or self.last_gray.shape != prepared.gray.shape:
            self.needs_detection = True
            return
        batches, owners = [], []
        for track in tracks:
            if track.points is None or len(track.points) < 8:
                track.points = self._seed(track, prepared)
            if track.points is not None:
                batches.append(track.points)
                owners.extend([track.id] * len(track.points))
        if not batches:
            for track in tracks:
                track.uncertain += 1
            self.needs_detection = True
            return
        points = np.concatenate(batches).astype(np.float32)
        options = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03))
        moved, forward, _ = cv2.calcOpticalFlowPyrLK(self.last_gray, prepared.gray, points, None, **options)
        if moved is None:
            self.needs_detection = True
            return
        returned, backward, _ = cv2.calcOpticalFlowPyrLK(prepared.gray, self.last_gray, moved, None, **options)
        if returned is None:
            self.needs_detection = True
            return
        valid = (forward.ravel() == 1) & (backward.ravel() == 1)
        valid &= np.linalg.norm(points.reshape(-1, 2) - returned.reshape(-1, 2), axis=1) < 1.25
        owners = np.asarray(owners)
        scale = np.array([prepared.scale_x, prepared.scale_y], np.float32)
        for track in tracks:
            selection = (owners == track.id) & valid
            good = False
            if selection.sum() >= 4:
                source, target = points.reshape(-1, 2)[selection], moved.reshape(-1, 2)[selection]
                transform, inliers = cv2.estimateAffinePartial2D(
                    source, target, method=cv2.RANSAC, ransacReprojThreshold=2,
                    maxIters=100, confidence=0.95, refineIters=3,
                )
                if transform is not None and inliers is not None:
                    factor = math.hypot(transform[0, 0], transform[1, 0])
                    angle = abs(math.degrees(math.atan2(transform[1, 0], transform[0, 0])))
                    good = inliers.sum() >= 4 and inliers.mean() >= 0.6 and 0.8 < factor < 1.25 and angle < 25
                    if good:
                        quad = cv2.transform((track.quad * scale)[None], transform)[0] / scale
                        box = bounds(quad, prepared.width, prepared.height)
                        shift = (quad.mean(axis=0) - track.quad.mean(axis=0))
                        good = min(box[2:]) > 2 and np.linalg.norm(shift) < max(24, max(track.box[2:]) * 0.85)
                        if good:
                            track.velocity = shift
                            track.box, track.quad = box, quad
                            track.smooth = np.array(box, np.float32)
                            track.points = target[inliers.ravel() == 1].reshape(-1, 1, 2)
                            track.uncertain = 0
            if not good:
                track.points = None
                track.uncertain += 1
                self.needs_detection = True

    def _outputs(self):
        if self.last_frame is None:
            return []
        result = []
        for track in self.tracks:
            # Include the full fresh detection even when smoothing its jitter.
            quad = np.concatenate([corners(track.box), corners(track.smooth)])
            result.append(bounds(quad, self.last_frame.width, self.last_frame.height))
        return result

    def correct(self, detections: list[Box]) -> list[Box]:
        """Apply an immediate detector refresh after unreliable skipped-frame flow."""
        matched, _, used = self._associate(detections)
        for i, j in matched:
            self._correct_track(self.tracks[i], detections[j])
        for j, box in enumerate(detections):
            if j not in used:
                self._add(box)
        return self._outputs()

    def update(self, detections: list[Box] | None, frame=None, *, prepared=None) -> list[Box]:
        prepared = prepared if prepared is not None else self.prepare(frame)
        self.last_frame = prepared
        self.needs_detection = False
        fresh = detections if detections is not None else []
        matched, used_tracks, used_detections = self._associate(fresh)
        for i, j in matched:
            self._correct_track(self.tracks[i], fresh[j])
        unmatched = [track for i, track in enumerate(self.tracks) if i not in used_tracks and track.ttl > 1]
        self._propagate(unmatched, prepared)
        for i, track in enumerate(self.tracks):
            if i not in used_tracks:
                track.ttl -= 1
        self.tracks = [track for track in self.tracks if track.ttl > 0]
        for j, box in enumerate(fresh):
            if j not in used_detections:
                self._add(box)
        self.last_gray = prepared.gray
        return self._outputs()
