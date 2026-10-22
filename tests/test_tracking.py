import cv2
import numpy as np
import pytest

import blur_all.core as core
from blur_all.tracking import FaceTracker, bounds, corners, overlap


def moving_patch(index=0, angle=0, scale=1):
    image = np.zeros((256, 384, 3), np.uint8)
    texture = np.random.default_rng(18).integers(0, 256, (96, 96, 3), np.uint8)
    image[70:166, 70:166] = texture
    transform = cv2.getRotationMatrix2D((118, 118), angle, scale)
    transform[:, 2] += [index * 6, index * 3]
    warped = cv2.warpAffine(image, transform, (384, 256))
    quad = cv2.transform(corners((70, 70, 96, 96))[None], transform)[0]
    return warped, bounds(quad, 384, 256)


def test_rotation_and_scale_are_recovered_during_detection_gaps():
    tracker = FaceTracker(10)
    first, box = moving_patch()
    tracker.update([box], first)
    for index in range(1, 5):
        frame, expected = moving_patch(index, angle=index * 4, scale=1 + index * 0.035)
        actual = tracker.update([], frame)[0]
        assert overlap(actual, expected) > 0.8
        assert actual[2] == pytest.approx(expected[2], rel=0.1)
        assert actual[3] == pytest.approx(expected[3], rel=0.1)


def test_features_are_allocated_to_each_face():
    tracker = FaceTracker(5)
    first = np.zeros((240, 480, 3), np.uint8)
    rng = np.random.default_rng(14)
    large = rng.integers(0, 256, (150, 150, 3), np.uint8)
    small = rng.integers(0, 256, (35, 35, 3), np.uint8)
    first[30:180, 30:180], first[100:135, 350:385] = large, small
    tracker.update([(30, 30, 150, 150), (350, 100, 35, 35)], first)
    moved = np.zeros_like(first)
    moved[34:184, 37:187], moved[96:131, 344:379] = large, small
    boxes = tracker.update([], moved)
    assert len(boxes) == 2
    assert max(overlap(box, (344, 96, 35, 35)) for box in boxes) > 0.8
    assert max(overlap(box, (37, 34, 150, 150)) for box in boxes) > 0.9


def test_one_detection_does_not_consume_two_overlapping_tracks():
    image = np.random.default_rng(9).integers(0, 256, (150, 180, 3), np.uint8)
    tracker = FaceTracker(5)
    first, second = (20, 30, 70, 70), (60, 30, 70, 70)
    tracker.update([first, second], image)
    identities = {track.id for track in tracker.tracks}
    output = tracker.update([second], image)
    assert len(output) == 2
    assert {track.id for track in tracker.tracks} == identities


def test_jitter_smoothing_never_uncovers_the_fresh_detection():
    image = np.zeros((160, 160, 3), np.uint8)
    tracker = FaceTracker(5)
    observed, stabilized = [], []
    for index in range(20):
        detection = (50 + (2 if index % 2 else -2), 40, 52, 62)
        box = tracker.update([detection], image)[0]
        assert box[0] <= detection[0]
        assert box[0] + box[2] >= detection[0] + detection[2]
        observed.append(detection[0] + detection[2] / 2)
        stabilized.append(box[0] + box[2] / 2)
    assert np.std(stabilized[3:]) < np.std(observed[3:]) * 0.9


def test_scene_cut_resets_tracks_but_a_small_exposure_change_does_not():
    image = np.zeros((128, 128, 3), np.uint8)
    image[:, :64] = 180
    tracker = FaceTracker(5)
    tracker.update([(20, 20, 40, 40)], image)
    brighter = (image.astype(int) + 20).astype(np.uint8)
    assert tracker.update([], brighter)
    new_scene = np.zeros_like(image)
    new_scene[:64, :] = 240
    assert tracker.update([], new_scene) == []
    assert tracker.last_frame.scene_cut


def test_expired_tracks_do_not_follow_the_background_forever():
    image, box = moving_patch()
    tracker = FaceTracker(3)
    tracker.update([box], image)
    assert tracker.update([], image)
    assert tracker.update([], image)
    assert tracker.update([], image) == []


@pytest.mark.parametrize("interval, expected_calls", [(1, 12), (2, 6), (4, 3)])
def test_detection_modes_save_work_without_dropping_the_tracked_face(monkeypatch, interval, expected_calls):
    frame, box = moving_patch()
    calls = []

    class Detector:
        def detect(self, image):
            calls.append(1)
            return [box]

    monkeypatch.setattr(core, "FaceDetector", lambda *args: Detector())
    analyzer = core.FaceAnalyzer(detection_interval=interval)
    for _ in range(12):
        assert max(overlap(box, out) for out in analyzer.analyze(frame).boxes) > 0.95
    assert len(calls) == expected_calls == analyzer.detector_frames


def test_no_tracks_means_look_for_new_faces_on_every_frame(monkeypatch):
    calls = []

    class Detector:
        def detect(self, image):
            calls.append(1)
            return []

    monkeypatch.setattr(core, "FaceDetector", lambda *args: Detector())
    analyzer = core.FaceAnalyzer(detection_interval=4)
    for _ in range(5):
        analyzer.analyze(np.zeros((100, 100, 3), np.uint8))
    assert len(calls) == 5


def test_weak_flow_requests_a_detector_refresh_on_the_same_frame(monkeypatch):
    calls = []

    class Detector:
        def detect(self, image):
            calls.append(1)
            return [(20, 20, 40, 40)]

    monkeypatch.setattr(core, "FaceDetector", lambda *args: Detector())
    analyzer = core.FaceAnalyzer(detection_interval=4)
    frame = np.zeros((100, 100, 3), np.uint8)
    analyzer.analyze(frame)
    assert analyzer.analyze(frame).did_detect
    assert len(calls) == 2


def test_a_scene_cut_forces_detection_between_scheduled_checks(monkeypatch):
    calls = []

    class Detector:
        def detect(self, image):
            calls.append(1)
            return [(20, 20, 40, 40)] if len(calls) == 1 else [(80, 20, 40, 40)]

    monkeypatch.setattr(core, "FaceDetector", lambda *args: Detector())
    analyzer = core.FaceAnalyzer(detection_interval=4)
    analyzer.analyze(np.zeros((150, 150, 3), np.uint8))
    analysis = analyzer.analyze(np.full((150, 150, 3), 240, np.uint8))
    assert analysis.did_detect and len(calls) == 2
    assert analysis.boxes == [(80, 20, 40, 40)]
