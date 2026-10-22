import cv2
import numpy as np

from blur_all.core import FaceAnalyzer, FaceDetector, Settings, blur_frame
from blur_all.masks import FaceRegion, face_polygon, polygon_mask
from blur_all.tracking import FaceTracker, bounds, corners


def reference_face():
    points = np.array([[60, 65], [100, 65], [80, 85], [65, 105], [95, 105]], np.float32)
    return FaceRegion((40, 30, 80, 100), 1, corners((40, 30, 80, 100)), points)


def test_contour_covers_the_facial_landmarks_but_leaves_box_corners_clear():
    face = reference_face()
    polygon = face_polygon(face, margin=0)
    assert cv2.isContourConvex(polygon)
    for point in face.landmarks:
        assert cv2.pointPolygonTest(polygon, tuple(map(float, point)), False) >= 0
    image = np.full((180, 180, 3), 160, np.uint8)
    treated = blur_frame(image, [face.box], Settings(effect="solid", margin=0, edge_softness=0), faces=[face])
    assert np.array_equal(treated[30, 40], image[30, 40])
    assert np.all(treated[85, 80] == 18)
    assert np.array_equal(treated[160:], image[160:])


def test_face_contour_rotates_with_landmarks_and_tracking_geometry():
    face = reference_face()
    original = face_polygon(face, margin=0.1)
    transform = cv2.getRotationMatrix2D((80, 80), 28, 1.15)
    quad = cv2.transform(face.quad[None], transform)[0]
    moved = FaceRegion(bounds(quad, 220, 220), 1, quad, cv2.transform(face.landmarks[None], transform)[0])
    expected = cv2.transform(original[None], transform)[0]
    assert np.allclose(face_polygon(moved, margin=0.1), expected, atol=0.1)


def test_soft_edges_only_fade_outward_and_keep_the_interior_opaque():
    polygon = face_polygon(reference_face(), margin=0)
    (left, top, right, bottom), alpha = polygon_mask(polygon, 180, 180, feather=8)
    assert alpha[85-top, 80-left] == 1
    assert np.any((alpha > 0) & (alpha < 1))
    assert np.any(alpha == 0)
    binary = np.zeros(alpha.shape, np.uint8)
    cv2.fillConvexPoly(binary, np.round((polygon - [left, top]) * 16).astype(np.int32), 255, shift=4)
    assert np.all(alpha[binary == 255] == 1)


def test_masks_clip_cleanly_when_a_face_is_partly_outside_the_frame():
    image = np.full((50, 60, 3), 160, np.uint8)
    face = FaceRegion((-20, -10, 55, 65))
    result = blur_frame(image, [face.box], Settings(effect="solid"), faces=[face])
    assert result.shape == image.shape
    assert (result != image).any()
    assert polygon_mask(face_polygon(FaceRegion((200, 200, 40, 40))), 60, 50) is None


def test_rectangular_option_retains_full_box_coverage():
    face = reference_face()
    image = np.full((180, 180, 3), 160, np.uint8)
    result = blur_frame(image, [face.box], Settings(effect="solid", mask_shape="rectangle", margin=0, edge_softness=0), faces=[face])
    assert np.all(result[30:130, 40:120] == 18)
    assert np.array_equal(result[:20], image[:20])


def test_detector_landmarks_are_exposed_and_used_by_the_analyzer(face):
    detector = FaceDetector()
    boxes = detector.detect(face)
    assert len(detector.last_landmarks) == len(boxes)
    assert detector.last_landmarks[0].shape == (5, 2)
    analysis = FaceAnalyzer().analyze(face)
    assert len(analysis.faces) == len(analysis.boxes) > 0
    assert analysis.faces[0].landmarks.shape == (5, 2)
    assert analysis.faces[0].track_id > 0


def test_tracking_propagates_landmarks_through_a_detection_dropout():
    image = np.zeros((180, 220, 3), np.uint8)
    image[30:130, 40:120] = np.random.default_rng(2).integers(0, 256, (100, 80, 3), np.uint8)
    face = reference_face()
    tracker = FaceTracker(5)
    tracker.update([face.box], image, landmarks=[face.landmarks])
    transform = cv2.getRotationMatrix2D((80, 80), 4, 1.04)
    transform[:, 2] += [5, 3]
    moved = cv2.warpAffine(image, transform, (220, 180))
    tracker.update([], moved)
    expected = cv2.transform(face.landmarks[None], transform)[0]
    assert np.allclose(tracker.regions()[0].landmarks, expected, atol=1.5)


def test_landmark_jitter_is_stabilized_without_lagging_large_movements():
    face = reference_face()
    tracker = FaceTracker(5)
    frame = np.zeros((180, 220, 3), np.uint8)
    measured, stabilized = [], []
    for index in range(12):
        shift = 1 if index % 2 else -1
        points = face.landmarks + [shift, 0]
        box = (face.box[0] + shift, *face.box[1:])
        tracker.update([box], frame, landmarks=[points])
        measured.append(points[:, 0].mean())
        stabilized.append(tracker.regions()[0].landmarks[:, 0].mean())
    assert np.std(stabilized[3:]) < np.std(measured[3:])
    points = face.landmarks + [20, 0]
    tracker.update([(face.box[0] + 20, *face.box[1:])], frame, landmarks=[points])
    assert np.allclose(tracker.regions()[0].landmarks, points)
