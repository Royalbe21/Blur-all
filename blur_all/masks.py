"""Landmark-guided face contours and outward-only feathered coverage."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FaceRegion:
    box: tuple[int, int, int, int]
    track_id: int = 0
    quad: np.ndarray | None = None
    landmarks: np.ndarray | None = None


def face_polygon(face: FaceRegion, margin: float = 0.15, shape: str = "face") -> np.ndarray:
    x, y, w, h = face.box
    if shape == "rectangle":
        return np.array([[x-w*margin, y-h*margin], [x+w+w*margin, y-h*margin], [x+w+w*margin, y+h+h*margin], [x-w*margin, y+h+h*margin]], np.float32)
    center = np.array([x + w / 2, y + h / 2], np.float32)
    horizontal, vertical = np.array([1, 0], np.float32), np.array([0, 1], np.float32)
    width, height = float(w), float(h)
    if face.quad is not None:
        edge = face.quad[1] - face.quad[0]
        length = np.linalg.norm(edge)
        if length > 1:
            horizontal = edge / length
            vertical = np.array([-horizontal[1], horizontal[0]])
            width = float(length)
            height = float(np.linalg.norm(face.quad[3] - face.quad[0]))
            center = face.quad.mean(axis=0)
    landmarks = face.landmarks
    if shape == "face" and landmarks is not None and landmarks.shape == (5, 2) and np.isfinite(landmarks).all():
        eyes, mouth = landmarks[:2], landmarks[3:5]
        eye_axis = eyes[1] - eyes[0]
        eye_distance = np.linalg.norm(eye_axis)
        if eye_distance > max(2, w * 0.12):
            horizontal = eye_axis / eye_distance
            if horizontal[0] < 0:
                horizontal = -horizontal
            vertical = np.array([-horizontal[1], horizontal[0]])
            eye_center, mouth_center = eyes.mean(axis=0), mouth.mean(axis=0)
            gap = float(np.dot(mouth_center - eye_center, vertical))
            if gap > 1:
                # The detector's bounds constrain the inferred forehead and jaw;
                # five landmarks guide a contour rather than a pixel-level matte.
                width = max(width * 0.94, float(eye_distance) * 2.5)
                height = max(height, gap * 2.85)
                center = eye_center + vertical * gap * 0.38
                center += horizontal * np.clip(np.dot(landmarks[2] - eye_center, horizontal) * 0.15, -width * 0.07, width * 0.07)
    theta = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    u, v = np.cos(theta), np.sin(theta)
    if shape == "face":
        # Round forehead and cheeks, taper gently toward the chin.
        u = u * (1 - 0.13 * np.maximum(v, 0) - 0.04 * np.maximum(-v, 0))
    half_width, half_height = width * (0.5 + margin), height * (0.54 + margin)
    return (center + u[:, None] * horizontal * half_width + v[:, None] * vertical * half_height).astype(np.float32)


def polygon_mask(polygon: np.ndarray, frame_width: int, frame_height: int, feather: float = 0):
    """Return a clipped ROI and alpha; the contour's interior stays fully opaque."""
    left, top = np.maximum(0, np.floor(polygon.min(axis=0) - feather - 2)).astype(int)
    right, bottom = np.minimum([frame_width, frame_height], np.ceil(polygon.max(axis=0) + feather + 3)).astype(int)
    if right <= left or bottom <= top:
        return None
    mask = np.zeros((bottom-top, right-left), np.uint8)
    local = np.round((polygon - [left, top]) * 16).astype(np.int32)
    cv2.fillConvexPoly(mask, local, 255, shift=4)
    alpha = mask.astype(np.float32) / 255
    if feather > 0:
        distance = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)
        t = np.clip(1 - distance / feather, 0, 1)
        alpha = np.maximum(alpha, t * t * (3 - 2 * t))
    return (int(left), int(top), int(right), int(bottom)), alpha
