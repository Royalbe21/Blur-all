from pathlib import Path
import subprocess
import threading

import cv2
import numpy as np
import pytest

from blur_all.core import (
    Cancelled, FaceDetector, FaceHold, Settings, VideoError,
    blur_frame, export_video, ffmpeg_executable, open_video, video_info,
)


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


def inspect(path):
    capture = open_video(path)
    try:
        info = video_info(capture)
        frames = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
        return info, frames
    finally:
        capture.release()


def test_real_detector_finds_face_and_blur_removes_detail(face):
    boxes = FaceDetector().detect(face)
    assert boxes, "The real bundled detector must detect the reference face"
    result = blur_frame(face, boxes, Settings())
    x, y, w, h = boxes[0]
    before = cv2.Laplacian(face[y:y+h, x:x+w], cv2.CV_64F).var()
    after = cv2.Laplacian(result[y:y+h, x:x+w], cv2.CV_64F).var()
    assert after < before * 0.1
    assert np.array_equal(result[-15:, -15:], face[-15:, -15:])


def test_multiple_faces_at_different_sizes(face):
    composite = np.zeros((320, 640, 3), dtype=np.uint8)
    composite[:, :320] = face
    composite[80:240, 400:560] = cv2.resize(face, (160, 160))
    boxes = FaceDetector().detect(composite)
    assert len(boxes) >= 2
    assert any(x < 320 for x, _, _, _ in boxes)
    assert any(x >= 320 for x, _, _, _ in boxes)
    blurred = blur_frame(composite, boxes, Settings())
    for x, y, w, h in boxes:
        before = cv2.Laplacian(composite[y:y+h, x:x+w], cv2.CV_64F).var()
        after = cv2.Laplacian(blurred[y:y+h, x:x+w], cv2.CV_64F).var()
        assert after < before * 0.1


def test_export_uses_real_detector_and_preserves_frame_count(video, tmp_path):
    original = video.read_bytes()
    output = tmp_path / "blurred.mp4"
    updates = []
    result = export_video(video, output, Settings(), progress=lambda *args: updates.append(args))
    info, frames = inspect(output)
    assert result.frames == len(frames) == 12
    assert result.frames_with_faces == 12
    assert info.fps == pytest.approx(10)
    assert video.read_bytes() == original
    assert updates[-1] == (12, 12, "Complete")
    assert not list(tmp_path.glob(".blur-all-*"))
    _, originals = inspect(video)
    x, y, w, h = FaceDetector().detect(originals[0])[0]
    before = cv2.Laplacian(originals[0][y:y+h, x:x+w], cv2.CV_64F).var()
    after = cv2.Laplacian(frames[0][y:y+h, x:x+w], cv2.CV_64F).var()
    assert after < before * 0.15


def test_audio_is_preserved_and_can_be_removed(video, tmp_path):
    with_audio = tmp_path / "音声 input.mp4"
    subprocess.run([
        ffmpeg_executable(), "-v", "error", "-i", str(video), "-f", "lavfi",
        "-i", "sine=frequency=440:sample_rate=44100:duration=1.2", "-c:v", "libx264",
        "-c:a", "aac", "-shortest", str(with_audio),
    ], check=True)
    for keep in [True, False]:
        output = tmp_path / f"audio-{keep}.mp4"
        export_video(with_audio, output, Settings(keep_audio=keep))
        # Require an actual decodable audio stream; return code alone isn't enough.
        decoded = subprocess.run([
            ffmpeg_executable(), "-v", "error", "-i", str(output), "-map", "0:a:0",
            "-f", "s16le", "-acodec", "pcm_s16le", "pipe:1",
        ], capture_output=True)
        if keep:
            assert decoded.returncode == 0
            assert len(decoded.stdout) > 44100
            samples = np.frombuffer(decoded.stdout, dtype=np.int16)
            assert np.abs(samples.astype(float)).mean() > 100
        else:
            assert decoded.returncode != 0
        assert len(inspect(output)[1]) == 12


def test_cancel_cleans_up_and_leaves_original(video, tmp_path):
    cancel = threading.Event()
    original = video.read_bytes()
    output = tmp_path / "cancelled.mp4"

    def progress(*_):
        cancel.set()

    with pytest.raises(Cancelled):
        export_video(video, output, Settings(), cancel, progress)
    assert not output.exists()
    assert video.read_bytes() == original
    assert not list(tmp_path.glob(".blur-all-*"))


def test_cancel_during_audio_mux_cleans_up(video, tmp_path):
    cancel = threading.Event()

    def progress(current, total, stage):
        if stage == "Finishing audio and MP4":
            cancel.set()

    output = tmp_path / "cancelled-audio.mp4"
    with pytest.raises(Cancelled):
        export_video(video, output, Settings(), cancel, progress)
    assert not output.exists()
    assert not list(tmp_path.glob(".blur-all-*"))


def test_refuses_to_overwrite_or_replace_original(video, tmp_path):
    original = video.read_bytes()
    with pytest.raises(VideoError, match="original"):
        export_video(video, video, Settings())
    existing = tmp_path / "existing.mp4"
    existing.write_bytes(b"keep this")
    with pytest.raises(VideoError, match="already exists"):
        export_video(video, existing, Settings())
    assert existing.read_bytes() == b"keep this"
    assert video.read_bytes() == original


def test_manual_region_clips_and_blurs_without_detections():
    frame = np.random.default_rng(5).integers(0, 256, (100, 100, 3), dtype=np.uint8)
    result = blur_frame(frame, [], Settings(manual_regions=((-0.1, -0.1, 0.6, 0.6),)))
    assert not np.array_equal(frame[:50, :50], result[:50, :50])
    assert np.array_equal(frame[50:, 50:], result[50:, 50:])


def test_detection_hold_expires_and_resets_on_scene_cut():
    hold = FaceHold(3)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    box = (10, 10, 20, 20)
    assert hold.update([box], frame) == [box]
    assert hold.update([], frame) == [box]
    assert hold.update([], frame) == [box]
    assert hold.update([], frame) == []
    hold.update([box], frame)
    assert hold.update([], np.full_like(frame, 255)) == []


def test_motion_tracking_follows_a_face_during_detection_dropout():
    hold = FaceHold(4)
    first = np.zeros((160, 240, 3), np.uint8)
    texture = np.random.default_rng(42).integers(0, 256, (60, 60, 3), np.uint8)
    first[40:100, 40:100] = texture
    second = np.zeros_like(first)
    second[45:105, 52:112] = texture
    hold.update([(40, 40, 60, 60)], first)
    box = hold.update([], second)[0]
    assert box[0] == pytest.approx(52, abs=2)
    assert box[1] == pytest.approx(45, abs=2)


@pytest.mark.parametrize("effect", ["pixelate", "solid", "distort"])
def test_mask_effects_change_only_the_selected_region(effect):
    image = np.random.default_rng(4).integers(0, 256, (100, 100, 3), np.uint8)
    result = blur_frame(image, [], Settings(effect=effect, manual_regions=((0.2, 0.2, 0.4, 0.4),)))
    assert np.array_equal(result[:20], image[:20])
    assert not np.array_equal(result[20:60, 20:60], image[20:60, 20:60])
    if effect == "solid":
        assert np.all(result[20:60, 20:60] == 18)


def test_distortion_is_stable_subtle_and_does_not_change_edges():
    x, y = np.meshgrid(np.arange(160), np.arange(120))
    image = np.stack([x, y, (x + y) // 2], axis=2).astype(np.uint8)
    region = ((0, 0, 1, 1),)
    mild = blur_frame(image, [], Settings(effect="distort", strength=25, manual_regions=region))
    strong = blur_frame(image, [], Settings(effect="distort", strength=100, manual_regions=region))
    assert np.array_equal(mild, blur_frame(image, [], Settings(effect="distort", strength=25, manual_regions=region)))
    assert np.array_equal(mild[0], image[0])
    assert np.array_equal(mild[:, -1], image[:, -1])
    mild_difference = np.abs(mild.astype(float) - image).mean()
    strong_difference = np.abs(strong.astype(float) - image).mean()
    assert 0 < mild_difference < strong_difference


@pytest.mark.parametrize("effect", ["pixelate", "distort"])
def test_alternative_effects_are_used_in_real_exports(effect, video, tmp_path):
    output = tmp_path / f"{effect}.mp4"
    result = export_video(video, output, Settings(effect=effect, strength=80))
    _, frames = inspect(output)
    assert result.frames == len(frames) == 12
    assert result.frames_with_faces == 12
    original = inspect(video)[1][0]
    x, y, w, h = FaceDetector().detect(original)[0]
    difference = np.abs(frames[0][y:y+h, x:x+w].astype(float) - original[y:y+h, x:x+w]).mean()
    assert difference > 5


def test_balanced_tracking_exports_all_frames_and_preserves_coverage(video, tmp_path):
    output = tmp_path / "balanced.mp4"
    result = export_video(video, output, Settings(effect="solid", detection_interval=2))
    _, frames = inspect(output)
    assert result.frames == len(frames) == 12
    assert result.frames_with_faces == 12
    assert result.detector_frames < result.frames
    original = inspect(video)[1][0]
    x, y, w, h = FaceDetector().detect(original)[0]
    for frame in frames:
        assert np.abs(frame[y:y+h, x:x+w].astype(float) - 18).mean() < 4


def test_pixelation_replaces_fine_texture_with_larger_blocks_at_higher_strength():
    image = np.random.default_rng(10).integers(0, 256, (100, 150, 3), np.uint8)
    settings = dict(effect="pixelate", manual_regions=((0, 0, 1, 1),))
    fine = blur_frame(image, [], Settings(strength=10, **settings))
    coarse = blur_frame(image, [], Settings(strength=90, **settings))
    fine_colors = len(np.unique(fine.reshape(-1, 3), axis=0))
    coarse_colors = len(np.unique(coarse.reshape(-1, 3), axis=0))
    assert coarse_colors < fine_colors / 3
    assert coarse_colors <= 40
    assert np.array_equal(coarse[1, 1], coarse[5, 5])


def test_bad_input_is_reported(tmp_path):
    broken = tmp_path / "bad.mp4"
    broken.write_bytes(b"not a video")
    with pytest.raises(VideoError, match="Cannot open"):
        export_video(broken, tmp_path / "out.mp4", Settings())


def test_pre_cancelled_export_has_no_output(video, tmp_path):
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(Cancelled):
        export_video(video, tmp_path / "out.mp4", Settings(), cancel)
