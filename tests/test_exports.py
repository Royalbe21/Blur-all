import subprocess

import cv2
import numpy as np
import pytest

from blur_all.core import FaceAnalyzer, Settings, VideoError, blur_frame, export_dimensions, export_video, ffmpeg_executable


def read_frames(path):
    capture = cv2.VideoCapture(str(path))
    assert capture.isOpened()
    result = []
    while True:
        ok, image = capture.read()
        if not ok:
            break
        result.append(image)
    capture.release()
    return result


@pytest.mark.parametrize("container", ["mp4", "mov", "mkv"])
def test_containers_have_decodable_video_and_audio(container, video, tmp_path):
    source = tmp_path / "with-audio.mp4"
    subprocess.run([
        ffmpeg_executable(), "-v", "error", "-i", str(video), "-f", "lavfi", "-i",
        "sine=frequency=440:duration=1.2", "-c:v", "libx264", "-c:a", "aac", "-shortest", str(source),
    ], check=True)
    destination = tmp_path / ("result." + container)
    export_video(source, destination, Settings(output_format=container, audio_bitrate=128))
    frames = read_frames(destination)
    assert len(frames) == 12
    assert frames[0].shape == (320, 320, 3)
    header = destination.read_bytes()[:16]
    if container == "mkv":
        assert header.startswith(b'\x1a\x45\xdf\xa3')
    else:
        assert header[4:8] == b'ftyp'
    decoded = subprocess.run([
        ffmpeg_executable(), "-v", "error", "-i", str(destination), "-map", "0:a:0",
        "-f", "s16le", "-acodec", "pcm_s16le", "pipe:1",
    ], capture_output=True, check=True)
    assert len(decoded.stdout) > 44100
    assert np.abs(np.frombuffer(decoded.stdout, np.int16).astype(float)).mean() > 100


def test_quality_presets_change_real_output_size_and_encoding_error(video, tmp_path):
    original = read_frames(video)[0]
    analysis = FaceAnalyzer().analyze(original)
    expected = blur_frame(original, analysis.boxes, Settings(), faces=analysis.faces)
    errors, sizes = {}, {}
    for quality in ("high", "small"):
        output = tmp_path / (quality + ".mp4")
        export_video(video, output, Settings(export_quality=quality))
        decoded = read_frames(output)[0]
        errors[quality] = np.mean((decoded.astype(float) - expected) ** 2)
        sizes[quality] = output.stat().st_size
    assert sizes["small"] < sizes["high"]
    assert errors["small"] > errors["high"]


@pytest.mark.parametrize("width,height,expected", [(960, 540, (852, 480)), (540, 960, (480, 852))])
def test_output_resize_preserves_orientation_and_masks_before_downscaling(width, height, expected, tmp_path):
    source = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (width, height))
    assert writer.isOpened()
    for _ in range(3):
        writer.write(np.full((height, width, 3), 160, np.uint8))
    writer.release()
    output = tmp_path / "scaled.mp4"
    export_video(source, output, Settings(resolution="480p", effect="solid", manual_regions=((0.2, 0.2, 0.4, 0.4),)))
    frames = read_frames(output)
    assert len(frames) == 3
    out_w, out_h = expected
    assert frames[0].shape == (out_h, out_w, 3)
    region = frames[0][round(out_h*.25):round(out_h*.5), round(out_w*.25):round(out_w*.5)]
    assert np.abs(region.astype(float) - 18).mean() < 4


@pytest.mark.parametrize("width,height,resolution,expected", [
    (2160, 3840, "1080p", (1080, 1920)), (3840, 2160, "1080p", (1920, 1080)),
    (320, 240, "1080p", (320, 240)), (321, 241, "source", (322, 242)),
])
def test_dimensions_fit_selected_size_without_upscaling(width, height, resolution, expected):
    assert export_dimensions(width, height, resolution) == expected


def test_mismatched_container_and_invalid_quality_leave_no_output(video, tmp_path):
    target = tmp_path / "invalid.mp4"
    with pytest.raises(VideoError, match="format"):
        export_video(video, target, Settings(output_format="mkv"))
    with pytest.raises(VideoError, match="quality"):
        export_video(video, target, Settings(export_quality="unknown"))
    assert not target.exists()
    assert not list(tmp_path.glob(".blur-all-*"))
