"""Reproducible comparison with a Git revision; no external dataset download.

Run: .venv/bin/python scripts/benchmark_tracking.py --baseline 0029f23
Synthetic transforms provide known face-box geometry; these are not real-world
privacy guarantees or a substitute for testing representative user videos.
"""

import argparse
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import types

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from blur_all.core import FaceAnalyzer, FaceDetector, MODEL_PATH
from blur_all.tracking import FaceTracker, bounds, corners, overlap


def baseline_module(ref):
    source = subprocess.check_output(["git", "show", f"{ref}:blur_all/core.py"], cwd=ROOT, text=True)
    module = types.ModuleType("blur_all_benchmark_baseline")
    module.__file__ = str(ROOT / "blur_all" / "core.py")
    sys.modules[module.__name__] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    module.MODEL_PATH = MODEL_PATH
    return module


def measure(factory, frames, repeats):
    durations, outputs, calls = [], [], 0
    for _ in range(repeats):
        process = factory()
        started = time.perf_counter()
        outputs = [process(frame) for frame in frames]
        durations.append(time.perf_counter() - started)
        calls = getattr(process, "detector_calls", lambda: len(frames))()
    return statistics.median(durations), outputs, calls


def video_sequence(face, count):
    base_box = FaceDetector().detect(face)[0]
    frames, ground_truth = [], []
    for index in range(count):
        size = round(250 + 12 * np.sin(index / 12))
        image = cv2.resize(face, (size, size))
        x, y = round(55 + index * 2.4), round(45 + 12 * np.sin(index / 9))
        canvas = np.full((360, 640, 3), 24, np.uint8)
        canvas[y:y+size, x:x+size] = image
        frames.append(canvas)
        scale = size / face.shape[0]
        ground_truth.append((round(x + base_box[0] * scale), round(y + base_box[1] * scale), round(base_box[2] * scale), round(base_box[3] * scale)))
    return frames, ground_truth


def recovery_sequence(count):
    base = np.zeros((300, 500, 3), np.uint8)
    base[80:176, 80:176] = np.random.default_rng(18).integers(0, 256, (96, 96, 3), np.uint8)
    frames, expected = [], []
    for index in range(count):
        transform = cv2.getRotationMatrix2D((128, 128), 18 * np.sin(index / 12), 1 + 0.18 * np.sin(index / 10))
        transform[:, 2] += [index * 2, 8 * np.sin(index / 9)]
        frames.append(cv2.warpAffine(base, transform, (500, 300)))
        expected.append(bounds(cv2.transform(corners((80, 80, 96, 96))[None], transform)[0], 500, 300))
    return frames, expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="0029f23")
    parser.add_argument("--frames", type=int, default=60, choices=range(24, 91))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    cv2.setNumThreads(4)
    old = baseline_module(options.baseline)
    face = cv2.imread(str(ROOT / "tests" / "assets" / "astronaut.jpg"))
    if face is None:
        raise SystemExit("Reference image is missing")
    FaceAnalyzer().analyze(face)  # Warm model/runtime loading before timing.
    report = {
        "baseline": subprocess.check_output(["git", "rev-parse", options.baseline], cwd=ROOT, text=True).strip(),
        "python": platform.python_version(), "opencv": cv2.__version__,
        "os": platform.system(), "opencv_threads": 4,
        "frames_per_run": options.frames, "repeats": options.repeats,
        "method": "Median wall time. Generated motion and known transformed face boxes; not a real-world accuracy evaluation.",
    }

    # Isolate tracker cost when the detector successfully supplies fresh boxes.
    frame = np.random.default_rng(8).integers(0, 256, (720, 1280, 3), np.uint8)
    boxes = [(100, 100, 100, 120), (550, 240, 80, 100), (1000, 400, 60, 80)]
    def tracker_factory(cls):
        def factory():
            tracker = cls(15)
            return lambda image: tracker.update(boxes, image)
        return factory
    before, _, _ = measure(tracker_factory(old.FaceHold), [frame] * options.frames, options.repeats)
    after, _, _ = measure(tracker_factory(FaceTracker), [frame] * options.frames, options.repeats)
    report["fresh_detections_tracking"] = {"baseline_ms_per_frame": before / options.frames * 1000, "current_ms_per_frame": after / options.frames * 1000, "speedup": before / after}

    frames, truth = recovery_sequence(options.frames)
    recovery = {}
    for name, cls in [("baseline", old.FaceHold), ("current", FaceTracker)]:
        tracker, overlaps = cls(15), []
        for index, image in enumerate(frames):
            boxes = tracker.update([truth[index]] if index % 6 == 0 else [], image)
            overlaps.append(max((overlap(box, truth[index]) for box in boxes), default=0))
        recovery[name] = {"mean_box_iou": float(np.mean(overlaps)), "fraction_iou_at_least_0_75": float(np.mean(np.asarray(overlaps) >= 0.75))}
    report["rotation_scale_detection_gaps"] = recovery

    frames, truth = video_sequence(face, options.frames)
    def previous_factory():
        detector, tracker = old.FaceDetector(), old.FaceHold(15)
        return lambda image: tracker.update(detector.detect(image), image)

    def current_factory(interval):
        def factory():
            analyzer = FaceAnalyzer(fps=30, detection_interval=interval)
            def process(image):
                return analyzer.analyze(image).boxes
            process.detector_calls = lambda: analyzer.detector_frames
            return process
        return factory

    cases = [("baseline", previous_factory)] + [(name, current_factory(interval)) for name, interval in [("accurate", 1), ("balanced", 2), ("fast", 4)]]
    results = {}
    for name, factory in cases:
        elapsed, outputs, calls = measure(factory, frames, options.repeats)
        coverage = [max((overlap(box, expected) for box in boxes), default=0) for boxes, expected in zip(outputs, truth)]
        results[name] = {"ms_per_frame": elapsed / options.frames * 1000, "fps": options.frames / elapsed, "detector_calls": calls, "mean_box_iou": float(np.mean(coverage)), "frames_with_iou_at_least_0_5": sum(value >= 0.5 for value in coverage)}
    for name in results:
        results[name]["speedup_over_baseline"] = results["baseline"]["ms_per_frame"] / results[name]["ms_per_frame"]
    report["detection_and_tracking_640x360"] = results
    text = json.dumps(report, indent=2)
    print(text)
    if options.output:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
