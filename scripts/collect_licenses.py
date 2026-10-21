"""Collect dependency notices into the portable build folder."""

from importlib.metadata import distribution
from pathlib import Path
import shutil
import subprocess


def collect_licenses(output: Path, root: Path, ffmpeg: Path):
    destination = output / "third-party-licenses"
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / "THIRD_PARTY_NOTICES.md", destination / "THIRD_PARTY_NOTICES.md")
    shutil.copyfile(root / "blur_all" / "assets" / "YUNET-LICENSE.txt", destination / "YUNET-LICENSE.txt")
    shutil.copytree(root / "blur_all" / "assets" / "qt-licenses", destination / "Qt", dirs_exist_ok=True)
    for package in ["numpy", "opencv-python-headless", "PySide6-Essentials", "shiboken6", "imageio-ffmpeg"]:
        dist = distribution(package)
        target = destination / package
        target.mkdir(exist_ok=True)
        for index, file in enumerate(dist.files or []):
            if file.name.lower().startswith(("license", "copying", "notice")) or "/licenses/" in str(file).lower():
                source = Path(dist.locate_file(file))
                if source.is_file():
                    shutil.copyfile(source, target / f"{index}-{source.name}")
    for option, name in [("-L", "FFmpeg-license.txt"), ("-buildconf", "FFmpeg-build.txt")]:
        result = subprocess.run([str(ffmpeg), option], capture_output=True, check=True)
        (destination / name).write_bytes(result.stdout + result.stderr)
