"""Build on Windows; PyInstaller does not cross-compile Windows executables."""

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    if sys.platform != "win32":
        raise SystemExit("Run this script on Windows, or use the Windows build GitHub Actions workflow.")
    from blur_all.core import FaceDetector, ffmpeg_executable

    FaceDetector()  # Check the bundled model's integrity before packaging.
    ffmpeg = Path(ffmpeg_executable())
    subprocess.run([
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--windowed",
        "--onedir", "--name", "BlurAll", "--contents-directory", "_internal",
        "--paths", str(ROOT),
        "--add-data", f"{ROOT / 'blur_all' / 'assets'}{os.pathsep}blur_all/assets",
        "--add-binary", f"{ffmpeg}{os.pathsep}imageio_ffmpeg/binaries",
        "--copy-metadata", "imageio-ffmpeg",
        str(ROOT / "blur_all" / "__main__.py"),
    ], cwd=ROOT, check=True)
    from collect_licenses import collect_licenses

    collect_licenses(ROOT / "dist" / "BlurAll", ROOT, ffmpeg)
    import shutil

    shutil.copyfile(ROOT / "README.md", ROOT / "dist" / "BlurAll" / "README.md")
    print("Built dist/BlurAll/BlurAll.exe. Distribute the ENTIRE BlurAll folder.")


if __name__ == "__main__":
    main()
