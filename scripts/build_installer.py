"""Compile a per-user Windows installer from the already tested portable folder."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from blur_all import __version__


def main():
    if sys.platform != "win32":
        raise SystemExit("Build the installer on Windows with Inno Setup 6 installed.")
    source = ROOT / "dist" / "BlurAll"
    if not (source / "BlurAll.exe").is_file():
        raise SystemExit("Build the portable app first: python scripts/build_windows.py")
    compiler = shutil.which("ISCC")
    if compiler is None:
        candidate = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Inno Setup 6" / "ISCC.exe"
        if candidate.is_file():
            compiler = str(candidate)
    if compiler is None:
        raise SystemExit("Install Inno Setup 6 from https://jrsoftware.org/isinfo.php, then retry.")
    subprocess.run([
        compiler, f"/DAppVersion={__version__}", f"/DSourceDir={source}",
        f"/DOutputDir={ROOT / 'dist' / 'installer'}", str(ROOT / "scripts" / "windows_installer.iss"),
    ], cwd=ROOT, check=True)
    installer = ROOT / "dist" / "installer" / f"BlurAll-Setup-{__version__}.exe"
    if not installer.is_file():
        raise SystemExit("The installer compiler did not produce the expected file.")
    print(f"Built {installer}")


if __name__ == "__main__":
    main()
