# Blur All

A Windows desktop application that blurs detected faces in videos. Processing happens locally: no uploads, accounts, or API keys. It produces a new H.264 MP4, with optional AAC audio, and never overwrites an existing file.

![Blur All video workspace](docs/workspace.png)

## Use on Windows

If you have a built release, extract the **whole** `BlurAll-Windows` archive and open `BlurAll.exe`. Keep the `_internal` folder beside the executable. Python and a separate FFmpeg installation are not needed for a packaged build.

To run from this source repository:

1. Install 64-bit Python **3.12** (or 3.11) from python.org, including the Python launcher.
2. Double-click `scripts\run_windows.bat`. Its first run installs the pinned dependencies into `.venv` and needs Internet access to PyPI. Subsequent app processing works offline.
3. Click **Open video** or drag a video into the window. Use **Space** to play/pause, **Left/Right** to step one frame, or the timeline to seek. Preview speed ranges from 0.25× to 2×. Preview audio has its own mute control; muting it does not change the export.
4. Choose **Soft blur**, **Pixelate**, or **Solid cover** in the sidebar. Adjust strength and extra face coverage. **Small faces** inspects a larger version of each frame; **Sensitive** detects more potential faces but can also cover non-face objects. **Show original** and **Detection outlines** help you review coverage.
5. For missed faces or other private details, enable **Draw a blur area** and drag on the paused preview. **Undo** / **Ctrl+Z** removes the last area; **Clear all** removes every drawn area. These rectangles remain fixed for the **entire video**; they do not follow a moving person. Automatic detection still runs alongside them.
6. Choose whether to **Keep original audio** and click **Export video**. Pick an unused `.mp4` filename. Processing speed and an estimated time remaining appear while encoding. Cancel safely at any time before completion.
7. Use **Review export** to open the completed file in your default player, then check it before sharing. The completion message reports how many frames contained face detections; zero detections is not proof that the video has no faces.

## Build the Windows program

Run on Windows using Python 3.12:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest -v
.venv\Scripts\python.exe scripts\build_windows.py
```

The program is written to `dist\BlurAll\BlurAll.exe`. Distribute the **entire `dist\BlurAll` folder** as a ZIP. The build includes the face model and FFmpeg. This is a portable application, not an installer, and it is not code-signed.

Alternatively, push this repository to GitHub and run **Actions → Windows build → Run workflow**. The workflow tests the application, builds it on Windows, runs a packaged smoke test, and uploads a `BlurAll-Windows` artifact. Download and extract that artifact. The workflow also runs on pull requests and pushes to `main`; it does not create a public release. Windows executables must be built on Windows; PyInstaller does not cross-compile from Linux.

## Development

Use the existing checkout: cloud tasks already have isolated workspaces; creating a Git worktree is unnecessary.

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m blur_all
```

Run tests (including real face detection, video exports, audio decoding, and Qt interaction):

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -v
QT_QPA_PLATFORM=offscreen .venv/bin/python -m blur_all --smoke-test
```

On Windows, use `.venv\Scripts\python.exe` and set `$env:QT_QPA_PLATFORM = 'offscreen'` in PowerShell when testing without a display. Remove that variable before launching the normal desktop interface. Linux requires Qt's usual system display libraries for an interactive desktop; the cloud environment can validate the UI offscreen.

The model is bundled and its SHA-256 is checked on load. The application does not download models at runtime. `imageio-ffmpeg` supplies the encoder binary; a separately installed FFmpeg is not required. Test clips are generated in temporary directories.

## Format support and limitations

- Common MP4, MOV, MKV, AVI, WebM, WMV, and MPEG videos work when OpenCV's bundled FFmpeg decoder supports their codec. DRM-protected, corrupt, and unsupported-codec videos cannot be processed. No program can promise support for every possible video format.
- YuNet detection runs on **every export frame**, with adjustable extra coverage around each face. Optical flow follows recent boxes during brief detection dropouts, and a scene-change heuristic clears old boxes. This short-term motion fallback does not identify people and cannot guarantee that every face will be covered. Very small, obscured, or extreme-profile faces may be missed.
- Decoding, detection, and masking for previews run on a background thread. Preview playback can skip frames to keep up on slower machines; export checks every decoded frame. Seek or step through important moments, then review the finished export. Preview imagery is limited to a 1280-pixel longest edge; exports keep the source resolution. Preview audio depends on the installed sound device and Qt's codec support; an unavailable preview audio backend does not prevent normal exports.
- Variable frame rate inputs are normalized to the decoder's reported average frame rate. Source timestamps are not preserved, so some phone/screen recordings can have audio synchronization drift. Review the output; convert to constant frame rate first if needed.
- Outputs are 8-bit SDR H.264 (`CRF 18`) with AAC audio when enabled. Audio is re-encoded, not bit-for-bit copied. HDR color, subtitles, chapters, alternate video tracks, and original metadata are not preserved. Odd frame dimensions are padded by one pixel for H.264 compatibility. Videos that change resolution midstream are rejected.
- Export needs space for a temporary silent MP4 and final MP4 in the destination folder. Only the completed file appears under your chosen output name; temporary files are cleaned after cancellation and errors. A hard process crash can leave a `.blur-all-*` temporary folder for manual removal.
- Automatic blurring and adjustable blur strength do **not** guarantee anonymity. Check all sensitive frames and audio before sharing. Increase coverage or draw additional areas when necessary.

See [third-party notices](THIRD_PARTY_NOTICES.md) for model, framework, encoder, and test-image attribution.
