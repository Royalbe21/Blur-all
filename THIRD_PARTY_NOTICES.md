# Third-party notices

## YuNet face detector

`blur_all/assets/face_detection_yunet_2023mar.onnx` is the OpenCV Zoo YuNet model, copyright Shiqi Yu, distributed under the MIT license in `blur_all/assets/YUNET-LICENSE.txt`.

Authoritative source: <https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet>.

The official Git LFS pointer declares SHA-256:

```
8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4
```

The application verifies this digest before loading the model.

For this checkout, the identical model bytes were retrieved from the ordinary Git blob at <https://github.com/leblancfg/autocrop/blob/0cdc398f4d608f6746d5ea4b5fafb961049a8559/autocrop/face_detection_yunet_2023mar.onnx> because the cloud network disallowed OpenCV's Git LFS media host. The bytes were accepted only after matching the official digest above.

## Runtime dependencies

- OpenCV: Apache 2.0; <https://opencv.org/license/>.
- NumPy: BSD 3-Clause; <https://numpy.org/doc/stable/license.html>.
- PySide6 / Qt for Python: LGPLv3 / GPLv3 or commercial terms, as applicable; <https://doc.qt.io/qtforpython-6/licenses.html>. Qt libraries are dynamically bundled in a directory distribution.
- imageio-ffmpeg wrapper: BSD 2-Clause; <https://github.com/imageio/imageio-ffmpeg>.
- The bundled FFmpeg executable has its own licensing, including GPL components such as libx264. Inspect its `-L` and `-buildconf` output. Sources and build instructions: <https://github.com/imageio/imageio-ffmpeg> and <https://github.com/imageio/ffmpeg-binaries>. FFmpeg source: <https://ffmpeg.org/download.html>; license information: <https://ffmpeg.org/legal.html>. Binary redistribution must satisfy the applicable upstream licenses and source-distribution requirements.

The Windows build copies installed dependency license files and FFmpeg license/build information into `third-party-licenses` beside the executable. This project does not relicense these dependencies.

Qt license texts in `blur_all/assets/qt-licenses` come from <https://github.com/qt/qtbase/tree/v6.8.3/LICENSES>.

## Test image

`tests/assets/astronaut.jpg` is a resized JPEG of the NASA astronaut Eileen Collins photograph distributed as `astronaut.png` by scikit-image. The original photograph is in the public domain as a NASA image.

The workspace screenshots in `docs/workspace.png`, `docs/distortion-workspace.png`, and `docs/export-workspace.png` and the reproducible benchmark use this image in generated demonstration clips.

Source: <https://github.com/scikit-image/scikit-image/blob/v0.19.3/skimage/data/astronaut.png>.
Attribution and provenance: <https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut>.
