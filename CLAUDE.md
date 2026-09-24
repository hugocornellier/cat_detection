# Cat Detection Flutter Package

## What this is

A Flutter plugin (`cat_detection`) that runs a multi-stage on-device TFLite pipeline for detecting cats in images. It mirrors the architecture of the sibling `dog_detection` package at `/Users/hugocornellier/IdeaProjects/dog_detection`.

## 5-Stage Pipeline

1. **SuperAnimal SSD**, body detection (via `animal_detection` package)
2. **Species classifier**, confirms cat species (via `animal_detection` package)
3. **RTMPose/HRNet body pose**, body keypoints (via `animal_detection` package)
4. **CatFaceLocalizer**, EfficientNetB2 regression model, letterbox 224×224 input, outputs normalized [x1,y1,x2,y2] face bbox. Trained on CatFLW dataset (2079 images, 48 landmarks). Val IoU: 81.5%.
5. **CatLandmarkModel**, MobileNetV3-Large + 4x Conv2DTranspose heatmap head + SoftArgmax2D, 384×384 input, predicts 48 facial landmarks. Val NME-IOD: 3.48 on the 311-image CatFLW split. Shipped as float16, static-shape (batch 1), with the deconv ReLU unfused so TRANSPOSE_CONV stays at version 3 and the GPU can run the whole graph.

Stages 1-3 are delegated to the `animal_detection` package's
`AnimalDetectorCore`. Stages 4-5 are cat-specific and handled in this package.

## Key Files

- `lib/cat_detection.dart`, Public exports, including the `animal_detection` types the API uses.
- `lib/src/cat_detector.dart`, Public detector and owned worker-isolate RPC.
- `lib/src/isolate/cat_detector_core.dart`, In-worker five-stage pipeline. Runs the face localizer and landmark model through `animal_detection`'s `FaceLocalizerModel` and `LandmarkModelRunnerBase`.
- `lib/src/types.dart`, Data types (`Cat`, `CatFace`, `CatLandmark`, enums).
- `assets/models/`, `cat_face_localizer.tflite` + `cat_face_landmarks_full.tflite` (CC BY-NC 4.0, see `LICENSE.md` there) and `species_mapping.json`.

## ML Training Repo

The models were trained in `/Users/hugocornellier/PycharmProjects/cats-in-the-wild-ml` (public as cat-face-landmarks-training):
- `scripts/train_cat_face_landmarks.py`, Landmark model training (CatFLW dataset; the shipped backbone is MobileNetV3-Large)
- `scripts/train_cat_face_detector.py`, Face localizer training (EfficientNetB2, CIoU+L1 loss)
- `scripts/infer_cat_landmarks_tflite.py`, Two-stage Python inference script for testing
- `artifacts/cat_face_detector/cat_face_localizer_224_float16.tflite`, Trained face localizer
- `artifacts/small_v3large_384_long/`, The shipped landmark model + metadata
- `scripts/reexport_static.py`, Static, GPU-ready `.tflite` export (batch 1, deconv ReLU unfused). Read "READ THIS BEFORE EXPORTING ANY MODEL" in that repo's `LANDMARK_DETECTION_REPORT.md` first.

## Git Rules

- NEVER include "Co-Authored-By" lines in commit messages
- NEVER run `dart pub publish --force` or `flutter pub publish --force`
