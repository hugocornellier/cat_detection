# Cat Detection Flutter Package

## What this is

A Flutter plugin (`cat_detection`) that runs a multi-stage on-device TFLite pipeline for detecting cats in images. It mirrors the architecture of the sibling `dog_detection` package at `/Users/hugocornellier/IdeaProjects/dog_detection`.

## 5-Stage Pipeline

1. **SuperAnimal SSD** — body detection (via `animal_detection` package)
2. **Species classifier** — confirms cat species (via `animal_detection` package)
3. **RTMPose/HRNet body pose** — body keypoints (via `animal_detection` package)
4. **CatFaceLocalizer** — EfficientNetB2 regression model, letterbox 224×224 input, outputs normalized [x1,y1,x2,y2] face bbox. Trained on CatFLW dataset (2079 images, 48 landmarks). Val IoU: 81.5%.
5. **CatLandmarkModel** — EfficientNetV2S + heatmap deconv head, 256×256 input, predicts 48 facial landmarks. Val NME-IOD: 3.72%.

Stages 1-3 are delegated to the `animal_detection` package's `AnimalDetector`. Stages 4-5 are cat-specific and handled in this package.

## Key Files

- `lib/src/cat_detector.dart` — Main detector class. `_detectWithBody()` runs the full pipeline: animal detection → expand body bbox → crop → face localizer → offset to image coords → crop face → landmark model. Has `_expandBox()` helper matching the dog repo pattern.
- `lib/src/models/cat_face_localizer.dart` — EfficientNetB2 face bbox regression. Handles letterbox/de-letterbox internally.
- `lib/src/models/cat_landmark_model.dart` — Single-scale landmark model runner.
- `lib/src/models/ensemble_landmark_model.dart` — Multi-scale ensemble (256+320+384px) landmark model.
- `lib/src/isolate/cat_detector_isolate.dart` — Runs the full pipeline in a background isolate. Transfers all model bytes via `TransferableTypedData` for zero-copy. Loads localizer + landmark assets from `rootBundle` on main isolate, materializes them in worker isolate.
- `lib/src/types.dart` — Data types (`Cat`, `CatFace`, `CatLandmark`, `BoundingBox`, enums).
- `lib/src/util/image_utils.dart` — Crop/resize, letterbox, mat-to-float32 utilities.
- `assets/models/` — TFLite model files (localizer + landmarks).

## ML Training Repo

The models were trained in `/Users/hugocornellier/PycharmProjects/cats-in-the-wild-ml`:
- `scripts/train_cat_face_landmarks.py` — Landmark model training (EfficientNetV2S, CatFLW dataset)
- `scripts/train_cat_face_detector.py` — Face localizer training (EfficientNetB2, CIoU+L1 loss)
- `scripts/infer_cat_landmarks_tflite.py` — Two-stage Python inference script for testing
- `artifacts/cat_face_detector/cat_face_localizer_224_float16.tflite` — Trained face localizer
- `artifacts/tight_margin_256/` — Trained landmark model + metadata

## Recent Work (March 2026)

The face localizer stage was just integrated. Previously, `_detectWithBody()` used the body bbox directly as the face region (placeholder). Now it follows the dog repo's pattern:
1. Expand body bbox by `cropMargin`
2. Crop that region from the image
3. Run `CatFaceLocalizer.detect()` on the crop
4. Offset the detected face bbox back to original image coordinates
5. Pass the face bbox to `_runFaceLandmarks()` for landmark prediction

Changes made:
- `cat_detector.dart`: `initialize()` creates localizer, `initializeFromBuffers()` accepts + validates `localizerBytes`, `_detectWithBody()` uses localizer, added `_expandBox()`
- `cat_detector_isolate.dart`: `_IsolateStartupData` carries `localizerBytes`, `_initialize()` loads localizer from assets, `_isolateEntry()` materializes and passes it
- `pubspec.yaml`: Added `cat_face_localizer.tflite` asset entry
- Copied trained TFLite model to `assets/models/`

## Pre-existing Issues (not from our changes)

- `example/test/widget_test.dart` references non-existent `MyApp` class
- `assets/samples/` directory listed in pubspec but doesn't exist

## Git Rules

- NEVER include "Co-Authored-By" lines in commit messages
- NEVER run `dart pub publish --force` or `flutter pub publish --force`
