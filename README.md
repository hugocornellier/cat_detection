<h1 align="center">cat_detection</h1>

<p align="center">
<a href="https://flutter.dev"><img src="https://img.shields.io/badge/Platform-Flutter-02569B?logo=flutter" alt="Platform"></a>
<a href="https://dart.dev"><img src="https://img.shields.io/badge/language-Dart-blue" alt="Language: Dart"></a>
<br>
<a href="https://pub.dev/packages/cat_detection"><img src="https://img.shields.io/pub/v/cat_detection?label=pub.dev&labelColor=333940&logo=dart" alt="Pub Version"></a>
<a href="https://pub.dev/packages/cat_detection/score"><img src="https://img.shields.io/pub/points/cat_detection?color=2E8B57&label=pub%20points" alt="pub points"></a>
<a href="https://github.com/hugocornellier/cat_detection/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-007A88.svg?logo=apache" alt="License"></a>
</p>

![Demo](assets/screenshots/cat_detection_demo.png)

On-device cat detection using TFLite models. Detects cats in images with breed identification, body pose estimation, face localization, and 48-point facial landmarks.

## Features 

- Cat body detection with bounding box (SSD-based)
- Breed identification with confidence score
- Body pose estimation via SuperAnimal keypoints
- Face localization and 48-point facial landmark extraction (CatFLW)
- Truly cross-platform: compatible with Android, iOS, macOS, Windows, and Linux
- Configurable performance with XNNPACK, GPU, and CoreML acceleration

## Quick Start

```dart
import 'package:cat_detection/cat_detection.dart';

final detector = CatDetector(mode: CatDetectionMode.full);
await detector.initialize();

final cats = await detector.detect(imageBytes);
for (final cat in cats) {
  print('${cat.species} at ${cat.boundingBox}');
  print('Breed: ${cat.breed} (${(cat.speciesConfidence! * 100).toStringAsFixed(0)}%)');
  print('Pose keypoints: ${cat.pose?.landmarks.length}');
  print('Face landmarks: ${cat.face?.landmarks.length}');
}

await detector.dispose();
```

## Cat Face Landmarks (48-Point)

The `landmarks` property returns a list of 48 `CatLandmark` objects representing key points on the detected cat face.

### Landmark Groups

| Group | Count | Points |
|-------|-------|--------|
| Left ear | 5 | Ear contour |
| Right ear | 5 | Ear contour |
| Left eye | 7 | Eye corners, contour, and center |
| Right eye | 7 | Eye corners, contour, and center |
| Nose bridge | 2 | Bridge left and right |
| Nose ring | 4 | Nostril outline |
| Nose tips/wings | 4 | Nose tip and wing points |
| Mouth/chin | 10 | Lips, jaw, muzzle, and chin |
| Face contour | 4 | Face outline and muzzle center |

### Accessing Landmarks

```dart
final CatFace face = cat.face!;

// Iterate through all landmarks
for (final landmark in face.landmarks) {
  print('${landmark.type.name}: (${landmark.x}, ${landmark.y})');
}
```

## Breed Identification

In `full` and `poseOnly` modes, each detected cat includes a predicted breed label and confidence score from the species classifier.

```dart
final cats = await detector.detect(imageBytes);
for (final cat in cats) {
  if (cat.breed != null) {
    print('Breed: ${cat.breed}');
    print('Confidence: ${(cat.speciesConfidence! * 100).toStringAsFixed(1)}%');
  }
}
```

## Bounding Boxes

The `boundingBox` property returns a `BoundingBox` object representing the cat body bounding box in absolute pixel coordinates.

```dart
final BoundingBox boundingBox = cat.boundingBox;

// Access edges
final double left = boundingBox.left;
final double top = boundingBox.top;
final double right = boundingBox.right;
final double bottom = boundingBox.bottom;

// Calculate dimensions
final double width = boundingBox.right - boundingBox.left;
final double height = boundingBox.bottom - boundingBox.top;

print('Box: ($left, $top) to ($right, $bottom)');
print('Size: $width x $height');
```

## Model Details

| Model | Size | Input | Purpose |
|-------|------|-------|---------|
| Face localizer | 17 MB | 224x224 | Cat face detection and bounding box |
| Landmark model (full) | 11 MB | 384x384 | 48-point facial landmark extraction |

## Configuration Options

The `CatDetector` constructor accepts several configuration options:

```dart
final detector = CatDetector(
  mode: CatDetectionMode.full,               // Detection mode
  poseModel: AnimalPoseModel.rtmpose,        // Body pose model variant
  landmarkModel: CatLandmarkModel.full,      // Face landmark model variant
  cropMargin: 0.20,                          // Margin around detected body for crop
  detThreshold: 0.5,                         // SSD detection confidence threshold
  interpreterPoolSize: 1,                    // TFLite interpreter pool size
  performanceConfig: const PerformanceConfig(), // Auto acceleration
);
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `mode` | `CatDetectionMode` | `full` | Detection mode |
| `poseModel` | `AnimalPoseModel` | `rtmpose` | Body pose model variant |
| `landmarkModel` | `CatLandmarkModel` | `full` | Face landmark model variant |
| `cropMargin` | `double` | `0.20` | Margin around detected body crop (0.0-1.0) |
| `detThreshold` | `double` | `0.5` | SSD detection confidence threshold |
| `interpreterPoolSize` | `int` | `1` | TFLite interpreter pool size |
| `performanceConfig` | `PerformanceConfig` | `auto` | Interpreter hardware acceleration config |

## Detection Modes

| Mode | Features | Speed |
|------|----------|-------|
| **full** | Body detection + breed ID + body pose + face landmarks | Standard |
| **poseOnly** | Body detection + breed ID + body pose (no face) | Faster |

## Background Isolate Detection

Detection always runs in a background isolate. `CatDetector` spawns and owns that
isolate during `initialize()`, so the whole pipeline (decode, SSD, species, pose,
localizer, landmarks) stays off the main thread and the UI is never blocked.
There is nothing extra to opt into:

```dart
import 'package:cat_detection/cat_detection.dart';

// initialize() loads the models and spawns the worker isolate
final detector = CatDetector(mode: CatDetectionMode.full);
await detector.initialize();

// Runs in the background isolate; the UI thread stays free
final cats = await detector.detect(imageBytes);

for (final cat in cats) {
  print('${cat.breed} at ${cat.boundingBox}');
  print('Face landmarks: ${cat.face?.landmarks.length}');
}

// Tears down the isolate and frees the native interpreters
await detector.dispose();
```

Model bytes are transferred into the isolate with `TransferableTypedData`, so the
~70MB of weights in the default configuration move without being copied.

## Performance

### Hardware Acceleration

The package automatically selects the best acceleration strategy for each platform:

| Platform | Default Delegate | Speedup | Notes |
|----------|-----------------|---------|-------|
| **macOS** | XNNPACK | 2-5x | SIMD vectorization (NEON on ARM, AVX on x86) |
| **Linux** | XNNPACK | 2-5x | SIMD vectorization |
| **iOS** | Metal GPU | 2-4x | Hardware GPU acceleration |
| **Android** | XNNPACK | 2-5x | ARM NEON SIMD acceleration |
| **Windows** | XNNPACK | 2-5x | SIMD vectorization (AVX on x86) |

No configuration needed, just call `initialize()` and you get the optimal performance for your platform.

### Advanced Performance Configuration

```dart
// Auto mode (default), optimal for each platform
await detector.initialize();

// Force XNNPACK (all native platforms)
final detector = CatDetector(
  performanceConfig: PerformanceConfig.xnnpack(numThreads: 4),
);
await detector.initialize();

// Force GPU delegate (iOS recommended, Android experimental)
final detector = CatDetector(
  performanceConfig: PerformanceConfig.gpu(),
);
await detector.initialize();

// CPU-only (maximum compatibility)
final detector = CatDetector(
  performanceConfig: PerformanceConfig.disabled,
);
await detector.initialize();
```

### LiteRT Next CompiledModel

CompiledModel is opt-in and covers the active body, classification, pose,
face-localizer, and face-landmark stages:

```dart
// Try GPU first, with verified CPU/stage fallback.
await detector.initialize(useCompiledModel: true);

// Pin CompiledModel to CPU.
await detector.initialize(
  useCompiledModel: true,
  accelerators: {Accelerator.cpu},
);
```

Every requested compiled graph is compared with a plain-CPU Interpreter before
use. A numerically unsafe GPU graph retries on CompiledModel CPU; if that also
fails, only that stage uses Interpreter. Interpreter remains the default and
`Precision.fp32` is used unless explicitly overridden.

## Live Camera Detection

For real-time detection, pass each `camera` package image directly to the
detector. Packing happens on the caller, while color conversion, rotation,
downscaling, and inference stay in the detector worker isolate.

```dart
final cats = await detector.detectFromCameraImage(
  cameraImage,
  rotation: rotation,
  isBgra: Platform.isMacOS,
  maxDim: 640,
);
```

For lower-level integrations, use `prepareCameraFrame(...)` followed by
`detectFromCameraFrame(...)`.

## Credits

Models trained on the [CatFLW dataset](https://github.com/catflw/catflw).

## Example

The [sample code](https://pub.dev/packages/cat_detection/example) includes
matching live-camera, still-image, and video-file demos. All three paint body
pose and 48-point face landmarks; video output uses temporal smoothing and can
be replayed in the app.
