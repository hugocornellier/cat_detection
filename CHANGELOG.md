## 2.0.0

* `CatDetector` now runs the whole pipeline in a background isolate that it owns.
  `initialize()` loads the model assets on the main isolate (where `rootBundle`
  is available) and transfers them into a worker it spawns, so detection no
  longer runs on the calling thread. This makes `CatDetector` the single entry
  point for the package.

* **Deprecated** `CatDetectorIsolate`. It is now a thin delegate to
  `CatDetector` and will be removed in the next major release. Migration is a
  rename: `CatDetectorIsolate.spawn(...)` becomes `CatDetector(...)` plus
  `await initialize()`, `detectCats` becomes `detect`, and `detectCatsFromMat`
  becomes `detectFromMat`. `onDownloadProgress` moves from `spawn()` to
  `initialize()`.

* `CatDetector.detectFromMat` now takes `imageWidth` and `imageHeight` as
  optional named arguments, defaulting to the Mat's own `cols` and `rows`.
  Existing call sites that pass them keep working.

* `CatDetector.initialize()` no longer accepts `useIsolateInterpreter`, and
  `initializeFromBuffers` is no longer part of the public API. The worker
  isolate owns interpreter creation, so neither had a meaningful effect on the
  public class. The buffer-based entry point now lives on the internal
  `CatDetectorCore`.

* `detThreshold` is now honored on the isolate path. The previous
  `CatDetectorIsolate` never forwarded it to the isolate, so a custom threshold
  was silently ignored and the pipeline ran at the 0.5 default.

* The example app now uses `CatDetector` with default (accelerated) performance
  settings instead of `CatDetectorIsolate` with `PerformanceConfig.disabled`.

* Require animal_detection 2.0.0, which replaces its boxed nested input and
  output tensors with reused flat `Float32List`s handed to TFLite as
  `ByteBuffer`s. Measured on this pipeline over a 3264x2448 photo in profile
  mode with `PerformanceMode.auto`, the full pipeline drops from 438 ms/frame to
  109 ms/frame and poseOnly from about 44 ms to 15 ms.

* Landmark and pose coordinates shift slightly. animal_detection 2.0.0 fixes
  `ImageUtils.cropAndResize` describing an integral crop with pre-truncation
  floats, which placed landmarks about 0.61px right and 0.52px down of ground
  truth. Measured over the 311-image CatFLW holdout with real localizer boxes,
  that cost 0.255 NME_IOD, rising to 1.14 at the 95th percentile, with 72% of
  images improving. `_pipelineVersion` is bumped to `pipeline_v3` accordingly,
  so downstream caches re-evaluate stored detections.

* `AnimalPoseModel.hrnet` now works. animal_detection was requesting
  `superanimal_hrnet_w32_256_float16.tflite` while its release publishes
  `superanimal_hrnet_w32_float16.tflite`, so selecting HRNet failed with an
  HTTP 404 on first use and had never worked.

## 1.5.0

* Replace the bundled face landmark model with the MobileNetV3Large 384px
  variant. The package asset drops from 57.3 MB to 11.6 MB (a 45.7 MB
  reduction) and accuracy improves: NME_IOD 3.51 vs 3.76 measured in image
  pixel space over the same 311 held-out CatFLW images, with TFLite invoke
  latency roughly halved (206 ms -> 103 ms, desktop CPU, 4 threads, XNNPACK).
  The previously bundled model was the first experiment of the training sweep
  and had been superseded by later runs.
* `CatLandmarkModel.full` now runs at 384px input instead of 256px. The input
  resolution is declared once as a constant rather than repeated at each call
  site, since the interpreter accepts a mismatched resize without erroring and
  then silently returns garbage coordinates.
* Bump `modelVersion` (`_packageVersion` 1.0.5 -> 1.5.0, `_pipelineVersion`
  pipeline_v1 -> pipeline_v2) so downstream caches invalidate detections
  produced by the previous model. `_packageVersion` had been stale since 1.0.5
  and did not track the four releases in between.

## 1.4.0

* Update animal_detection -> 1.4.0, which replaces its shipped 12,944-line SSD
  anchor table with runtime generation. Detection output is unchanged: verified
  against the real model over 9 images at 100 runs each with identical detection
  counts, bit-identical scores, and a worst-case box coordinate delta of
  9.3e-05 px. The shared library drops from 15,222 to 2,355 lines and the
  compiled binary shrinks by about 32 KB.
* Update flutter_litert -> 3.6.0.

## 1.3.3

* Update flutter_litert -> 3.5.0

## 1.3.2

* Update flutter_litert -> 3.4.1
* Update animal_detection -> 1.3.2

## 1.3.1

* Update flutter_litert -> 3.3.1

## 1.3.0

* Update flutter_litert -> 3.2.0
* Require animal_detection 1.3.0

## 1.2.3

* Update flutter_litert -> 3.1.1

## 1.2.2

* Update flutter_litert -> 3.1.0

## 1.2.1

* Update flutter_litert -> 2.8.3

## 1.2.0

* Update flutter_litert -> 2.8.0
* Complete Swift Package Manager migration: example apps build via SPM without CocoaPods

## 1.1.1

* Remove unused Darwin podspecs for Dart-only iOS/macOS plugin registration.
* Require animal_detection 1.1.1.

## 1.1.0

* Update animal_detection -> 1.1.0
* Update flutter_litert -> 2.5.8

## 1.0.13

* Update flutter_litert -> 2.5.5

## 1.0.12

* Update flutter_litert -> 2.5.4

## 1.0.11

* Update flutter_litert to 2.5.3

## 1.0.10

* Update flutter_litert -> 2.5.2

## 1.0.9

* Update flutter_litert -> 2.5.0 

## 1.0.8

* Update flutter_litert -> 2.4.1

## 1.0.7

* Update flutter_litert -> 2.4.0

## 1.0.6

* Update flutter_litert -> 2.3.0

## 1.0.5

* Add public `CatDetector.modelVersion` and `CatDetector.modelVersionFor(...)` APIs for downstream cache invalidation.

## 1.0.4

* Update flutter_litert -> 2.2.0
 
## 1.0.3

* Update flutter_litert -> 2.1.0

## 1.0.2

* Update flutter_litert to 2.0.13
* Update animal_detection to 1.0.2

## 1.0.1

* Update flutter_litert -> 2.0.12

## 1.0.0

* First stable release. On-device cat face detection and 48-point facial landmark prediction using TensorFlow Lite. Supports Android, iOS, macOS, Windows, and Linux with automatic hardware acceleration.

## 0.0.10

* Update documentation

## 0.0.9

* Update flutter_litert 2.0.8 -> 2.0.10

## 0.0.8

* Enable auto hardware acceleration by default (XNNPACK on all native platforms, Metal GPU on iOS)
* Update flutter_litert 2.0.6 -> 2.0.8
* Update animal_detection 0.0.5 -> 0.0.6

## 0.0.7

* Fix Android hang on sequential detect calls

## 0.0.6

* Fix isolate hanging on sequential detect calls

## 0.0.5

* Fix Windows build: rename private header include guard to avoid collision with public header

## 0.0.4

* Fix Windows build: export CatDetectionPluginRegisterWithRegistrar symbol

## 0.0.3

* Update animal_detection 0.0.3 -> 0.0.4

## 0.0.2

* Add Swift Package Manager support.

## 0.0.1

* Initial release with cat face detection and landmark prediction pipeline.
