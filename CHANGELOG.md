## 3.0.1

* **Re-exported both face models with static shapes so GPU backends can run
  them.** Trained weights are unchanged; only the export path changed. The
  previous models were converted with `from_keras_model`, which leaves the
  batch dimension dynamic and emits SHAPE / STRIDED_SLICE / PACK in the graph
  tail. Every GPU backend refuses a graph with dynamic-sized tensors, so both
  stages ran on CPU on every platform, and switching between Interpreter and
  CompiledModel changed nothing because both fell back to the same CPU path.
  Converting from a batch-1 concrete function removes those ops
  (cat_face_localizer 689 to 592 ops, cat_face_landmarks_full 295 to 283).
  The landmark model additionally has its deconv ReLU moved out of
  TRANSPOSE_CONV into a separate RELU op, dropping the opcode from version 4
  to 3, which is what lets CompiledModel's GPU accelerator claim the head.
* Output parity against the 3.0.0 models is 3.99e-06 (localizer) and 4.17e-07
  (landmarks), so detection quality is unchanged.
* **`useCompiledModel` now defaults to true.** The re-exported graphs are
  accepted by CompiledModel with the default `{gpu, cpu}` accelerator set,
  which is the fastest configuration measured on every Apple platform. The
  existing per-stage try/catch still falls back to the Interpreter if
  CompiledModel construction fails.
* **The landmark stage now defaults to the GPU delegate instead of auto.**
  XNNPACK claims the deconv region with a kernel slower than TFLite's built-in
  ruy one, so on the re-exported graph XNNPACK is slower than bare CPU. Auto
  resolves to XNNPACK on Android, macOS, Linux and Windows, so leaving it on
  auto would have made the landmark stage slower than 3.0.0. Pass
  `landmarkPerformanceConfig` to override. Platforms with no GPU delegate fall
  through to bare CPU, which measures the same as 3.0.0 on this graph.
* Require `animal_detection` ^3.0.1, which ships its live-camera APIs and
  updated native example and restores complete pub.dev package analysis.
* Measured on macOS M4 Max, flutter_litert 3.9.0, 25 iterations after 8 warmup,
  median of `sync_p50_ms`:

  | stage | 3.0.0 best | 3.0.1 best |
  | --- | --- | --- |
  | localizer | 8.11 ms (XNNPACK) | 1.70 ms (CompiledModel {gpu, cpu}) |
  | landmarks | 27.10 ms (XNNPACK) | 3.82 ms (CompiledModel {gpu, cpu}) |

  Measured on iPhone 15 Pro, iOS 26.5, same protocol:

  | stage | 3.0.0 best | 3.0.1 best |
  | --- | --- | --- |
  | localizer | 15.80 ms | 3.05 ms |
  | landmarks | 47.48 ms | 11.66 ms |

## 3.0.0

* Add opt-in LiteRT Next CompiledModel support to `CatDetector.initialize()`
  and the new `CatDetector.create()`. `useCompiledModel` defaults to false;
  `accelerators` defaults to GPU with CPU fallback and `precision` to fp32.
* Route the selected backend through the worker isolate and every active stage:
  animal body detection, species classification, body pose, face localization,
  and face landmarks. Each compiled graph is numerically verified; unsafe
  graphs retry on CPU or fall back only that stage to Interpreter.
* Add a macOS full-pipeline parity integration test comparing body boxes and
  scores, species, every pose point, the face box, and all 48 face landmarks in
  Interpreter, CompiledModel CPU, and requested GPU+CPU modes.
* Remove the deprecated `CatDetectorIsolate`. `CatDetector` has owned its
  background isolate since 2.0.0 and is now the package's only detector class.
* Require `animal_detection` ^3.0.0 and keep `flutter_litert` ^3.8.0.
* Add `detectFromCameraFrame()` and `detectFromCameraImage()`, keeping camera
  pixel conversion, rotation, downscaling, and inference in the detector
  worker. The native example now matches the face, pose, and hand examples with
  live camera, still image, and smoothed video-file demos.

## 2.1.0

* **Default precision is now `Precision.fp32` instead of `fp16`.** This changes
  numeric output. `flutter_litert` 3.8.0 changed its own default for the same
  reason: across 29 published detection models measured on five GPUs, fp16
  matched a plain-CPU reference for only about a fifth of them, while fp32
  matched every model that compiled. These graphs emit pixel-space coordinates
  and landmark positions, and fp16 carries about three decimal digits of
  mantissa, so the error lands directly on output geometry. The cost is real and
  worth stating plainly: fp32 is a median 29.9% slower on GPU across those five
  GPUs, with Apple M4 the lone exception at 6.5% faster. Pass
  `precision: Precision.fp16` explicitly to restore the previous behaviour,
  ideally per model and validated on your target GPU.
* Pin `flutter_litert` to `^3.8.0`.

## 2.0.0

* Added `CatDetectionMode.faceOnly`, matching the mode dog_detection already
  had. It runs the face localizer on the whole letterboxed image and then the
  landmark model, skipping SSD body detection, species classification and body
  pose. Those three stages account for about 23MB of model weights that are
  never loaded and, measured on a 3264x2448 photo, roughly 16ms per frame
  (118.1 ms to 101.7 ms).

  Running the localizer on the whole image is what it was trained for; `full`
  instead runs it inside an SSD body crop. The localizer emits a single box, so
  `faceOnly` returns at most one face however many cats are present, and the
  returned `Cat` has no species, breed or pose. `full` is unchanged and remains
  the default, still returning body box, species, body pose and face landmarks
  together.

  Adding an enum value is breaking for exhaustive switches over
  `CatDetectionMode`, which is why it lands in this release.

* **Removed** `CatLandmarkModel.ensemble`. The mode required two extra models
  from a GitHub release that was never published, so selecting it always failed
  with an HTTP 404 and it has never worked. Rebuilding it was not worthwhile:
  the only 256px and 320px cat models available are EfficientNetV2S, which
  measured 156 ms and 242 ms per inference against the bundled MobileNetV3Large
  model's 83 ms. A three-model ensemble with flip TTA would have cost roughly
  960 ms/frame and 109 MB of downloads for an accuracy that was never measured
  in that mixed-backbone configuration. `CatDetector.isEnsembleCached()` is
  removed with it. `CatLandmarkModel.full` is unchanged and remains the default.

* **Renamed two groups of `CatLandmarkType` values that were mislabelled.**
  Coordinates are unaffected; only the names change.

  `rightEyeTop` and `rightEyeBottom` were swapped. Index 36 sits above index 38
  in only 2.9% of the 2079 CatFLW images, and `catLandmarkFlipIndex` pairs 36
  with `leftEyeBottom` and 38 with `leftEyeTop`. A horizontal flip preserves
  vertical position, so both the data and the flip table agree the labels were
  inverted. Index 36 is now `rightEyeBottom` and index 38 is `rightEyeTop`.

  The chin contour points were grouped wrongly. `catLandmarkFlipIndex` mirrors
  18 with 20 and 19 with 21, making those the left/right pairs, but the names
  implied the pairs were 18/19 and 20/21. Index 19 is now `chinLeft1` and index
  20 is `chinRight0`.

  `catLandmarkConnections` is updated so the right-eye ring spans the same
  landmark indices as before.

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
  mode with `PerformanceMode.auto`, poseOnly drops from 47.7 ms/frame to
  15.1 ms, a 3.2x speedup on the shared body pipeline. The full pipeline goes
  from 234.6 ms/frame to 113.7 ms, though that figure also includes the
  landmark model swap below rather than the tensor change alone.

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
