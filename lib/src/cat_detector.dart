import 'dart:isolate';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_litert/native.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:animal_detection/animal_detection.dart';

import 'types.dart';
import 'isolate/cat_detector_core.dart';

/// Startup payload for the detection isolate.
///
/// Model bytes travel as [TransferableTypedData] so ownership moves to the
/// worker isolate without copying (~70MB of weights in the default config).
class _IsolateStartupData {
  final SendPort sendPort;

  // Face pipeline
  final TransferableTypedData? localizerBytes;
  final TransferableTypedData? landmarkBytes;

  // Body pipeline
  final TransferableTypedData? bodyDetectorBytes;
  final TransferableTypedData? classifierBytes;
  final String? speciesMappingJson;
  final TransferableTypedData? poseModelBytes;
  final String poseModelName;

  // Configuration
  final String modeName;
  final String landmarkModelName;
  final double cropMargin;
  final double detThreshold;
  final int interpreterPoolSize;
  final String performanceModeName;
  final int? numThreads;
  // Null means the landmark stage follows performanceModeName. PerformanceConfig
  // is not sendable across an isolate boundary, so it travels as a mode name plus
  // a thread count, the same way the pipeline-wide config does.
  final String? landmarkPerformanceModeName;
  final int? landmarkNumThreads;

  const _IsolateStartupData({
    required this.sendPort,
    this.localizerBytes,
    this.landmarkBytes,
    this.bodyDetectorBytes,
    this.classifierBytes,
    this.speciesMappingJson,
    this.poseModelBytes,
    required this.poseModelName,
    required this.modeName,
    required this.landmarkModelName,
    required this.cropMargin,
    required this.detThreshold,
    required this.interpreterPoolSize,
    required this.performanceModeName,
    this.numThreads,
    this.landmarkPerformanceModeName,
    this.landmarkNumThreads,
  });
}

/// On-device cat detection using a unified multi-stage TensorFlow Lite pipeline.
///
/// Inference runs in a background isolate that this class owns, so the pipeline
/// never blocks the UI thread. Model assets are loaded (and downloaded, when
/// needed) on the main isolate and transferred into the worker during
/// [initialize].
///
/// Supports three modes:
/// - [CatDetectionMode.full]: SSD body detection + species classification +
///   body pose estimation + face landmarks.
/// - [CatDetectionMode.poseOnly]: Body detection + species + body pose only.
/// - [CatDetectionMode.faceOnly]: Face localizer + face landmarks, no SSD stage.
///
/// Usage:
/// ```dart
/// final detector = CatDetector(mode: CatDetectionMode.full);
/// await detector.initialize();
/// final cats = await detector.detect(imageBytes);
/// await detector.dispose();
/// ```
class CatDetector {
  static const String _packageVersion = '2.0.0';
  static const String _pipelineVersion = 'pipeline_v3';

  /// Version key for the default cat detection pipeline.
  ///
  /// Downstream caches can use this to invalidate stored detections when model
  /// weights, preprocessing, post-processing, thresholds, or coordinate
  /// conventions change.
  static const String modelVersion =
      'cat_detection:$_packageVersion:mode=full:poseModel=rtmpose:'
      'landmarkModel=full:$_pipelineVersion';

  /// Builds a version key for a specific cat detector configuration.
  static String modelVersionFor({
    CatDetectionMode mode = CatDetectionMode.full,
    AnimalPoseModel poseModel = AnimalPoseModel.rtmpose,
    CatLandmarkModel landmarkModel = CatLandmarkModel.full,
  }) {
    return 'cat_detection:$_packageVersion:mode=${mode.name}:'
        'poseModel=${poseModel.name}:landmarkModel=${landmarkModel.name}:'
        '$_pipelineVersion';
  }

  /// Detection mode controlling pipeline behavior.
  final CatDetectionMode mode;

  /// Body pose model variant.
  final AnimalPoseModel poseModel;

  /// Cat face landmark model variant.
  final CatLandmarkModel landmarkModel;

  /// Margin fraction added to each side of the body bounding box before cropping.
  final double cropMargin;

  /// SSD detection score threshold.
  final double detThreshold;

  /// Number of TensorFlow Lite interpreter instances in the landmark model pool.
  final int interpreterPoolSize;

  /// Performance configuration for TensorFlow Lite inference.
  ///
  /// By default, auto mode selects the optimal delegate per platform:
  /// - iOS: Metal GPU delegate
  /// - Android/macOS/Linux/Windows: XNNPACK (2-5x SIMD acceleration)
  final PerformanceConfig performanceConfig;

  /// Optional override of [performanceConfig] for the face landmark stage alone.
  ///
  /// Null means the landmark stage uses [performanceConfig] like every other
  /// stage, which is the previous and still the default behaviour.
  ///
  /// This exists because the best delegate genuinely differs per stage, and a
  /// single pipeline-wide setting cannot express that. Measured on macOS arm64
  /// (M4 Max) with flutter_litert 3.7.0, XNNPACK versus the Metal GPU delegate:
  ///
  /// | stage | XNNPACK | Metal |
  /// |---|---|---|
  /// | ssdlite | 4.42 ms | 5.87 ms |
  /// | species classifier | 1.25 ms | interpreter creation fails |
  /// | rtmpose_s | 7.82 ms | 10.85 ms, and output deviates by 2.6e-01 |
  /// | face localizer | 8.65 ms | interpreter creation fails |
  /// | cat face landmarks | 26.40 ms | 5.02 ms |
  ///
  /// So GPU is a large win for the landmark stage and a loss or a hard failure
  /// everywhere else. Setting [performanceConfig] to [PerformanceMode.gpu]
  /// pipeline-wide would throw during [initialize].
  ///
  /// The 5.02 ms figure requires a landmark asset whose `TRANSPOSE_CONV` ops
  /// carry no fused activation, because the GPU delegate supports that op only
  /// to version 3 and a fused activation forces version 4. The asset bundled
  /// today does carry one, so it fails GPU interpreter creation and this
  /// override will not help it yet. It is wired up now so the routing exists
  /// once a compatible asset ships.
  ///
  /// Only macOS has been measured. iOS resolves Metal from a different binary
  /// and Android uses an entirely different GPU delegate, so verify on device
  /// before setting this in production.
  final PerformanceConfig? landmarkPerformanceConfig;

  _CatDetectorWorker? _worker;

  /// Creates a cat detector with the specified configuration.
  CatDetector({
    this.mode = CatDetectionMode.full,
    this.poseModel = AnimalPoseModel.rtmpose,
    this.landmarkModel = CatLandmarkModel.full,
    this.cropMargin = 0.20,
    this.detThreshold = 0.5,
    int interpreterPoolSize = 1,
    this.performanceConfig = const PerformanceConfig(),
    this.landmarkPerformanceConfig,
  }) : interpreterPoolSize =
            (landmarkPerformanceConfig ?? performanceConfig).mode ==
                    PerformanceMode.disabled
                ? interpreterPoolSize
                : 1;

  /// Returns true if the detector has been initialized and is ready to use.
  bool get isReady => _worker?.isReady ?? false;

  /// Returns true if the detector has been initialized and is ready to use.
  bool get isInitialized => isReady;

  /// Returns true if the HRNet model is already cached locally.
  static Future<bool> isHrnetCached() => ModelDownloader.isHrnetCached();

  /// Loads TensorFlow Lite models and spawns the background detection isolate.
  ///
  /// Must be called before [detect] or [detectFromMat]. Model assets are read on
  /// the main isolate (where `rootBundle` is available) and transferred into the
  /// worker, so this takes 100-500ms depending on the device.
  ///
  /// When [poseModel] is [AnimalPoseModel.hrnet], the HRNet model (~54.6 MB) is
  /// downloaded from GitHub Releases on first use and cached locally.
  ///
  /// [onDownloadProgress] is called during any model download with
  /// (modelName, bytesReceived, totalBytes).
  Future<void> initialize({
    void Function(String model, int received, int total)? onDownloadProgress,
  }) async {
    if (_worker != null) {
      await dispose();
    }

    final bool needsFace =
        mode == CatDetectionMode.full || mode == CatDetectionMode.faceOnly;
    final bool needsBody =
        mode == CatDetectionMode.full || mode == CatDetectionMode.poseOnly;

    // Face pipeline assets
    TransferableTypedData? localizerTtd;
    TransferableTypedData? landmarkTtd;

    if (needsFace) {
      const localizerPath =
          'packages/cat_detection/assets/models/cat_face_localizer.tflite';
      const landmarkPath =
          'packages/cat_detection/assets/models/cat_face_landmarks_full.tflite';

      final results = await Future.wait([
        rootBundle.load(localizerPath),
        rootBundle.load(landmarkPath),
      ]);

      localizerTtd = TransferableTypedData.fromList(
        [results[0].buffer.asUint8List()],
      );
      landmarkTtd = TransferableTypedData.fromList(
        [results[1].buffer.asUint8List()],
      );
    }

    // Body pipeline assets
    TransferableTypedData? bodyDetectorTtd;
    TransferableTypedData? classifierTtd;
    String? speciesMappingJson;
    TransferableTypedData? poseModelTtd;

    if (needsBody) {
      const bodyDetectorPath =
          'packages/animal_detection/assets/models/superanimal_ssdlite_float16.tflite';
      const classifierPath =
          'packages/animal_detection/assets/models/species_classifier_float16.tflite';
      const speciesMappingPath =
          'packages/animal_detection/assets/models/species_mapping.json';

      final bodyResults = await Future.wait([
        rootBundle.load(bodyDetectorPath),
        rootBundle.load(classifierPath),
        rootBundle.loadString(speciesMappingPath),
      ]);

      bodyDetectorTtd = TransferableTypedData.fromList(
        [(bodyResults[0] as ByteData).buffer.asUint8List()],
      );
      classifierTtd = TransferableTypedData.fromList(
        [(bodyResults[1] as ByteData).buffer.asUint8List()],
      );
      speciesMappingJson = bodyResults[2] as String;

      if (poseModel == AnimalPoseModel.hrnet) {
        final hrnetBytes = await ModelDownloader.getHrnetModel(
          onProgress: onDownloadProgress == null
              ? null
              : (received, total) => onDownloadProgress(
                  ModelDownloader.modelHrnet, received, total),
        );
        poseModelTtd = TransferableTypedData.fromList([hrnetBytes]);
      } else {
        const rtmposePath =
            'packages/animal_detection/assets/models/superanimal_rtmpose_s_float16.tflite';
        final rtmposeData = await rootBundle.load(rtmposePath);
        poseModelTtd = TransferableTypedData.fromList(
          [rtmposeData.buffer.asUint8List()],
        );
      }
    }

    final worker = _CatDetectorWorker();
    try {
      await worker.initialize(
        startupData: (SendPort sendPort) => _IsolateStartupData(
          sendPort: sendPort,
          localizerBytes: localizerTtd,
          landmarkBytes: landmarkTtd,
          bodyDetectorBytes: bodyDetectorTtd,
          classifierBytes: classifierTtd,
          speciesMappingJson: speciesMappingJson,
          poseModelBytes: poseModelTtd,
          poseModelName: poseModel.name,
          modeName: mode.name,
          landmarkModelName: landmarkModel.name,
          cropMargin: cropMargin,
          detThreshold: detThreshold,
          interpreterPoolSize: interpreterPoolSize,
          performanceModeName: performanceConfig.mode.name,
          numThreads: performanceConfig.numThreads,
          landmarkPerformanceModeName: landmarkPerformanceConfig?.mode.name,
          landmarkNumThreads: landmarkPerformanceConfig?.numThreads,
        ),
      );
    } catch (_) {
      await worker.dispose();
      rethrow;
    }
    _worker = worker;
  }

  /// Detects cats in an encoded image (JPEG, PNG, etc.).
  ///
  /// Decoding and inference both happen in the background isolate.
  ///
  /// Throws [StateError] if called before [initialize].
  Future<List<Cat>> detect(Uint8List imageBytes) async {
    final worker = _requireWorker();
    final result = await worker.sendRequest<List<dynamic>>(
      'detect',
      {
        'bytes': TransferableTypedData.fromList([imageBytes])
      },
    );
    return _deserializeCats(result);
  }

  /// Detects cats in a pre-decoded [cv.Mat].
  ///
  /// The raw pixel data is transferred to the isolate using zero-copy
  /// [TransferableTypedData]. The supplied Mat is NOT disposed by this method.
  ///
  /// [imageWidth] and [imageHeight] default to the Mat's own dimensions and only
  /// need to be passed when they differ.
  ///
  /// Throws [StateError] if called before [initialize].
  Future<List<Cat>> detectFromMat(
    cv.Mat image, {
    int? imageWidth,
    int? imageHeight,
  }) async {
    final worker = _requireWorker();
    final result = await worker.sendRequest<List<dynamic>>(
      'detectMat',
      {
        'bytes': TransferableTypedData.fromList([image.data]),
        'width': imageWidth ?? image.cols,
        'height': imageHeight ?? image.rows,
        'matType': image.type.value,
      },
    );
    return _deserializeCats(result);
  }

  /// Releases the background isolate and all native model resources.
  ///
  /// After disposing, call [initialize] again before detecting.
  Future<void> dispose() async {
    final worker = _worker;
    _worker = null;
    if (worker == null) return;

    // Graceful shutdown: sends 'dispose' as an RPC and awaits the ack before
    // force-killing the isolate, so it can free its native TFLite interpreters
    // instead of being reaped mid-cleanup by Isolate.kill(priority: immediate).
    await worker.disposeGracefully();
  }

  _CatDetectorWorker _requireWorker() {
    final worker = _worker;
    if (worker == null || !worker.isReady) {
      throw StateError('CatDetector not initialized. Call initialize() first.');
    }
    return worker;
  }

  List<Cat> _deserializeCats(List<dynamic> result) => result
      .map((map) => Cat.fromMap(Map<String, dynamic>.from(map as Map)))
      .toList();

  /// Reconstructs a [cv.Mat] from raw bytes WITHOUT the boxed, double-copy
  /// `cv.Mat.fromList` path (it takes a `List<num>`, boxing every byte and
  /// copying twice, ~100x slower for full frames). Allocates once with
  /// `Mat.create` and bulk-copies into the Mat's contiguous native data view.
  static cv.Mat _matFromBytes(
    int rows,
    int cols,
    cv.MatType type,
    Uint8List bytes,
  ) {
    final mat = cv.Mat.create(rows: rows, cols: cols, type: type);
    mat.data.setRange(0, bytes.length, bytes);
    return mat;
  }

  /// Isolate entry point: builds a [CatDetectorCore] from the transferred model
  /// bytes, then serves detection requests.
  ///
  /// Sends its [SendPort] back to the main isolate on success, or an error map
  /// on failure.
  @pragma('vm:entry-point')
  static void _isolateEntry(_IsolateStartupData data) async {
    final SendPort mainSendPort = data.sendPort;
    final ReceivePort workerReceivePort = ReceivePort();

    CatDetectorCore? detector;

    try {
      final localizerBytes = data.localizerBytes?.materialize().asUint8List();
      final landmarkBytes = data.landmarkBytes?.materialize().asUint8List();

      final bodyDetectorBytes =
          data.bodyDetectorBytes?.materialize().asUint8List();
      final classifierBytes = data.classifierBytes?.materialize().asUint8List();
      final poseModelBytes = data.poseModelBytes?.materialize().asUint8List();

      final mode = CatDetectionMode.values.firstWhere(
        (m) => m.name == data.modeName,
      );
      final poseModel = AnimalPoseModel.values.firstWhere(
        (m) => m.name == data.poseModelName,
      );
      final landmarkModel = CatLandmarkModel.values.firstWhere(
        (m) => m.name == data.landmarkModelName,
      );
      final performanceMode = PerformanceMode.values.firstWhere(
        (m) => m.name == data.performanceModeName,
      );
      final landmarkMode = data.landmarkPerformanceModeName == null
          ? null
          : PerformanceMode.values.firstWhere(
              (m) => m.name == data.landmarkPerformanceModeName,
            );

      detector = CatDetectorCore(
        mode: mode,
        poseModel: poseModel,
        landmarkModel: landmarkModel,
        cropMargin: data.cropMargin,
        detThreshold: data.detThreshold,
        interpreterPoolSize: data.interpreterPoolSize,
        performanceConfig: PerformanceConfig(
          mode: performanceMode,
          numThreads: data.numThreads,
        ),
        landmarkPerformanceConfig: landmarkMode == null
            ? null
            : PerformanceConfig(
                mode: landmarkMode,
                numThreads: data.landmarkNumThreads,
              ),
      );

      await detector.initializeFromBuffers(
        localizerBytes: localizerBytes,
        landmarkBytes: landmarkBytes,
        bodyDetectorBytes: bodyDetectorBytes,
        classifierBytes: classifierBytes,
        speciesMappingJson: data.speciesMappingJson,
        poseModelBytes: poseModelBytes,
        useIsolateInterpreter: false,
      );

      mainSendPort.send(workerReceivePort.sendPort);
    } catch (e, st) {
      mainSendPort.send({
        'error': 'Cat detection isolate initialization failed: $e\n$st',
      });
      return;
    }

    workerReceivePort.listen((message) async {
      if (message is! Map) return;

      final int? id = message['id'] as int?;
      final String? op = message['op'] as String?;

      if (id == null || op == null) return;

      try {
        switch (op) {
          case 'detect':
            if (detector == null || !detector!.isInitialized) {
              mainSendPort.send({
                'id': id,
                'error': 'CatDetector not initialized in isolate',
              });
              return;
            }

            final ByteBuffer bb =
                (message['bytes'] as TransferableTypedData).materialize();
            final cats = await detector!.detect(bb.asUint8List());

            mainSendPort.send({
              'id': id,
              'result': cats.map((c) => c.toMap()).toList(),
            });

          case 'detectMat':
            if (detector == null || !detector!.isInitialized) {
              mainSendPort.send({
                'id': id,
                'error': 'CatDetector not initialized in isolate',
              });
              return;
            }

            final ByteBuffer bb =
                (message['bytes'] as TransferableTypedData).materialize();
            final int width = message['width'] as int;
            final int height = message['height'] as int;
            final matType = cv.MatType(message['matType'] as int);
            final mat = _matFromBytes(height, width, matType, bb.asUint8List());

            try {
              final cats = await detector!.detectFromMat(
                mat,
                imageWidth: width,
                imageHeight: height,
              );
              mainSendPort.send({
                'id': id,
                'result': cats.map((c) => c.toMap()).toList(),
              });
            } finally {
              mat.dispose();
            }

          case 'dispose':
            await detector?.dispose();
            detector = null;
            mainSendPort.send({'id': id, 'result': true});
            workerReceivePort.close();
        }
      } catch (e, st) {
        mainSendPort.send({'id': id, 'error': '$e\n$st'});
      }
    });
  }
}

/// Owns the detection isolate and its RPC channel.
class _CatDetectorWorker extends IsolateWorkerBase {
  @override
  String get workerDisposeOp => 'dispose';

  Future<void> initialize({
    required _IsolateStartupData Function(SendPort sendPort) startupData,
  }) async {
    await initWorker(
      (sendPort) => Isolate.spawn(
        CatDetector._isolateEntry,
        startupData(sendPort),
        debugName: 'CatDetector',
      ),
      timeout: const Duration(seconds: 60),
      timeoutMessage: 'Cat detection isolate initialization timed out',
    );
  }
}
