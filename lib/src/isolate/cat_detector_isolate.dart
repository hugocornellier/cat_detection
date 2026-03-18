import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;

import 'package:animal_detection/animal_detection.dart'
    show AnimalPoseModel, ModelDownloader;
import '../cat_detector.dart';
import '../types.dart';
import '../util/model_downloader.dart';

/// Startup payload transferred to the background isolate via [Isolate.spawn].
///
/// Carries model bytes (as [TransferableTypedData] for zero-copy transfer)
/// and all configuration needed to reconstruct a [CatDetector] inside the isolate.
class _IsolateStartupData {
  final SendPort sendPort;

  // Face pipeline (full)
  final TransferableTypedData? localizerBytes;
  final TransferableTypedData? landmarkBytes;
  final TransferableTypedData? ensemble256Bytes;
  final TransferableTypedData? ensemble320Bytes;

  // Body pipeline (full / poseOnly)
  final TransferableTypedData? bodyDetectorBytes;
  final TransferableTypedData? classifierBytes;
  final String? speciesMappingJson;
  final TransferableTypedData? poseModelBytes;
  final String poseModelName;

  final String modeName;
  final String landmarkModelName;
  final double cropMargin;
  final int interpreterPoolSize;
  final String performanceModeName;
  final int? numThreads;

  _IsolateStartupData({
    required this.sendPort,
    this.localizerBytes,
    this.landmarkBytes,
    this.ensemble256Bytes,
    this.ensemble320Bytes,
    this.bodyDetectorBytes,
    this.classifierBytes,
    this.speciesMappingJson,
    this.poseModelBytes,
    required this.poseModelName,
    required this.modeName,
    required this.landmarkModelName,
    required this.cropMargin,
    required this.interpreterPoolSize,
    required this.performanceModeName,
    required this.numThreads,
  });
}

/// A wrapper that runs the entire cat detection pipeline in a background isolate.
///
/// This class spawns a dedicated isolate containing a full [CatDetector] instance,
/// keeping all TensorFlow Lite inference off the main UI thread. This prevents
/// frame drops during live camera processing.
///
/// Image data is transferred to the worker using zero-copy
/// [TransferableTypedData], minimizing memory overhead.
///
/// ## Usage
///
/// ```dart
/// final detector = await CatDetectorIsolate.spawn(
///   mode: CatDetectionMode.full,
///   performanceConfig: const PerformanceConfig.xnnpack(),
/// );
///
/// // From encoded image bytes (JPEG, PNG, etc.)
/// final cats = await detector.detectCats(imageBytes);
///
/// // From a cv.Mat (e.g., from camera frame conversion)
/// final cats = await detector.detectCatsFromMat(mat);
///
/// await detector.dispose();
/// ```
///
/// ## Memory Considerations
///
/// The background isolate holds all TFLite models in memory.
/// Call [dispose] when finished to release these resources.
class CatDetectorIsolate {
  CatDetectorIsolate._();

  Isolate? _isolate;
  SendPort? _sendPort;
  final ReceivePort _receivePort = ReceivePort();
  final Map<int, Completer<dynamic>> _pending = {};
  int _nextId = 0;

  bool _initialized = false;

  /// Returns true if the isolate is initialized and ready for detection.
  bool get isReady => _initialized;

  /// Spawns a new isolate with an initialized [CatDetector].
  ///
  /// The isolate loads all TFLite models during spawn, so this operation
  /// may take 100-500ms depending on the device.
  ///
  /// Parameters:
  /// - [mode]: Detection mode (poseOnly or full)
  /// - [poseModel]: Body pose model variant (rtmpose or hrnet)
  /// - [landmarkModel]: Cat landmark model variant
  /// - [cropMargin]: Margin fraction added to bbox sides before Stage 2 crop
  /// - [interpreterPoolSize]: Number of landmark model interpreter instances
  /// - [performanceConfig]: Hardware acceleration settings
  ///
  /// Example:
  /// ```dart
  /// final detector = await CatDetectorIsolate.spawn(
  ///   performanceConfig: const PerformanceConfig.xnnpack(),
  /// );
  /// ```
  static Future<CatDetectorIsolate> spawn({
    CatDetectionMode mode = CatDetectionMode.full,
    AnimalPoseModel poseModel = AnimalPoseModel.rtmpose,
    CatLandmarkModel landmarkModel = CatLandmarkModel.full,
    double cropMargin = 0.20,
    int interpreterPoolSize = 1,
    PerformanceConfig performanceConfig = PerformanceConfig.disabled,
    void Function(String model, int received, int total)? onDownloadProgress,
  }) async {
    final instance = CatDetectorIsolate._();
    await instance._initialize(
      mode: mode,
      poseModel: poseModel,
      landmarkModel: landmarkModel,
      cropMargin: cropMargin,
      interpreterPoolSize: interpreterPoolSize,
      performanceConfig: performanceConfig,
      onDownloadProgress: onDownloadProgress,
    );
    return instance;
  }

  /// Loads model assets and spawns the background isolate with an initialized [CatDetector].
  ///
  /// When [landmarkModel] is [CatLandmarkModel.ensemble], downloads the extra
  /// 256px and 320px models from GitHub Releases if not cached.
  ///
  /// When [poseModel] is [AnimalPoseModel.hrnet], downloads the HRNet model from
  /// GitHub Releases if not cached.
  Future<void> _initialize({
    required CatDetectionMode mode,
    required AnimalPoseModel poseModel,
    required CatLandmarkModel landmarkModel,
    required double cropMargin,
    required int interpreterPoolSize,
    required PerformanceConfig performanceConfig,
    void Function(String model, int received, int total)? onDownloadProgress,
  }) async {
    if (_initialized) {
      throw StateError('CatDetectorIsolate already initialized');
    }

    final bool needsFace = mode == CatDetectionMode.full;
    final bool needsBody =
        mode == CatDetectionMode.full || mode == CatDetectionMode.poseOnly;

    try {
      // Face pipeline assets
      TransferableTypedData? localizerTtd;
      TransferableTypedData? landmarkTtd;
      TransferableTypedData? ensemble256;
      TransferableTypedData? ensemble320;

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

        if (landmarkModel == CatLandmarkModel.ensemble) {
          final (bytes256, bytes320) =
              await CatModelDownloader.getEnsembleModels(
            onProgress: onDownloadProgress,
          );
          ensemble256 = TransferableTypedData.fromList([bytes256]);
          ensemble320 = TransferableTypedData.fromList([bytes320]);
        }
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
          final hrnetBytes = await ModelDownloader.getHrnetModel();
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

      _isolate = await Isolate.spawn(
        _isolateEntry,
        _IsolateStartupData(
          sendPort: _receivePort.sendPort,
          localizerBytes: localizerTtd,
          landmarkBytes: landmarkTtd,
          ensemble256Bytes: ensemble256,
          ensemble320Bytes: ensemble320,
          bodyDetectorBytes: bodyDetectorTtd,
          classifierBytes: classifierTtd,
          speciesMappingJson: speciesMappingJson,
          poseModelBytes: poseModelTtd,
          poseModelName: poseModel.name,
          modeName: mode.name,
          landmarkModelName: landmarkModel.name,
          cropMargin: cropMargin,
          interpreterPoolSize: interpreterPoolSize,
          performanceModeName: performanceConfig.mode.name,
          numThreads: performanceConfig.numThreads,
        ),
        debugName: 'CatDetectorIsolate',
      );

      _sendPort = await _setupIsolateListener(
        receivePort: _receivePort,
        responseHandler: _handleResponse,
        timeout: const Duration(seconds: 60),
        timeoutMsg: 'Cat detection isolate initialization timed out',
      );

      _initialized = true;
    } catch (e) {
      _isolate?.kill(priority: Isolate.immediate);
      _receivePort.close();
      _initialized = false;
      rethrow;
    }
  }

  /// Sets up init handshake and message routing for the isolate.
  static Future<SendPort> _setupIsolateListener({
    required ReceivePort receivePort,
    required void Function(dynamic) responseHandler,
    required Duration timeout,
    required String timeoutMsg,
  }) async {
    final Completer<SendPort> initCompleter = Completer<SendPort>();
    late final StreamSubscription<dynamic> subscription;

    subscription = receivePort.listen((message) {
      if (!initCompleter.isCompleted) {
        if (message is SendPort) {
          initCompleter.complete(message);
        } else if (message is Map && message['error'] != null) {
          initCompleter.completeError(StateError(message['error'] as String));
        } else {
          initCompleter.completeError(
            StateError('Expected SendPort, got ${message.runtimeType}'),
          );
        }
        return;
      }
      responseHandler(message);
    });

    return initCompleter.future.timeout(
      timeout,
      onTimeout: () {
        subscription.cancel();
        throw TimeoutException(timeoutMsg);
      },
    );
  }

  /// Routes a response message from the isolate to the correct pending [Completer].
  void _handleResponse(dynamic message) {
    if (message is! Map) return;

    final int? id = message['id'] as int?;
    if (id == null) return;

    final Completer<dynamic>? completer = _pending.remove(id);
    if (completer == null) return;

    if (message['error'] != null) {
      completer.completeError(StateError(message['error'] as String));
    } else {
      completer.complete(message['result']);
    }
  }

  /// Sends a request to the isolate and returns the typed response.
  ///
  /// Assigns a unique [id] to each request so [_handleResponse] can match
  /// responses to their corresponding [Completer].
  Future<T> _sendRequest<T>(
    String operation,
    Map<String, dynamic> params,
  ) async {
    if (!_initialized) {
      throw StateError(
        'CatDetectorIsolate not initialized. Use CatDetectorIsolate.spawn().',
      );
    }
    if (_sendPort == null) {
      throw StateError('Isolate SendPort not available.');
    }

    final int id = _nextId++;
    final Completer<T> completer = Completer<T>();
    _pending[id] = completer;

    try {
      _sendPort!.send({'id': id, 'op': operation, ...params});
      return await completer.future;
    } catch (e) {
      _pending.remove(id);
      rethrow;
    }
  }

  /// Detects cats in the given encoded image in the background isolate.
  ///
  /// All processing (image decoding, body detection, pose, face landmark
  /// extraction) runs in the background isolate.
  ///
  /// Parameters:
  /// - [bytes]: Encoded image data (JPEG, PNG, etc.)
  ///
  /// Returns a list of [Cat] objects.
  ///
  /// Example:
  /// ```dart
  /// final cats = await detector.detectCats(imageBytes);
  /// ```
  Future<List<Cat>> detectCats(Uint8List bytes) async {
    final List<dynamic> result = await _sendRequest<List<dynamic>>(
      'detect',
      {
        'bytes': TransferableTypedData.fromList([bytes]),
      },
    );

    return result
        .map((map) => Cat.fromMap(Map<String, dynamic>.from(map as Map)))
        .toList();
  }

  /// Detects cats in a pre-decoded [cv.Mat] image in the background isolate.
  ///
  /// The raw pixel data is extracted and transferred using zero-copy
  /// [TransferableTypedData]. The original Mat is NOT disposed by this method.
  ///
  /// Example:
  /// ```dart
  /// final mat = cv.Mat.fromList(height, width, cv.MatType.CV_8UC3, bgrBytes);
  /// final cats = await detector.detectCatsFromMat(mat);
  /// mat.dispose();
  /// ```
  Future<List<Cat>> detectCatsFromMat(cv.Mat image) async {
    final int rows = image.rows;
    final int cols = image.cols;
    final int type = image.type.value;
    final Uint8List data = image.data;

    final List<dynamic> result = await _sendRequest<List<dynamic>>(
      'detectMat',
      {
        'bytes': TransferableTypedData.fromList([data]),
        'width': cols,
        'height': rows,
        'matType': type,
      },
    );

    return result
        .map((map) => Cat.fromMap(Map<String, dynamic>.from(map as Map)))
        .toList();
  }

  /// Disposes the background isolate and releases all resources.
  ///
  /// After calling dispose, the instance cannot be reused. Create a new
  /// instance with [spawn] if needed.
  Future<void> dispose() async {
    for (final completer in _pending.values) {
      if (!completer.isCompleted) {
        completer.completeError(StateError('CatDetectorIsolate disposed'));
      }
    }
    _pending.clear();

    if (_sendPort != null) {
      try {
        _sendPort!.send({'id': -1, 'op': 'dispose'});
      } catch (_) {}
    }

    _isolate?.kill(priority: Isolate.immediate);
    _receivePort.close();

    _isolate = null;
    _sendPort = null;
    _initialized = false;
  }

  /// Isolate entry point: initializes the [CatDetector] and listens for detection requests.
  ///
  /// Sends its [SendPort] back to the main isolate on success, or an error map on failure.
  @pragma('vm:entry-point')
  static void _isolateEntry(_IsolateStartupData data) async {
    final SendPort mainSendPort = data.sendPort;
    final ReceivePort workerReceivePort = ReceivePort();

    CatDetector? detector;

    try {
      final localizerBytes = data.localizerBytes?.materialize().asUint8List();
      final landmarkBytes = data.landmarkBytes?.materialize().asUint8List();
      final ensemble256Bytes =
          data.ensemble256Bytes?.materialize().asUint8List();
      final ensemble320Bytes =
          data.ensemble320Bytes?.materialize().asUint8List();

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

      detector = CatDetector(
        mode: mode,
        poseModel: poseModel,
        landmarkModel: landmarkModel,
        cropMargin: data.cropMargin,
        interpreterPoolSize: data.interpreterPoolSize,
        performanceConfig: PerformanceConfig(
          mode: performanceMode,
          numThreads: data.numThreads,
        ),
      );

      await detector.initializeFromBuffers(
        localizerBytes: localizerBytes,
        landmarkBytes: landmarkBytes,
        ensemble256Bytes: ensemble256Bytes,
        ensemble320Bytes: ensemble320Bytes,
        bodyDetectorBytes: bodyDetectorBytes,
        classifierBytes: classifierBytes,
        speciesMappingJson: data.speciesMappingJson,
        poseModelBytes: poseModelBytes,
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
            final Uint8List imageBytes = bb.asUint8List();

            final cats = await detector!.detect(imageBytes);
            final serialized = cats.map((c) => c.toMap()).toList();

            mainSendPort.send({'id': id, 'result': serialized});

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
            final Uint8List matBytes = bb.asUint8List();
            final int width = message['width'] as int;
            final int height = message['height'] as int;
            final int matTypeValue = message['matType'] as int;

            final matType = cv.MatType(matTypeValue);
            final mat = cv.Mat.fromList(height, width, matType, matBytes);

            try {
              final cats = await detector!.detectFromMat(
                mat,
                imageWidth: width,
                imageHeight: height,
              );
              final serialized = cats.map((c) => c.toMap()).toList();
              mainSendPort.send({'id': id, 'result': serialized});
            } finally {
              mat.dispose();
            }

          case 'dispose':
            await detector?.dispose();
            detector = null;
            workerReceivePort.close();
        }
      } catch (e, st) {
        mainSendPort.send({'id': id, 'error': '$e\n$st'});
      }
    });
  }
}
