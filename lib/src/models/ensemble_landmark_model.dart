import 'dart:async';
import 'package:flutter/services.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:flutter_litert/flutter_litert.dart';
import '../types.dart';
import '../util/image_utils.dart';
import '../util/model_downloader.dart';

/// 3-model ensemble landmark runner with multi-scale + flip TTA.
///
/// Mirrors the Python evaluation pipeline:
/// - 3 models (256px + 320px + 384px)
/// - 3 scales (0.9, 1.0, 1.1) via center-crop / reflect-pad
/// - 2 orientations (original + horizontal flip with landmark remapping)
/// = 18 inference passes per image, averaged in normalized [0,1] space.
class EnsembleLandmarkModel {
  static const int _size256 = 256;
  static const int _size320 = 320;
  static const int _size384 = 384;

  static const List<double> _scales = [0.9, 1.0, 1.1];

  static const String _bundledModelPath =
      'packages/cat_detection/assets/models/cat_face_landmarks_full.tflite';

  final int _poolSize;

  InterpreterPool? _pool256;
  InterpreterPool? _pool320;
  InterpreterPool? _pool384;

  /// Creates an ensemble model runner with the specified interpreter pool size.
  EnsembleLandmarkModel({int poolSize = 1}) : _poolSize = poolSize;

  /// Initializes all 3 models.
  ///
  /// The 256px model is loaded from bundled assets.
  /// The 320px and 384px models are downloaded from GitHub Releases if not cached.
  ///
  /// [onDownloadProgress] is called during download with (modelName, received, total).
  Future<void> initialize(
    PerformanceConfig performanceConfig, {
    void Function(String model, int received, int total)? onDownloadProgress,
  }) async {
    // Download extra models (or load from cache)
    final (bytes256, bytes320) = await CatModelDownloader.getEnsembleModels(
      onProgress: onDownloadProgress,
    );

    // Load bundled 256 model
    final byteData384 = await rootBundle.load(_bundledModelPath);
    final bytes384 = byteData384.buffer.asUint8List();

    await _initializeFromBytes(
      bytes256: bytes256,
      bytes320: bytes320,
      bytes384: bytes384,
      performanceConfig: performanceConfig,
    );
  }

  /// Initializes from pre-loaded model bytes (for use in isolates).
  Future<void> initializeFromBuffers({
    required Uint8List bytes256,
    required Uint8List bytes320,
    required Uint8List bytes384,
    required PerformanceConfig performanceConfig,
  }) async {
    await _initializeFromBytes(
      bytes256: bytes256,
      bytes320: bytes320,
      bytes384: bytes384,
      performanceConfig: performanceConfig,
    );
  }

  Future<void> _initializeFromBytes({
    required Uint8List bytes256,
    required Uint8List bytes320,
    required Uint8List bytes384,
    required PerformanceConfig performanceConfig,
  }) async {
    _pool256 = InterpreterPool(poolSize: _poolSize);
    _pool320 = InterpreterPool(poolSize: _poolSize);
    _pool384 = InterpreterPool(poolSize: _poolSize);

    await Future.wait([
      _initPool(_pool256!, bytes256, _size256, performanceConfig),
      _initPool(_pool320!, bytes320, _size320, performanceConfig),
      _initPool(_pool384!, bytes384, _size384, performanceConfig),
    ]);
  }

  Future<void> _initPool(
    InterpreterPool pool,
    Uint8List bytes,
    int inputSize,
    PerformanceConfig config,
  ) async {
    await pool.initialize(
      (options, _) async {
        final interpreter = Interpreter.fromBuffer(bytes, options: options);
        interpreter.resizeInputTensor(0, [1, inputSize, inputSize, 3]);
        interpreter.allocateTensors();
        return interpreter;
      },
      performanceConfig: config,
    );
  }

  /// The input size used for cropping (largest model size).
  int get inputSize => _size384;

  /// Runs all 3 models with multi-scale + flip TTA (18 passes) and averages.
  Future<List<CatLandmark>> predict(
    cv.Mat crop384,
    CropMetadata meta,
    int imageWidth,
    int imageHeight,
  ) async {
    // Resize crop to each model's input size
    final crop256 = cv.resize(crop384, (_size256, _size256));
    final crop320 = cv.resize(crop384, (_size320, _size320));

    // Pre-create all scaled/flipped Mats for all 18 passes
    final tempMats = <cv.Mat>[];
    final futures = <Future<List<double>>>[];

    for (final (pool, crop, size) in [
      (_pool256!, crop256, _size256),
      (_pool320!, crop320, _size320),
      (_pool384!, crop384, _size384),
    ]) {
      for (final scale in _scales) {
        // Scale the crop
        final scaled = _scaleCrop(crop, scale, size);
        if ((scale - 1.0).abs() >= 1e-6) tempMats.add(scaled);

        // Normal prediction with scale unscaling
        futures.add(
          _runModelRaw(pool, scaled, size).then(
            (raw) => _unscaleCoords(raw, scale),
          ),
        );

        // Flipped prediction with landmark remapping + scale unscaling
        final flipped = cv.flip(scaled, 1);
        tempMats.add(flipped);

        futures.add(
          _runModelRaw(pool, flipped, size).then((raw) {
            final remapped = List<double>.filled(96, 0.0);
            for (int i = 0; i < numCatLandmarks; i++) {
              final srcIdx = catLandmarkFlipIndex[i];
              remapped[i * 2] = 1.0 - raw[srcIdx * 2]; // flip x
              remapped[i * 2 + 1] = raw[srcIdx * 2 + 1]; // keep y
            }
            return _unscaleCoords(remapped, scale);
          }),
        );
      }
    }

    try {
      final allPreds = await Future.wait(futures);

      // Average all 18 predictions
      final landmarks = <CatLandmark>[];
      final types = CatLandmarkType.values;

      for (int i = 0; i < numCatLandmarks; i++) {
        double xSum = 0.0;
        double ySum = 0.0;
        for (final pred in allPreds) {
          xSum += pred[i * 2];
          ySum += pred[i * 2 + 1];
        }
        final xNorm = (xSum / allPreds.length).clamp(0.0, 1.0);
        final yNorm = (ySum / allPreds.length).clamp(0.0, 1.0);

        final x = xNorm * meta.cropW + meta.cx1;
        final y = yNorm * meta.cropH + meta.cy1;

        landmarks.add(CatLandmark(type: types[i], x: x, y: y));
      }

      return landmarks;
    } finally {
      for (final mat in tempMats) {
        mat.dispose();
      }
      crop256.dispose();
      crop320.dispose();
    }
  }

  /// Scale crop to simulate zoom via center-crop (scale<1) or reflect-pad (scale>1).
  ///
  /// Matches Python's `scale_crop()`:
  /// - scale < 1.0: center-crop to (size*scale), then resize back to size
  /// - scale > 1.0: reflect-pad by (size*(scale-1)/2), then resize back to size
  /// - scale == 1.0: return original (no copy)
  cv.Mat _scaleCrop(cv.Mat crop, double scale, int inputSize) {
    if ((scale - 1.0).abs() < 1e-6) return crop;

    if (scale < 1.0) {
      final newH = (inputSize * scale).round();
      final newW = (inputSize * scale).round();
      final offsetY = (inputSize - newH) ~/ 2;
      final offsetX = (inputSize - newW) ~/ 2;
      final cropped = crop.region(cv.Rect(offsetX, offsetY, newW, newH));
      final resized = cv.resize(cropped, (inputSize, inputSize));
      cropped.dispose();
      return resized;
    } else {
      final padH = (inputSize * (scale - 1.0) / 2.0).round();
      final padW = (inputSize * (scale - 1.0) / 2.0).round();
      final padded = cv.copyMakeBorder(
        crop,
        padH,
        padH,
        padW,
        padW,
        cv.BORDER_REFLECT_101,
      );
      final resized = cv.resize(padded, (inputSize, inputSize));
      padded.dispose();
      return resized;
    }
  }

  /// Unscale coords from scaled crop space back to original crop space.
  ///
  /// Matches Python's `unscale_coords()`: `(coords - 0.5) * scale + 0.5`
  List<double> _unscaleCoords(List<double> coords, double scale) {
    if ((scale - 1.0).abs() < 1e-6) return coords;
    return List<double>.generate(
      coords.length,
      (i) => (coords[i] - 0.5) * scale + 0.5,
    );
  }

  /// Run model and get raw normalized [0,1] output (96 values).
  Future<List<double>> _runModelRaw(
    InterpreterPool pool,
    cv.Mat crop,
    int inputSize,
  ) async {
    return pool.withInterpreter((interpreter, isolateInterpreter) async {
      final inputTensor = createNHWCTensor4D(inputSize, inputSize);
      final outputBuffer = allocTensorShape([1, 96]);

      final rgb = ImageUtils.matToFloat32(crop);
      fillNHWC4D(rgb, inputTensor, inputSize, inputSize);

      final outputs = {0: outputBuffer};
      if (isolateInterpreter != null) {
        await isolateInterpreter.runForMultipleInputs([inputTensor], outputs);
      } else {
        interpreter.runForMultipleInputs([inputTensor], outputs);
      }

      final raw = (outputBuffer as List)[0] as List;
      return List<double>.generate(
        96,
        (i) => (raw[i] as double).clamp(0.0, 1.0),
      );
    });
  }

  /// Disposes all interpreter pools.
  void dispose() {
    _pool256?.dispose();
    _pool320?.dispose();
    _pool384?.dispose();
  }
}
