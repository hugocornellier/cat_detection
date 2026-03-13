import 'dart:async';
import 'dart:typed_data';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:flutter_litert/flutter_litert.dart';
import '../types.dart';
import '../util/image_utils.dart';

/// EfficientNetV2S landmark regressor for cat face keypoints.
///
/// Input 256x256, output [1,96] flattened 48 landmarks as
/// [x0,y0,...,x47,y47] normalized to [0,1] relative to the crop.
class CatLandmarkModelRunner {
  /// Input spatial dimension for the landmark model (256x256).
  static const int inputSize = 256;

  final InterpreterPool _pool;

  final CatLandmarkModel model;

  CatLandmarkModelRunner({
    required this.model,
    int poolSize = 1,
  }) : _pool = InterpreterPool(poolSize: poolSize);

  String get _modelPath {
    return 'packages/cat_detection/assets/models/cat_face_landmarks_full.tflite';
  }

  Future<void> initialize(PerformanceConfig performanceConfig) async {
    final path = _modelPath;
    await _pool.initialize(
      (options, _) async {
        final interpreter = await Interpreter.fromAsset(path, options: options);
        interpreter.resizeInputTensor(0, [1, inputSize, inputSize, 3]);
        interpreter.allocateTensors();
        return interpreter;
      },
      performanceConfig: performanceConfig,
    );
  }

  Future<void> initializeFromBuffer(
    Uint8List bytes,
    PerformanceConfig performanceConfig,
  ) async {
    await _pool.initialize(
      (options, _) async {
        final interpreter = Interpreter.fromBuffer(bytes, options: options);
        interpreter.resizeInputTensor(0, [1, inputSize, inputSize, 3]);
        interpreter.allocateTensors();
        return interpreter;
      },
      performanceConfig: performanceConfig,
    );
  }

  Future<List<CatLandmark>> predict(
    cv.Mat crop,
    CropMetadata meta,
    int imageWidth,
    int imageHeight,
  ) async {
    return _pool.withInterpreter((interpreter, isolateInterpreter) async {
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
      final landmarks = <CatLandmark>[];
      final types = CatLandmarkType.values;

      for (int i = 0; i < numCatLandmarks; i++) {
        final xNorm = (raw[i * 2] as double).clamp(0.0, 1.0);
        final yNorm = (raw[i * 2 + 1] as double).clamp(0.0, 1.0);

        final x = xNorm * meta.cropW + meta.cx1;
        final y = yNorm * meta.cropH + meta.cy1;

        landmarks.add(CatLandmark(type: types[i], x: x, y: y));
      }

      return landmarks;
    });
  }

  void dispose() {
    _pool.dispose();
  }
}
