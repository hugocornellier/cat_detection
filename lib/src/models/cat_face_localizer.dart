import 'dart:async';
import 'dart:typed_data';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:flutter_litert/flutter_litert.dart';
import '../util/image_utils.dart';

/// EfficientNetB2 regression model for cat face localization.
///
/// Input 224x224, output [1,4] single bounding box in XYXY format,
/// normalized to [0,1]. No anchors, no NMS, single bbox per image.
///
/// Trained on CatFLW dataset with CIoU + L1 loss, two-phase training
/// (frozen backbone + fine-tune). Bounding boxes derived from 48 landmark
/// annotations with 0.12 margin.
class CatFaceLocalizer {
  static const int inputSize = 224;

  static const String _modelPath =
      'packages/cat_detection/assets/models/cat_face_localizer.tflite';

  Interpreter? _interpreter;
  IsolateInterpreter? _isolateInterpreter;
  Delegate? _delegate;

  late List<List<List<List<double>>>> _inputTensor;
  late List<List<double>> _outputBuffer;
  Float32List? _rgbBuffer;

  Future<void> initialize(PerformanceConfig performanceConfig) async {
    final (options, delegate) = InterpreterFactory.create(performanceConfig);
    _delegate = delegate;
    _interpreter = await Interpreter.fromAsset(_modelPath, options: options);
    _isolateInterpreter = await InterpreterFactory.createIsolateIfNeeded(
      _interpreter!,
      _delegate,
    );
    _inputTensor = createNHWCTensor4D(inputSize, inputSize);
    _outputBuffer = List.generate(1, (_) => List.filled(4, 0.0));
  }

  Future<void> initializeFromBuffer(
    Uint8List bytes,
    PerformanceConfig performanceConfig,
  ) async {
    final (options, delegate) = InterpreterFactory.create(performanceConfig);
    _delegate = delegate;
    _interpreter = Interpreter.fromBuffer(bytes, options: options);
    _isolateInterpreter = await InterpreterFactory.createIsolateIfNeeded(
      _interpreter!,
      _delegate,
    );
    _inputTensor = createNHWCTensor4D(inputSize, inputSize);
    _outputBuffer = List.generate(1, (_) => List.filled(4, 0.0));
  }

  Future<BoundingBox?> detect(cv.Mat image) async {
    final (padded, params) = ImageUtils.letterboxResize(image, inputSize);

    _rgbBuffer = ImageUtils.matToFloat32(padded);
    fillNHWC4D(_rgbBuffer!, _inputTensor, inputSize, inputSize);
    padded.dispose();

    if (_isolateInterpreter != null) {
      await _isolateInterpreter!.run(_inputTensor, _outputBuffer);
    } else {
      _interpreter!.run(_inputTensor, _outputBuffer);
    }

    final raw = _outputBuffer[0];
    final xa = raw[0].clamp(0.0, 1.0) * inputSize;
    final ya = raw[1].clamp(0.0, 1.0) * inputSize;
    final xb = raw[2].clamp(0.0, 1.0) * inputSize;
    final yb = raw[3].clamp(0.0, 1.0) * inputSize;
    final x1 = xa < xb ? xa : xb;
    final x2 = xa < xb ? xb : xa;
    final y1 = ya < yb ? ya : yb;
    final y2 = ya < yb ? yb : ya;

    final ratio = params.scale;
    final dw = params.padLeft;
    final dh = params.padTop;
    final origX1 = ((x1 - dw) / ratio).clamp(0.0, image.cols.toDouble());
    final origY1 = ((y1 - dh) / ratio).clamp(0.0, image.rows.toDouble());
    final origX2 = ((x2 - dw) / ratio).clamp(0.0, image.cols.toDouble());
    final origY2 = ((y2 - dh) / ratio).clamp(0.0, image.rows.toDouble());

    if (origX2 - origX1 < 1.0 || origY2 - origY1 < 1.0) return null;

    return BoundingBox.ltrb(origX1, origY1, origX2, origY2);
  }

  void dispose() {
    _isolateInterpreter?.close();
    _interpreter?.close();
    _delegate?.delete();
  }
}
