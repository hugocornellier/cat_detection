import 'dart:typed_data';

import 'package:opencv_dart/opencv_dart.dart' as cv;
import '../cat_detector.dart';
import '../types.dart';

/// Background-isolate wrapper for cat detection.
///
/// Deprecated: [CatDetector] now runs its pipeline in a background isolate that
/// it owns, so this wrapper is redundant. It remains as a thin delegate to
/// [CatDetector] and will be removed in the next major release.
///
/// Migration:
/// ```dart
/// // Before
/// final detector = await CatDetectorIsolate.spawn(mode: CatDetectionMode.full);
/// final cats = await detector.detectCats(bytes);
/// final more = await detector.detectCatsFromMat(mat);
///
/// // After
/// final detector = CatDetector(mode: CatDetectionMode.full);
/// await detector.initialize();
/// final cats = await detector.detect(bytes);
/// final more = await detector.detectFromMat(mat);
/// ```
@Deprecated(
  'Use CatDetector instead, which now runs detection in a background isolate '
  'automatically. Will be removed in the next major release.',
)
class CatDetectorIsolate {
  CatDetectorIsolate._(this._delegate);

  final CatDetector _delegate;

  /// Returns true if the isolate is initialized and ready for detection.
  bool get isReady => _delegate.isReady;

  /// Spawns a new isolate with an initialized cat detection pipeline.
  ///
  /// Deprecated: construct a [CatDetector] and call `initialize()` instead.
  @Deprecated(
    'Use CatDetector(...)..initialize() instead. '
    'Will be removed in the next major release.',
  )
  static Future<CatDetectorIsolate> spawn({
    CatDetectionMode mode = CatDetectionMode.full,
    AnimalPoseModel poseModel = AnimalPoseModel.rtmpose,
    CatLandmarkModel landmarkModel = CatLandmarkModel.full,
    double cropMargin = 0.20,
    int interpreterPoolSize = 1,
    PerformanceConfig performanceConfig = const PerformanceConfig(),
    void Function(String model, int received, int total)? onDownloadProgress,
  }) async {
    final detector = CatDetector(
      mode: mode,
      poseModel: poseModel,
      landmarkModel: landmarkModel,
      cropMargin: cropMargin,
      interpreterPoolSize: interpreterPoolSize,
      performanceConfig: performanceConfig,
    );
    await detector.initialize(onDownloadProgress: onDownloadProgress);
    return CatDetectorIsolate._(detector);
  }

  /// Detects cats in an encoded image in the background isolate.
  ///
  /// Deprecated: use [CatDetector.detect] instead.
  @Deprecated(
    'Use CatDetector.detect instead. Will be removed in the next major release.',
  )
  Future<List<Cat>> detectCats(Uint8List bytes) => _delegate.detect(bytes);

  /// Detects cats in a pre-decoded [cv.Mat] in the background isolate.
  ///
  /// The supplied Mat is NOT disposed by this method.
  ///
  /// Deprecated: use [CatDetector.detectFromMat] instead.
  @Deprecated(
    'Use CatDetector.detectFromMat instead. '
    'Will be removed in the next major release.',
  )
  Future<List<Cat>> detectCatsFromMat(cv.Mat image) =>
      _delegate.detectFromMat(image);

  /// Disposes the background isolate and releases all resources.
  ///
  /// Deprecated: use [CatDetector.dispose] instead.
  @Deprecated(
    'Use CatDetector.dispose instead. '
    'Will be removed in the next major release.',
  )
  Future<void> dispose() => _delegate.dispose();
}
