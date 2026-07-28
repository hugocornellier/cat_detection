import 'dart:typed_data';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:animal_detection/animal_detection.dart';
import '../types.dart';

/// In-isolate implementation of the cat detection pipeline.
///
/// Supports two modes:
/// - [CatDetectionMode.full]: SSD body detection + species classification +
///   body pose estimation + face landmarks.
/// - [CatDetectionMode.poseOnly]: Body detection + species + body pose only.
/// - [CatDetectionMode.faceOnly]: Face localizer + face landmarks, no SSD.
///
/// Uses [AnimalDetector] from the animal_detection package for body detection,
/// species classification, and pose estimation. Cat-specific face landmark
/// extraction is handled directly.
///
/// This class runs inside the background isolate spawned by [CatDetector] and
/// is initialized from pre-loaded model bytes, since Flutter asset loading and
/// model downloads are only available on the main isolate. Consumers should use
/// [CatDetector], which owns an isolate running this core.
class CatDetectorCore {
  /// Input resolution of the bundled landmark model.
  ///
  /// Must match the bundled TFLite model's native input shape. The interpreter
  /// accepts a mismatched resize without erroring and then produces garbage
  /// coordinates, so this is declared once rather than at each call site.
  static const int _landmarkInputSize = 384;

  // Animal detection pipeline (full / poseOnly)
  AnimalDetector? _animalDetector;

  // Face pipeline (full / faceOnly)
  FaceLocalizerModel? _localizer;
  LandmarkModelRunnerBase? _lm;

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

  bool _isInitialized = false;

  /// Creates a cat detector with the specified configuration.
  CatDetectorCore({
    this.mode = CatDetectionMode.full,
    this.poseModel = AnimalPoseModel.rtmpose,
    this.landmarkModel = CatLandmarkModel.full,
    this.cropMargin = 0.20,
    this.detThreshold = 0.5,
    int interpreterPoolSize = 1,
    this.performanceConfig = const PerformanceConfig(),
  }) : interpreterPoolSize = performanceConfig.mode == PerformanceMode.disabled
            ? interpreterPoolSize
            : 1;

  /// Initializes the detector from pre-loaded model bytes.
  ///
  /// Called by the worker isolate that [CatDetector] spawns. All model bytes
  /// are loaded (and downloaded, if needed) on the main isolate and transferred
  /// in, since `rootBundle` is not available here.
  Future<void> initializeFromBuffers({
    Uint8List? localizerBytes,
    Uint8List? landmarkBytes,
    Uint8List? bodyDetectorBytes,
    Uint8List? classifierBytes,
    String? speciesMappingJson,
    Uint8List? poseModelBytes,
    bool useIsolateInterpreter = true,
  }) async {
    if (_isInitialized) {
      await dispose();
    }

    final bool needsBody =
        mode == CatDetectionMode.full || mode == CatDetectionMode.poseOnly;
    final bool needsFace =
        mode == CatDetectionMode.full || mode == CatDetectionMode.faceOnly;

    if (needsBody) {
      if (bodyDetectorBytes == null) {
        throw ArgumentError(
          'bodyDetectorBytes is required for full/poseOnly mode',
        );
      }
      if (classifierBytes == null) {
        throw ArgumentError(
          'classifierBytes is required for full/poseOnly mode',
        );
      }
      if (speciesMappingJson == null) {
        throw ArgumentError(
          'speciesMappingJson is required for full/poseOnly mode',
        );
      }
      if (poseModelBytes == null) {
        throw ArgumentError(
          'poseModelBytes is required for full/poseOnly mode',
        );
      }

      _animalDetector = AnimalDetector(
        poseModel: poseModel,
        enablePose: true,
        cropMargin: cropMargin,
        detThreshold: detThreshold,
        performanceConfig: performanceConfig,
      );
      await _animalDetector!.initializeFromBuffers(
        bodyDetectorBytes: bodyDetectorBytes,
        classifierBytes: classifierBytes,
        speciesMappingJson: speciesMappingJson,
        poseModelBytes: poseModelBytes,
        useIsolateInterpreter: useIsolateInterpreter,
      );
    }

    if (needsFace) {
      if (localizerBytes == null) {
        throw ArgumentError(
          'localizerBytes is required for full/faceOnly mode',
        );
      }
      if (landmarkBytes == null) {
        throw ArgumentError(
          'landmarkBytes is required for full/faceOnly mode',
        );
      }

      _localizer = FaceLocalizerModel(
        inputSize: 224,
        modelPath:
            'packages/cat_detection/assets/models/cat_face_localizer.tflite',
      );
      await _localizer!.initializeFromBuffer(
        localizerBytes,
        performanceConfig,
        useIsolateInterpreter: useIsolateInterpreter,
      );

      _lm = LandmarkModelRunnerBase(
        inputSize: _landmarkInputSize,
        numLandmarks: numCatLandmarks,
        modelPath:
            'packages/cat_detection/assets/models/cat_face_landmarks_full.tflite',
        poolSize: interpreterPoolSize,
      );
      await _lm!.initializeFromBuffer(
        landmarkBytes,
        performanceConfig,
        useIsolateInterpreter: useIsolateInterpreter,
      );
    }

    _isInitialized = true;
  }

  /// Returns true if the detector has been initialized and is ready to use.
  bool get isInitialized => _isInitialized;

  /// Releases all resources used by the detector.
  Future<void> dispose() async {
    await _animalDetector?.dispose();
    _localizer?.dispose();
    _lm?.dispose();
    _animalDetector = null;
    _localizer = null;
    _lm = null;
    _isInitialized = false;
  }

  /// Detects cats in an image from raw bytes.
  Future<List<Cat>> detect(Uint8List imageBytes) async {
    if (!_isInitialized) {
      throw StateError('CatDetector not initialized. Call initialize() first.');
    }
    try {
      final mat = cv.imdecode(imageBytes, cv.IMREAD_COLOR);
      if (mat.isEmpty) {
        return <Cat>[];
      }
      try {
        return await detectFromMat(
          mat,
          imageWidth: mat.cols,
          imageHeight: mat.rows,
        );
      } finally {
        mat.dispose();
      }
    } catch (e) {
      return <Cat>[];
    }
  }

  /// Detects cats in an OpenCV Mat image.
  Future<List<Cat>> detectFromMat(
    cv.Mat image, {
    required int imageWidth,
    required int imageHeight,
  }) async {
    if (!_isInitialized) {
      throw StateError('CatDetector not initialized. Call initialize() first.');
    }

    if (mode == CatDetectionMode.faceOnly) {
      return _detectFaceOnly(image, imageWidth, imageHeight);
    }

    return _detectWithBody(image, imageWidth, imageHeight);
  }

  /// Pipeline for [CatDetectionMode.faceOnly].
  ///
  /// Runs the face localizer directly on the whole image, then the landmark
  /// model on the box it returns. No SSD, species classifier or body pose, so
  /// the returned [Cat] has no species, breed or pose, and its bounding box is
  /// the face rather than the body.
  Future<List<Cat>> _detectFaceOnly(
    cv.Mat image,
    int imageWidth,
    int imageHeight,
  ) async {
    final BoundingBox? bbox = await _localizer!.detect(image);
    if (bbox == null) return <Cat>[];

    final CatFace face =
        await _runFaceLandmarks(image, bbox, imageWidth, imageHeight);

    return [
      Cat(
        boundingBox: bbox,
        score: 1.0,
        face: face,
        imageWidth: imageWidth,
        imageHeight: imageHeight,
      ),
    ];
  }

  /// Pipeline for [CatDetectionMode.full] and [CatDetectionMode.poseOnly].
  ///
  /// Uses [AnimalDetector] for SSD detection, species classification, and pose
  /// estimation, then runs face landmarks on each detected cat.
  Future<List<Cat>> _detectWithBody(
    cv.Mat image,
    int imageWidth,
    int imageHeight,
  ) async {
    final animals = await _animalDetector!.detectFromMat(
      image,
      imageWidth: imageWidth,
      imageHeight: imageHeight,
    );
    if (animals.isEmpty) return <Cat>[];

    final cats = <Cat>[];

    for (int i = 0; i < animals.length; i++) {
      final animal = animals[i];
      CatFace? face;

      if (mode == CatDetectionMode.full) {
        final (cx1, cy1, cx2, cy2) = ImageUtils.expandBox(
          animal.boundingBox.left,
          animal.boundingBox.top,
          animal.boundingBox.right,
          animal.boundingBox.bottom,
          cropMargin,
          imageWidth,
          imageHeight,
        );

        final int cropW = cx2 - cx1;
        final int cropH = cy2 - cy1;
        if (cropW >= 1 && cropH >= 1) {
          final expandedCrop = image.region(cv.Rect(cx1, cy1, cropW, cropH));
          try {
            final BoundingBox? faceBboxInCrop =
                await _localizer!.detect(expandedCrop);

            if (faceBboxInCrop != null) {
              final faceBboxInImage = BoundingBox.ltrb(
                faceBboxInCrop.left + cx1,
                faceBboxInCrop.top + cy1,
                faceBboxInCrop.right + cx1,
                faceBboxInCrop.bottom + cy1,
              );

              face = await _runFaceLandmarks(
                image,
                faceBboxInImage,
                imageWidth,
                imageHeight,
              );
            }
          } finally {
            expandedCrop.dispose();
          }
        }
      }

      cats.add(Cat(
        boundingBox: animal.boundingBox,
        score: animal.score,
        species: animal.species,
        breed: animal.breed,
        speciesConfidence: animal.speciesConfidence,
        face: face,
        pose: animal.pose,
        imageWidth: imageWidth,
        imageHeight: imageHeight,
      ));
    }

    return cats;
  }

  /// Crops the face region from [image] using [faceBbox] and runs landmark estimation.
  Future<CatFace> _runFaceLandmarks(
    cv.Mat image,
    BoundingBox faceBbox,
    int imageWidth,
    int imageHeight,
  ) async {
    final int cropSize = _lm!.inputSize;

    final (faceCrop, meta) = ImageUtils.cropAndResize(
      image,
      faceBbox,
      cropMargin,
      cropSize,
    );

    final List<CatLandmark> landmarks;
    try {
      final coords = await _lm!.predictRaw(faceCrop, meta);
      landmarks = [
        for (int i = 0; i < coords.length; i++)
          CatLandmark(
              type: CatLandmarkType.values[i],
              x: coords[i].$1,
              y: coords[i].$2),
      ];
    } finally {
      faceCrop.dispose();
    }

    return CatFace(boundingBox: faceBbox, landmarks: landmarks);
  }
}
