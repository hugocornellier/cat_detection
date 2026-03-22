import 'dart:typed_data';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:animal_detection/animal_detection.dart';
import 'types.dart';
import 'util/model_downloader.dart';

/// On-device cat detection using a unified multi-stage TensorFlow Lite pipeline.
///
/// Supports two modes:
/// - [CatDetectionMode.full]: SSD body detection + species classification +
///   body pose estimation + face landmarks.
/// - [CatDetectionMode.poseOnly]: Body detection + species + body pose only.
///
/// Uses [AnimalDetector] from the animal_detection package for body detection,
/// species classification, and pose estimation. Cat-specific face landmark
/// extraction is handled directly.
///
/// Usage:
/// ```dart
/// final detector = CatDetector(mode: CatDetectionMode.full);
/// await detector.initialize();
/// final cats = await detector.detect(imageBytes);
/// await detector.dispose();
/// ```
class CatDetector {
  // Animal detection pipeline (full / poseOnly)
  AnimalDetector? _animalDetector;

  // Face pipeline (full)
  FaceLocalizerModel? _localizer;
  LandmarkModelRunnerBase? _lm;
  EnsembleLandmarkModelBase? _ensemble;

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
  final PerformanceConfig performanceConfig;

  bool _isInitialized = false;

  /// Creates a cat detector with the specified configuration.
  CatDetector({
    this.mode = CatDetectionMode.full,
    this.poseModel = AnimalPoseModel.rtmpose,
    this.landmarkModel = CatLandmarkModel.full,
    this.cropMargin = 0.20,
    this.detThreshold = 0.5,
    int interpreterPoolSize = 1,
    this.performanceConfig = PerformanceConfig.disabled,
  }) : interpreterPoolSize = performanceConfig.mode == PerformanceMode.disabled
            ? interpreterPoolSize
            : 1;

  /// Initializes the detector by loading TensorFlow Lite models.
  ///
  /// Must be called before [detect] or [detectFromMat].
  /// If already initialized, disposes existing models before reinitializing.
  ///
  /// When [poseModel] is [AnimalPoseModel.hrnet], the HRNet model (~54.6 MB) is
  /// downloaded from GitHub Releases on first use and cached locally.
  ///
  /// When [landmarkModel] is [CatLandmarkModel.ensemble], the extra 320px and
  /// 384px models are downloaded on first use.
  ///
  /// [onDownloadProgress] is called during any model download with
  /// (modelName, bytesReceived, totalBytes).
  Future<void> initialize({
    void Function(String model, int received, int total)? onDownloadProgress,
    bool useIsolateInterpreter = true,
  }) async {
    if (_isInitialized) {
      await dispose();
    }

    final bool needsBody =
        mode == CatDetectionMode.full || mode == CatDetectionMode.poseOnly;
    final bool needsFace = mode == CatDetectionMode.full;

    if (needsBody) {
      _animalDetector = AnimalDetector(
        poseModel: poseModel,
        enablePose: true,
        cropMargin: cropMargin,
        detThreshold: detThreshold,
        performanceConfig: performanceConfig,
      );
      await _animalDetector!.initialize(
        onDownloadProgress: onDownloadProgress,
        useIsolateInterpreter: useIsolateInterpreter,
      );
    }

    if (needsFace) {
      _localizer = FaceLocalizerModel(
        inputSize: 224,
        modelPath:
            'packages/cat_detection/assets/models/cat_face_localizer.tflite',
      );
      await _localizer!.initialize(
        performanceConfig,
        useIsolateInterpreter: useIsolateInterpreter,
      );

      if (landmarkModel == CatLandmarkModel.ensemble) {
        _ensemble = EnsembleLandmarkModelBase(
          numLandmarks: numCatLandmarks,
          flipIndex: catLandmarkFlipIndex,
          bundledModelPath:
              'packages/cat_detection/assets/models/cat_face_landmarks_full.tflite',
          getEnsembleModels: CatModelDownloader.getEnsembleModels,
          poolSize: interpreterPoolSize,
        );
        await _ensemble!.initialize(
          performanceConfig,
          onDownloadProgress: onDownloadProgress,
          useIsolateInterpreter: useIsolateInterpreter,
        );
      } else {
        _lm = LandmarkModelRunnerBase(
          inputSize: 256,
          numLandmarks: numCatLandmarks,
          modelPath:
              'packages/cat_detection/assets/models/cat_face_landmarks_full.tflite',
          poolSize: interpreterPoolSize,
        );
        await _lm!.initialize(
          performanceConfig,
          useIsolateInterpreter: useIsolateInterpreter,
        );
      }
    }

    _isInitialized = true;
  }

  /// Initializes the detector from pre-loaded model bytes.
  ///
  /// Used by [CatDetectorIsolate] to initialize within a background isolate
  /// where Flutter asset loading is not available.
  Future<void> initializeFromBuffers({
    Uint8List? localizerBytes,
    Uint8List? landmarkBytes,
    Uint8List? ensemble256Bytes,
    Uint8List? ensemble320Bytes,
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
    final bool needsFace = mode == CatDetectionMode.full;

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
          'localizerBytes is required for full mode',
        );
      }
      if (landmarkBytes == null) {
        throw ArgumentError(
          'landmarkBytes is required for full mode',
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

      if (landmarkModel == CatLandmarkModel.ensemble) {
        if (ensemble256Bytes == null || ensemble320Bytes == null) {
          throw ArgumentError(
            'ensemble256Bytes and ensemble320Bytes are required for ensemble mode',
          );
        }
        _ensemble = EnsembleLandmarkModelBase(
          numLandmarks: numCatLandmarks,
          flipIndex: catLandmarkFlipIndex,
          bundledModelPath:
              'packages/cat_detection/assets/models/cat_face_landmarks_full.tflite',
          getEnsembleModels: CatModelDownloader.getEnsembleModels,
          poolSize: interpreterPoolSize,
        );
        await _ensemble!.initializeFromBuffers(
          bytes256: ensemble256Bytes,
          bytes320: ensemble320Bytes,
          bytes384: landmarkBytes,
          performanceConfig: performanceConfig,
          useIsolateInterpreter: useIsolateInterpreter,
        );
      } else {
        _lm = LandmarkModelRunnerBase(
          inputSize: 256,
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
    }

    _isInitialized = true;
  }

  /// Returns true if the detector has been initialized and is ready to use.
  bool get isInitialized => _isInitialized;

  /// Returns true if the ensemble models are already cached locally.
  static Future<bool> isEnsembleCached() =>
      CatModelDownloader.isEnsembleCached();

  /// Returns true if the HRNet model is already cached locally.
  static Future<bool> isHrnetCached() => ModelDownloader.isHrnetCached();

  /// Releases all resources used by the detector.
  Future<void> dispose() async {
    await _animalDetector?.dispose();
    _localizer?.dispose();
    _lm?.dispose();
    _ensemble?.dispose();
    _animalDetector = null;
    _localizer = null;
    _lm = null;
    _ensemble = null;
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

    return _detectWithBody(image, imageWidth, imageHeight);
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
    final int cropSize = landmarkModel == CatLandmarkModel.ensemble
        ? _ensemble!.inputSize
        : _lm!.inputSize;

    final (faceCrop, meta) = ImageUtils.cropAndResize(
      image,
      faceBbox,
      cropMargin,
      cropSize,
    );

    final List<CatLandmark> landmarks;
    try {
      if (landmarkModel == CatLandmarkModel.ensemble) {
        final coords = await _ensemble!.predictRaw(faceCrop, meta);
        landmarks = [
          for (int i = 0; i < coords.length; i++)
            CatLandmark(
                type: CatLandmarkType.values[i],
                x: coords[i].$1,
                y: coords[i].$2),
        ];
      } else {
        final coords = await _lm!.predictRaw(faceCrop, meta);
        landmarks = [
          for (int i = 0; i < coords.length; i++)
            CatLandmark(
                type: CatLandmarkType.values[i],
                x: coords[i].$1,
                y: coords[i].$2),
        ];
      }
    } finally {
      faceCrop.dispose();
    }

    return CatFace(boundingBox: faceBbox, landmarks: landmarks);
  }
}
