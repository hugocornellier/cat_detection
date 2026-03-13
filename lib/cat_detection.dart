/// On-device cat detection and landmark estimation using TensorFlow Lite.
///
/// This library provides a Flutter plugin for cat detection using a unified
/// multi-stage TFLite pipeline: SSD body detection, species classification,
/// body pose estimation, and face landmark extraction.
///
/// **Quick Start:**
/// ```dart
/// import 'package:cat_detection/cat_detection.dart';
///
/// final detector = CatDetector(mode: CatDetectionMode.full);
/// await detector.initialize();
///
/// final cats = await detector.detect(imageBytes);
/// for (final cat in cats) {
///   print('Cat at ${cat.boundingBox} score=${cat.score}');
///   if (cat.pose != null) {
///     final tail = cat.pose!.getLandmark(AnimalPoseLandmarkType.tailEnd);
///     print('Tail: (${tail?.x}, ${tail?.y})');
///   }
///   if (cat.face != null && cat.face!.hasLandmarks) {
///     final nose = cat.face!.getLandmark(CatLandmarkType.noseTipLeft);
///     print('Nose tip: (${nose?.x}, ${nose?.y})');
///   }
/// }
///
/// await detector.dispose();
/// ```
///
/// **Main Classes:**
/// - [CatDetectorIsolate]: Background isolate wrapper for cat detection
/// - [CatDetector]: Main API for cat detection
/// - [Cat]: Top-level detection result with body, pose and face info
/// - [CatFace]: Detected cat face with bounding box and landmarks
/// - [CatLandmark]: Single face keypoint with 2D coordinates
/// - [BoundingBox]: Axis-aligned rectangle in pixel coordinates
///
/// **Detection Modes:**
/// - [CatDetectionMode.full]: SSD body detection + species + body pose + face landmarks
/// - [CatDetectionMode.poseOnly]: Body detection + species + body pose only
///
/// **Pose Model Variants:**
/// - [AnimalPoseModel.rtmpose]: RTMPose-S (11.6MB, bundled). Fast SimCC-based decoder.
/// - [AnimalPoseModel.hrnet]: HRNet-w32 (54.6MB, downloaded on demand). Most accurate.
///
/// **Face Landmark Model Variants:**
/// - [CatLandmarkModel.full]: Single model at 256px input resolution (bundled)
/// - [CatLandmarkModel.ensemble]: 3-model ensemble (256px + 320px + 384px) with
///   multi-scale + flip TTA. Extra models downloaded on-demand from GitHub Releases.
///
/// **Skeleton Connections:**
/// - [catLandmarkConnections]: Face landmark skeleton edges (CatFLW 48-landmark topology)
/// - [animalPoseConnections]: Body pose skeleton edges (SuperAnimal topology)
library;

export 'src/types.dart';
export 'src/cat_detector.dart' show CatDetector;
export 'src/isolate/cat_detector_isolate.dart' show CatDetectorIsolate;

// Re-export everything from animal_detection that consumers need
export 'package:animal_detection/animal_detection.dart'
    show
        AnimalDetector,
        Animal,
        AnimalPoseModel,
        AnimalPose,
        AnimalPoseLandmark,
        AnimalPoseLandmarkType,
        BoundingBox,
        Point,
        CropMetadata,
        animalPoseConnections,
        ModelDownloader,
        PerformanceMode,
        PerformanceConfig,
        Mat,
        imdecode,
        IMREAD_COLOR;

export 'src/dart_registration.dart';
