// Re-export shared types from animal_detection
export 'package:animal_detection/animal_detection.dart'
    show
        AnimalPoseModel,
        AnimalPoseLandmarkType,
        AnimalPoseLandmark,
        AnimalPose,
        animalPoseConnections,
        BoundingBox,
        Point,
        CropMetadata,
        Animal,
        PerformanceMode,
        PerformanceConfig;

import 'package:animal_detection/animal_detection.dart';

/// Cat landmark model variant for landmark extraction.
///
/// - [full]: Single model at 256px input resolution (bundled, ~55MB).
/// - [ensemble]: 3-model ensemble (256px + 320px + 384px) averaging predictions
///   for improved accuracy. The 320px and 384px models are downloaded on-demand
///   from GitHub Releases on first use.
enum CatLandmarkModel {
  /// Full model at 256px input resolution (bundled with the package).
  full,

  /// 3-model ensemble (256px + 320px + 384px) with multi-scale + flip TTA.
  ensemble,
}

/// Detection mode controlling the full pipeline behavior.
///
/// - [full]: SSD body detection + species + body pose + face landmarks.
/// - [poseOnly]: Body detection + species + body pose only (no face detection).
enum CatDetectionMode {
  /// Full pipeline: SSD body detection + species + body pose + face landmarks.
  full,

  /// Body detection + species + body pose only (no face detection).
  poseOnly,
}

/// Cat face landmark types based on the CatFLW dataset topology.
///
/// 48 landmarks covering ear contours, eyes, nose bridge, nose ring,
/// nose tips, and mouth/chin contour.
enum CatLandmarkType {
  /// Chin center (index 0). Self-symmetric under horizontal flip.
  chinCenter,

  /// Left face contour point in the left eye region (index 1).
  leftFaceContour,

  /// Muzzle center (index 2). Self-symmetric under horizontal flip.
  muzzleCenter,

  /// Right face contour point in the right eye region (index 3).
  rightFaceContour,

  /// Outer corner of right eye (index 4).
  rightEyeOuter,

  /// Right eye contour point 0 (index 5).
  rightEye0,

  /// Right eye contour point 1 (index 6).
  rightEye1,

  /// Right eye contour point 2 (index 7).
  rightEye2,

  /// Outer corner of left eye (index 8).
  leftEyeOuter,

  /// Left eye contour point 0 (index 9).
  leftEye0,

  /// Left eye contour point 1 (index 10).
  leftEye1,

  /// Left eye contour point 2 (index 11).
  leftEye2,

  /// Left side of nose (index 12).
  noseLeft,

  /// Right side of nose (index 13).
  noseRight,

  /// Left nose bridge point (index 14).
  noseBridgeLeft,

  /// Right nose bridge point (index 15).
  noseBridgeRight,

  /// Top of mouth (index 16). Self-symmetric under horizontal flip.
  mouthTop,

  /// Bottom of mouth (index 17). Self-symmetric under horizontal flip.
  mouthBottom,

  /// Left chin contour point 0 (index 18).
  chinLeft0,

  /// Right chin contour point 0 (index 19).
  chinRight0,

  /// Left chin contour point 1 (index 20).
  chinLeft1,

  /// Right chin contour point 1 (index 21).
  chinRight1,

  /// Right ear contour point 0 (index 22).
  rightEar0,

  /// Right ear contour point 1 (index 23).
  rightEar1,

  /// Right ear contour point 2 (index 24).
  rightEar2,

  /// Right ear contour point 3 (index 25).
  rightEar3,

  /// Right ear contour point 4 (index 26).
  rightEar4,

  /// Left ear contour point 0 (index 27).
  leftEar0,

  /// Left ear contour point 1 (index 28).
  leftEar1,

  /// Left ear contour point 2 (index 29).
  leftEar2,

  /// Left ear contour point 3 (index 30).
  leftEar3,

  /// Left ear contour point 4 (index 31).
  leftEar4,

  /// Left nose ring point 0 (index 32).
  noseRingLeft0,

  /// Left nose ring point 1 (index 33).
  noseRingLeft1,

  /// Right nose ring point 0 (index 34).
  noseRingRight0,

  /// Right nose ring point 1 (index 35).
  noseRingRight1,

  /// Top of right eye (index 36).
  rightEyeTop,

  /// Inner corner of right eye (index 37).
  rightEyeInner,

  /// Bottom of right eye (index 38).
  rightEyeBottom,

  /// Top of left eye (index 39).
  leftEyeTop,

  /// Inner corner of left eye (index 40).
  leftEyeInner,

  /// Bottom of left eye (index 41).
  leftEyeBottom,

  /// Left nose tip point (index 42).
  noseTipLeft,

  /// Right nose tip point (index 43).
  noseTipRight,

  /// Left nose wing point (index 44).
  noseWingLeft,

  /// Right nose wing point (index 45).
  noseWingRight,

  /// Left mouth corner (index 46).
  mouthCornerLeft,

  /// Right mouth corner (index 47).
  mouthCornerRight,
}

/// Number of cat face landmarks (48 for the CatFLW model).
const int numCatLandmarks = 48;

/// Landmark index permutation for horizontal flip (CatFLW convention).
///
/// When an image is horizontally flipped, left/right landmarks swap.
/// Used internally by the ensemble model for flip test-time augmentation.
const List<int> catLandmarkFlipIndex = [
  0, 3, 2, 1, 8, 9, 10, 11,
  4, 5, 6, 7, 13, 12, 15, 14,
  16, 17, 20, 21, 18, 19, 31, 30,
  29, 28, 27, 26, 25, 24, 23, 22,
  35, 34, 33, 32, 41, 40, 39, 38,
  37, 36, 43, 42, 45, 44, 47, 46,
];

/// A single cat face keypoint with 2D coordinates.
///
/// Coordinates are in the original image space (pixels).
class CatLandmark {
  /// The landmark type this represents
  final CatLandmarkType type;

  /// X coordinate in pixels (original image space)
  final double x;

  /// Y coordinate in pixels (original image space)
  final double y;

  /// Creates a cat face landmark with 2D coordinates.
  CatLandmark({
    required this.type,
    required this.x,
    required this.y,
  });

  /// Serializes this landmark to a map for cross-isolate transfer.
  Map<String, dynamic> toMap() => {
        'type': type.name,
        'x': x,
        'y': y,
      };

  /// Deserializes a landmark from a map.
  static CatLandmark fromMap(Map<String, dynamic> map) => CatLandmark(
        type: CatLandmarkType.values.firstWhere((e) => e.name == map['type']),
        x: (map['x'] as num).toDouble(),
        y: (map['y'] as num).toDouble(),
      );

  /// Converts x coordinate to normalized range (0.0 to 1.0)
  double xNorm(int imageWidth) => (x / imageWidth).clamp(0.0, 1.0);

  /// Converts y coordinate to normalized range (0.0 to 1.0)
  double yNorm(int imageHeight) => (y / imageHeight).clamp(0.0, 1.0);

  /// Converts landmark coordinates to a pixel point
  Point toPixel(int imageWidth, int imageHeight) {
    return Point(x, y);
  }
}

/// Defines the standard skeleton connections between cat face landmarks.
const List<List<CatLandmarkType>> catLandmarkConnections = [
  // Right ear contour
  [CatLandmarkType.rightEar0, CatLandmarkType.rightEar1],
  [CatLandmarkType.rightEar1, CatLandmarkType.rightEar2],
  [CatLandmarkType.rightEar2, CatLandmarkType.rightEar3],
  [CatLandmarkType.rightEar3, CatLandmarkType.rightEar4],
  // Left ear contour
  [CatLandmarkType.leftEar0, CatLandmarkType.leftEar1],
  [CatLandmarkType.leftEar1, CatLandmarkType.leftEar2],
  [CatLandmarkType.leftEar2, CatLandmarkType.leftEar3],
  [CatLandmarkType.leftEar3, CatLandmarkType.leftEar4],
  // Right eye (outer → top → inner → bottom → outer)
  [CatLandmarkType.rightEyeOuter, CatLandmarkType.rightEyeTop],
  [CatLandmarkType.rightEyeTop, CatLandmarkType.rightEyeInner],
  [CatLandmarkType.rightEyeInner, CatLandmarkType.rightEyeBottom],
  [CatLandmarkType.rightEyeBottom, CatLandmarkType.rightEyeOuter],
  // Left eye (outer → top → inner → bottom → outer)
  [CatLandmarkType.leftEyeOuter, CatLandmarkType.leftEyeTop],
  [CatLandmarkType.leftEyeTop, CatLandmarkType.leftEyeInner],
  [CatLandmarkType.leftEyeInner, CatLandmarkType.leftEyeBottom],
  [CatLandmarkType.leftEyeBottom, CatLandmarkType.leftEyeOuter],
  // Nose bridge
  [CatLandmarkType.noseBridgeLeft, CatLandmarkType.noseLeft],
  [CatLandmarkType.noseBridgeRight, CatLandmarkType.noseRight],
  // Nose ring
  [CatLandmarkType.noseRingLeft0, CatLandmarkType.noseRingLeft1],
  [CatLandmarkType.noseRingRight0, CatLandmarkType.noseRingRight1],
  [CatLandmarkType.noseRingLeft1, CatLandmarkType.noseRingRight0],
  // Nose tips
  [CatLandmarkType.noseTipLeft, CatLandmarkType.noseTipRight],
  [CatLandmarkType.noseWingLeft, CatLandmarkType.noseTipLeft],
  [CatLandmarkType.noseWingRight, CatLandmarkType.noseTipRight],
  // Mouth
  [CatLandmarkType.mouthCornerLeft, CatLandmarkType.mouthTop],
  [CatLandmarkType.mouthTop, CatLandmarkType.mouthCornerRight],
  [CatLandmarkType.mouthCornerLeft, CatLandmarkType.mouthBottom],
  [CatLandmarkType.mouthBottom, CatLandmarkType.mouthCornerRight],
];

/// Detected cat face with bounding box and optional landmarks.
class CatFace {
  /// Bounding box of the detected face in pixel coordinates
  final BoundingBox boundingBox;

  /// List of 48 landmarks. Empty if face landmarks were not run.
  final List<CatLandmark> landmarks;

  /// Creates a detected cat face with a bounding box and optional landmarks.
  const CatFace({
    required this.boundingBox,
    required this.landmarks,
  });

  /// Serializes this face to a map for cross-isolate transfer.
  Map<String, dynamic> toMap() => {
        'boundingBox': {
          'left': boundingBox.left,
          'top': boundingBox.top,
          'right': boundingBox.right,
          'bottom': boundingBox.bottom
        },
        'landmarks': landmarks.map((l) => l.toMap()).toList(),
      };

  /// Deserializes a cat face from a map.
  static CatFace fromMap(Map<String, dynamic> map) => CatFace(
        boundingBox: BoundingBox.ltrb(
          (map['boundingBox']['left'] as num).toDouble(),
          (map['boundingBox']['top'] as num).toDouble(),
          (map['boundingBox']['right'] as num).toDouble(),
          (map['boundingBox']['bottom'] as num).toDouble(),
        ),
        landmarks: (map['landmarks'] as List<dynamic>)
            .map((l) => CatLandmark.fromMap(l as Map<String, dynamic>))
            .toList(),
      );

  /// Gets a specific landmark by type, or null if not found
  CatLandmark? getLandmark(CatLandmarkType type) {
    try {
      return landmarks.firstWhere((l) => l.type == type);
    } catch (_) {
      return null;
    }
  }

  /// Returns true if this face has landmarks
  bool get hasLandmarks => landmarks.isNotEmpty;

  @override
  String toString() {
    final String landmarksInfo = landmarks
        .map((l) =>
            '${l.type.name}: (${l.x.toStringAsFixed(2)}, ${l.y.toStringAsFixed(2)})')
        .join('\n');
    return 'CatFace(\n'
        '  landmarks=${landmarks.length},\n'
        '  coords:\n$landmarksInfo\n)';
  }
}

/// Top-level result for a single detected cat.
///
/// Uses [AnimalPose] from the animal_detection package for body pose data.
class Cat {
  /// Body bounding box in pixel coordinates (original image space)
  final BoundingBox boundingBox;

  /// SSD detector confidence score (0.0 to 1.0)
  final double score;

  /// Predicted species label (e.g. "cat"), or null if classification was not run
  final String? species;

  /// Predicted breed label, or null if classification was not run
  final String? breed;

  /// Species classifier confidence (0.0 to 1.0), or null if not run
  final double? speciesConfidence;

  /// Face detection and landmark result, or null if not run / not found
  final CatFace? face;

  /// Body pose keypoints, or null if pose estimation was not run
  final AnimalPose? pose;

  /// Width of the original image in pixels
  final int imageWidth;

  /// Height of the original image in pixels
  final int imageHeight;

  /// Creates a top-level cat detection result.
  const Cat({
    required this.boundingBox,
    required this.score,
    this.species,
    this.breed,
    this.speciesConfidence,
    this.face,
    this.pose,
    required this.imageWidth,
    required this.imageHeight,
  });

  /// Serializes this result to a map for cross-isolate transfer.
  Map<String, dynamic> toMap() => {
        'boundingBox': {
          'left': boundingBox.left,
          'top': boundingBox.top,
          'right': boundingBox.right,
          'bottom': boundingBox.bottom
        },
        'score': score,
        'species': species,
        'breed': breed,
        'speciesConfidence': speciesConfidence,
        'face': face?.toMap(),
        'pose': pose?.toMap(),
        'imageWidth': imageWidth,
        'imageHeight': imageHeight,
      };

  /// Deserializes a cat detection result from a map.
  static Cat fromMap(Map<String, dynamic> map) => Cat(
        boundingBox: BoundingBox.ltrb(
          (map['boundingBox']['left'] as num).toDouble(),
          (map['boundingBox']['top'] as num).toDouble(),
          (map['boundingBox']['right'] as num).toDouble(),
          (map['boundingBox']['bottom'] as num).toDouble(),
        ),
        score: (map['score'] as num).toDouble(),
        species: map['species'] as String?,
        breed: map['breed'] as String?,
        speciesConfidence: (map['speciesConfidence'] as num?)?.toDouble(),
        face: map['face'] != null
            ? CatFace.fromMap(map['face'] as Map<String, dynamic>)
            : null,
        pose: map['pose'] != null
            ? AnimalPose.fromMap(map['pose'] as Map<String, dynamic>)
            : null,
        imageWidth: map['imageWidth'] as int,
        imageHeight: map['imageHeight'] as int,
      );

  @override
  String toString() =>
      'Cat(score=${score.toStringAsFixed(3)}, species=$species, breed=$breed, face=${face != null}, pose=${pose != null})';
}
