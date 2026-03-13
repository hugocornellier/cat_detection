import 'package:flutter_test/flutter_test.dart';
import 'package:cat_detection/cat_detection.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  // ---------------------------------------------------------------------------
  // CatLandmarkModel enum
  // ---------------------------------------------------------------------------
  group('CatLandmarkModel enum', () {
    test('has exactly 2 values', () {
      expect(CatLandmarkModel.values.length, 2);
    });

    test('full is at index 0', () {
      expect(CatLandmarkModel.full.index, 0);
    });

    test('ensemble is at index 1', () {
      expect(CatLandmarkModel.ensemble.index, 1);
    });

    test('values are full and ensemble', () {
      expect(CatLandmarkModel.values.contains(CatLandmarkModel.full), true);
      expect(CatLandmarkModel.values.contains(CatLandmarkModel.ensemble), true);
    });

    test('name property works', () {
      expect(CatLandmarkModel.full.name, 'full');
      expect(CatLandmarkModel.ensemble.name, 'ensemble');
    });
  });

  // ---------------------------------------------------------------------------
  // CatDetectionMode enum
  // ---------------------------------------------------------------------------
  group('CatDetectionMode enum', () {
    test('has exactly 2 values', () {
      expect(CatDetectionMode.values.length, 2);
    });

    test('values are full and poseOnly', () {
      expect(CatDetectionMode.values.contains(CatDetectionMode.full), true);
      expect(CatDetectionMode.values.contains(CatDetectionMode.poseOnly), true);
    });

    test('full has index 0', () {
      expect(CatDetectionMode.full.index, 0);
    });

    test('poseOnly has index 1', () {
      expect(CatDetectionMode.poseOnly.index, 1);
    });

    test('name property works', () {
      expect(CatDetectionMode.full.name, 'full');
      expect(CatDetectionMode.poseOnly.name, 'poseOnly');
    });
  });

  // ---------------------------------------------------------------------------
  // CatLandmarkType enum — 48 values, CatFLW dataset topology
  // ---------------------------------------------------------------------------
  group('CatLandmarkType enum', () {
    test('has exactly 48 values', () {
      expect(CatLandmarkType.values.length, 48);
    });

    test('first and last landmark indices', () {
      expect(CatLandmarkType.chinCenter.index, 0);
      expect(CatLandmarkType.mouthCornerRight.index, 47);
    });

    test('chin and face contour indices (0-3)', () {
      expect(CatLandmarkType.chinCenter.index, 0);
      expect(CatLandmarkType.leftFaceContour.index, 1);
      expect(CatLandmarkType.muzzleCenter.index, 2);
      expect(CatLandmarkType.rightFaceContour.index, 3);
    });

    test('eye region indices (4-11)', () {
      expect(CatLandmarkType.rightEyeOuter.index, 4);
      expect(CatLandmarkType.rightEye0.index, 5);
      expect(CatLandmarkType.rightEye1.index, 6);
      expect(CatLandmarkType.rightEye2.index, 7);
      expect(CatLandmarkType.leftEyeOuter.index, 8);
      expect(CatLandmarkType.leftEye0.index, 9);
      expect(CatLandmarkType.leftEye1.index, 10);
      expect(CatLandmarkType.leftEye2.index, 11);
    });

    test('nose region indices (12-15)', () {
      expect(CatLandmarkType.noseLeft.index, 12);
      expect(CatLandmarkType.noseRight.index, 13);
      expect(CatLandmarkType.noseBridgeLeft.index, 14);
      expect(CatLandmarkType.noseBridgeRight.index, 15);
    });

    test('mouth indices (16-17)', () {
      expect(CatLandmarkType.mouthTop.index, 16);
      expect(CatLandmarkType.mouthBottom.index, 17);
    });

    test('chin contour indices (18-21)', () {
      expect(CatLandmarkType.chinLeft0.index, 18);
      expect(CatLandmarkType.chinRight0.index, 19);
      expect(CatLandmarkType.chinLeft1.index, 20);
      expect(CatLandmarkType.chinRight1.index, 21);
    });

    test('right ear contour indices (22-26)', () {
      expect(CatLandmarkType.rightEar0.index, 22);
      expect(CatLandmarkType.rightEar1.index, 23);
      expect(CatLandmarkType.rightEar2.index, 24);
      expect(CatLandmarkType.rightEar3.index, 25);
      expect(CatLandmarkType.rightEar4.index, 26);
    });

    test('left ear contour indices (27-31)', () {
      expect(CatLandmarkType.leftEar0.index, 27);
      expect(CatLandmarkType.leftEar1.index, 28);
      expect(CatLandmarkType.leftEar2.index, 29);
      expect(CatLandmarkType.leftEar3.index, 30);
      expect(CatLandmarkType.leftEar4.index, 31);
    });

    test('nose ring indices (32-35)', () {
      expect(CatLandmarkType.noseRingLeft0.index, 32);
      expect(CatLandmarkType.noseRingLeft1.index, 33);
      expect(CatLandmarkType.noseRingRight0.index, 34);
      expect(CatLandmarkType.noseRingRight1.index, 35);
    });

    test('eye detail indices (36-41)', () {
      expect(CatLandmarkType.rightEyeTop.index, 36);
      expect(CatLandmarkType.rightEyeInner.index, 37);
      expect(CatLandmarkType.rightEyeBottom.index, 38);
      expect(CatLandmarkType.leftEyeTop.index, 39);
      expect(CatLandmarkType.leftEyeInner.index, 40);
      expect(CatLandmarkType.leftEyeBottom.index, 41);
    });

    test('nose tip and wing indices (42-45)', () {
      expect(CatLandmarkType.noseTipLeft.index, 42);
      expect(CatLandmarkType.noseTipRight.index, 43);
      expect(CatLandmarkType.noseWingLeft.index, 44);
      expect(CatLandmarkType.noseWingRight.index, 45);
    });

    test('mouth corner indices (46-47)', () {
      expect(CatLandmarkType.mouthCornerLeft.index, 46);
      expect(CatLandmarkType.mouthCornerRight.index, 47);
    });

    test('verify specific landmark names by index', () {
      expect(CatLandmarkType.values[0].name, 'chinCenter');
      expect(CatLandmarkType.values[1].name, 'leftFaceContour');
      expect(CatLandmarkType.values[2].name, 'muzzleCenter');
      expect(CatLandmarkType.values[3].name, 'rightFaceContour');
      expect(CatLandmarkType.values[4].name, 'rightEyeOuter');
      expect(CatLandmarkType.values[8].name, 'leftEyeOuter');
      expect(CatLandmarkType.values[12].name, 'noseLeft');
      expect(CatLandmarkType.values[13].name, 'noseRight');
      expect(CatLandmarkType.values[22].name, 'rightEar0');
      expect(CatLandmarkType.values[26].name, 'rightEar4');
      expect(CatLandmarkType.values[27].name, 'leftEar0');
      expect(CatLandmarkType.values[31].name, 'leftEar4');
      expect(CatLandmarkType.values[32].name, 'noseRingLeft0');
      expect(CatLandmarkType.values[35].name, 'noseRingRight1');
      expect(CatLandmarkType.values[46].name, 'mouthCornerLeft');
      expect(CatLandmarkType.values[47].name, 'mouthCornerRight');
    });
  });

  // ---------------------------------------------------------------------------
  // numCatLandmarks constant
  // ---------------------------------------------------------------------------
  group('numCatLandmarks constant', () {
    test('equals 48', () {
      expect(numCatLandmarks, 48);
    });

    test('matches CatLandmarkType.values.length', () {
      expect(numCatLandmarks, CatLandmarkType.values.length);
    });
  });

  // ---------------------------------------------------------------------------
  // catLandmarkFlipIndex constant
  // ---------------------------------------------------------------------------
  group('catLandmarkFlipIndex', () {
    test('has exactly 48 entries', () {
      expect(catLandmarkFlipIndex.length, 48);
    });

    test('all indices are valid (0-47)', () {
      for (final idx in catLandmarkFlipIndex) {
        expect(idx, greaterThanOrEqualTo(0));
        expect(idx, lessThan(48));
      }
    });

    test('is a valid permutation (each index appears exactly once)', () {
      final sorted = List<int>.from(catLandmarkFlipIndex)..sort();
      expect(sorted, List.generate(48, (i) => i));
    });

    test('is an involution (applying twice gives identity)', () {
      for (int i = 0; i < 48; i++) {
        expect(catLandmarkFlipIndex[catLandmarkFlipIndex[i]], i);
      }
    });

    test('self-symmetric entries: chinCenter (0) maps to itself', () {
      expect(catLandmarkFlipIndex[0], 0);
    });

    test('self-symmetric entries: muzzleCenter (2) maps to itself', () {
      expect(catLandmarkFlipIndex[2], 2);
    });

    test('self-symmetric entries: mouthTop (16) maps to itself', () {
      expect(catLandmarkFlipIndex[16], 16);
    });

    test('self-symmetric entries: mouthBottom (17) maps to itself', () {
      expect(catLandmarkFlipIndex[17], 17);
    });

    test('swaps leftFaceContour (1) <-> rightFaceContour (3)', () {
      expect(catLandmarkFlipIndex[1], 3);
      expect(catLandmarkFlipIndex[3], 1);
    });

    test('swaps right eye outer (4) <-> left eye outer (8)', () {
      expect(catLandmarkFlipIndex[4], 8);
      expect(catLandmarkFlipIndex[8], 4);
    });

    test('swaps right eye points (5-7) <-> left eye points (9-11)', () {
      expect(catLandmarkFlipIndex[5], 9);
      expect(catLandmarkFlipIndex[9], 5);
      expect(catLandmarkFlipIndex[6], 10);
      expect(catLandmarkFlipIndex[10], 6);
      expect(catLandmarkFlipIndex[7], 11);
      expect(catLandmarkFlipIndex[11], 7);
    });

    test('swaps noseLeft (12) <-> noseRight (13)', () {
      expect(catLandmarkFlipIndex[12], 13);
      expect(catLandmarkFlipIndex[13], 12);
    });

    test('swaps noseBridgeLeft (14) <-> noseBridgeRight (15)', () {
      expect(catLandmarkFlipIndex[14], 15);
      expect(catLandmarkFlipIndex[15], 14);
    });

    test('swaps chin contour pairs', () {
      expect(catLandmarkFlipIndex[18], 20);
      expect(catLandmarkFlipIndex[20], 18);
      expect(catLandmarkFlipIndex[19], 21);
      expect(catLandmarkFlipIndex[21], 19);
    });

    test('swaps right ear (22-26) <-> left ear (27-31)', () {
      expect(catLandmarkFlipIndex[22], 31);
      expect(catLandmarkFlipIndex[31], 22);
      expect(catLandmarkFlipIndex[23], 30);
      expect(catLandmarkFlipIndex[30], 23);
      expect(catLandmarkFlipIndex[24], 29);
      expect(catLandmarkFlipIndex[29], 24);
      expect(catLandmarkFlipIndex[25], 28);
      expect(catLandmarkFlipIndex[28], 25);
      expect(catLandmarkFlipIndex[26], 27);
      expect(catLandmarkFlipIndex[27], 26);
    });

    test('swaps nose ring left <-> nose ring right', () {
      expect(catLandmarkFlipIndex[32], 35);
      expect(catLandmarkFlipIndex[35], 32);
      expect(catLandmarkFlipIndex[33], 34);
      expect(catLandmarkFlipIndex[34], 33);
    });

    test('swaps right eye detail (36-38) <-> left eye detail (39-41)', () {
      expect(catLandmarkFlipIndex[36], 41);
      expect(catLandmarkFlipIndex[41], 36);
      expect(catLandmarkFlipIndex[37], 40);
      expect(catLandmarkFlipIndex[40], 37);
      expect(catLandmarkFlipIndex[38], 39);
      expect(catLandmarkFlipIndex[39], 38);
    });

    test('swaps noseTipLeft (42) <-> noseTipRight (43)', () {
      expect(catLandmarkFlipIndex[42], 43);
      expect(catLandmarkFlipIndex[43], 42);
    });

    test('swaps noseWingLeft (44) <-> noseWingRight (45)', () {
      expect(catLandmarkFlipIndex[44], 45);
      expect(catLandmarkFlipIndex[45], 44);
    });

    test('swaps mouthCornerLeft (46) <-> mouthCornerRight (47)', () {
      expect(catLandmarkFlipIndex[46], 47);
      expect(catLandmarkFlipIndex[47], 46);
    });

    test('matches expected flip index exactly', () {
      const expectedFlipIndex = [
        0, 3, 2, 1, 8, 9, 10, 11,
        4, 5, 6, 7, 13, 12, 15, 14,
        16, 17, 20, 21, 18, 19, 31, 30,
        29, 28, 27, 26, 25, 24, 23, 22,
        35, 34, 33, 32, 41, 40, 39, 38,
        37, 36, 43, 42, 45, 44, 47, 46,
      ];
      expect(catLandmarkFlipIndex, expectedFlipIndex);
    });
  });

  // ---------------------------------------------------------------------------
  // BoundingBox class
  // ---------------------------------------------------------------------------
  group('BoundingBox', () {
    test('ltrb constructor stores left, top, right, bottom', () {
      final bbox = BoundingBox.ltrb(10.5, 20.3, 100.7, 200.1);
      expect(bbox.left, 10.5);
      expect(bbox.top, 20.3);
      expect(bbox.right, 100.7);
      expect(bbox.bottom, 200.1);
    });

    test('toMap produces correct map', () {
      final bbox = BoundingBox.ltrb(1.0, 2.0, 3.0, 4.0);
      final map = bbox.toMap();
      expect(map.containsKey('topLeft'), true);
      expect(map.containsKey('topRight'), true);
      expect(map.containsKey('bottomRight'), true);
      expect(map.containsKey('bottomLeft'), true);
    });

    test('fromMap factory reconstructs correctly', () {
      final bbox = BoundingBox.ltrb(10.5, 20.3, 100.7, 200.1);
      final restored = BoundingBox.fromMap(bbox.toMap());
      expect(restored.left, 10.5);
      expect(restored.top, 20.3);
      expect(restored.right, 100.7);
      expect(restored.bottom, 200.1);
    });

    test('toMap/fromMap round-trip', () {
      final original = BoundingBox.ltrb(10.5, 20.3, 100.7, 200.1);
      final restored = BoundingBox.fromMap(original.toMap());
      expect(restored.left, 10.5);
      expect(restored.top, 20.3);
      expect(restored.right, 100.7);
      expect(restored.bottom, 200.1);
    });

    test('zero-size box is preserved', () {
      final bbox = BoundingBox.ltrb(50.0, 50.0, 50.0, 50.0);
      final restored = BoundingBox.fromMap(bbox.toMap());
      expect(restored.left, restored.right);
      expect(restored.top, restored.bottom);
    });

    test('negative coordinates are stored as-is', () {
      final bbox = BoundingBox.ltrb(-50.0, -30.0, -10.0, -5.0);
      expect(bbox.left, -50.0);
      expect(bbox.top, -30.0);
      expect(bbox.right, -10.0);
      expect(bbox.bottom, -5.0);
    });

    test('negative coordinates round-trip via toMap/fromMap', () {
      final original = BoundingBox.ltrb(-100.0, -80.0, -20.0, -10.0);
      final restored = BoundingBox.fromMap(original.toMap());
      expect(restored.left, -100.0);
      expect(restored.top, -80.0);
      expect(restored.right, -20.0);
      expect(restored.bottom, -10.0);
    });
  });

  // ---------------------------------------------------------------------------
  // CatLandmark class
  // ---------------------------------------------------------------------------
  group('CatLandmark', () {
    test('constructor stores all fields correctly', () {
      final landmark = CatLandmark(
        type: CatLandmarkType.noseBridgeLeft,
        x: 100.5,
        y: 200.3,
      );
      expect(landmark.type, CatLandmarkType.noseBridgeLeft);
      expect(landmark.x, 100.5);
      expect(landmark.y, 200.3);
    });

    test('toMap produces correct map with type, x, y keys', () {
      final landmark = CatLandmark(
        type: CatLandmarkType.chinCenter,
        x: 10.0,
        y: 20.0,
      );
      final map = landmark.toMap();
      expect(map['type'], 'chinCenter');
      expect(map['x'], 10.0);
      expect(map['y'], 20.0);
      expect(map.containsKey('type'), true);
      expect(map.containsKey('x'), true);
      expect(map.containsKey('y'), true);
    });

    test('fromMap factory reconstructs correctly', () {
      final map = {'type': 'muzzleCenter', 'x': 50.0, 'y': 60.0};
      final landmark = CatLandmark.fromMap(map);
      expect(landmark.type, CatLandmarkType.muzzleCenter);
      expect(landmark.x, 50.0);
      expect(landmark.y, 60.0);
    });

    test('fromMap handles integer coordinates', () {
      final map = {'type': 'leftEar0', 'x': 100, 'y': 200};
      final landmark = CatLandmark.fromMap(map);
      expect(landmark.type, CatLandmarkType.leftEar0);
      expect(landmark.x, 100.0);
      expect(landmark.y, 200.0);
    });

    test('toMap/fromMap round-trip', () {
      final original = CatLandmark(
        type: CatLandmarkType.rightEar2,
        x: 123.45,
        y: 678.9,
      );
      final restored = CatLandmark.fromMap(original.toMap());
      expect(restored.type, CatLandmarkType.rightEar2);
      expect(restored.x, 123.45);
      expect(restored.y, 678.9);
    });

    test('round-trip all landmark types', () {
      for (final type in CatLandmarkType.values) {
        final original = CatLandmark(type: type, x: 50.0, y: 50.0);
        final restored = CatLandmark.fromMap(original.toMap());
        expect(restored.type, type);
      }
    });

    test('xNorm returns x / width', () {
      final landmark = CatLandmark(
        type: CatLandmarkType.noseBridgeLeft,
        x: 320.0,
        y: 240.0,
      );
      expect(landmark.xNorm(640), closeTo(0.5, 0.0001));
    });

    test('yNorm returns y / height', () {
      final landmark = CatLandmark(
        type: CatLandmarkType.noseBridgeLeft,
        x: 320.0,
        y: 240.0,
      );
      expect(landmark.yNorm(480), closeTo(0.5, 0.0001));
    });

    test('xNorm clamps negative x to 0.0', () {
      final landmark = CatLandmark(
        type: CatLandmarkType.leftEar0,
        x: -10.0,
        y: 100.0,
      );
      expect(landmark.xNorm(640), 0.0);
    });

    test('yNorm clamps negative y to 0.0', () {
      final landmark = CatLandmark(
        type: CatLandmarkType.leftEar0,
        x: 100.0,
        y: -50.0,
      );
      expect(landmark.yNorm(480), 0.0);
    });

    test('xNorm clamps x beyond width to 1.0', () {
      final landmark = CatLandmark(
        type: CatLandmarkType.rightEar0,
        x: 800.0,
        y: 100.0,
      );
      expect(landmark.xNorm(640), 1.0);
    });

    test('yNorm clamps y beyond height to 1.0', () {
      final landmark = CatLandmark(
        type: CatLandmarkType.rightEar0,
        x: 100.0,
        y: 600.0,
      );
      expect(landmark.yNorm(480), 1.0);
    });

    test('xNorm with width = 1 clamps out-of-range x', () {
      final inRange = CatLandmark(
        type: CatLandmarkType.muzzleCenter,
        x: 0.5,
        y: 0.0,
      );
      expect(inRange.xNorm(1), closeTo(0.5, 0.0001));

      final over = CatLandmark(
        type: CatLandmarkType.muzzleCenter,
        x: 2.0,
        y: 0.0,
      );
      expect(over.xNorm(1), 1.0);

      final under = CatLandmark(
        type: CatLandmarkType.muzzleCenter,
        x: -1.0,
        y: 0.0,
      );
      expect(under.xNorm(1), 0.0);
    });

    test('yNorm with height = 1 clamps out-of-range y', () {
      final inRange = CatLandmark(
        type: CatLandmarkType.muzzleCenter,
        x: 0.0,
        y: 0.5,
      );
      expect(inRange.yNorm(1), closeTo(0.5, 0.0001));

      final over = CatLandmark(
        type: CatLandmarkType.muzzleCenter,
        x: 0.0,
        y: 2.0,
      );
      expect(over.yNorm(1), 1.0);

      final under = CatLandmark(
        type: CatLandmarkType.muzzleCenter,
        x: 0.0,
        y: -1.0,
      );
      expect(under.yNorm(1), 0.0);
    });

    test('toPixel returns Point with coordinates', () {
      final landmark = CatLandmark(
        type: CatLandmarkType.leftEyeOuter,
        x: 123.7,
        y: 456.9,
      );
      final point = landmark.toPixel(640, 480);
      expect(point.x, 123.7);
      expect(point.y, 456.9);
    });

    test('toPixel with whole-number coordinates', () {
      final landmark = CatLandmark(
        type: CatLandmarkType.rightEyeOuter,
        x: 200.0,
        y: 150.0,
      );
      final point = landmark.toPixel(640, 480);
      expect(point.x, 200.0);
      expect(point.y, 150.0);
    });

    test('edge case: zero coordinates', () {
      final landmark = CatLandmark(
        type: CatLandmarkType.chinCenter,
        x: 0.0,
        y: 0.0,
      );
      expect(landmark.x, 0.0);
      expect(landmark.y, 0.0);
      expect(landmark.xNorm(640), 0.0);
      expect(landmark.yNorm(480), 0.0);
      final point = landmark.toPixel(640, 480);
      expect(point.x, 0.0);
      expect(point.y, 0.0);
    });

    test('edge case: negative coordinates', () {
      final landmark = CatLandmark(
        type: CatLandmarkType.noseRingLeft0,
        x: -10.9,
        y: -5.3,
      );
      expect(landmark.x, -10.9);
      expect(landmark.y, -5.3);
      final point = landmark.toPixel(640, 480);
      expect(point.x, -10.9);
      expect(point.y, -5.3);
    });

    test('edge case: very large coordinates', () {
      final landmark = CatLandmark(
        type: CatLandmarkType.noseRingRight1,
        x: 10000.0,
        y: 10000.0,
      );
      expect(landmark.x, 10000.0);
      expect(landmark.y, 10000.0);
      final restored = CatLandmark.fromMap(landmark.toMap());
      expect(restored.x, 10000.0);
      expect(restored.y, 10000.0);
    });

    test('xNorm at exact boundary (x == width)', () {
      final landmark = CatLandmark(
        type: CatLandmarkType.mouthTop,
        x: 640.0,
        y: 0.0,
      );
      expect(landmark.xNorm(640), 1.0);
    });

    test('yNorm at exact boundary (y == height)', () {
      final landmark = CatLandmark(
        type: CatLandmarkType.mouthBottom,
        x: 0.0,
        y: 480.0,
      );
      expect(landmark.yNorm(480), 1.0);
    });

    test('fromMap reconstructs all 48 landmark type names', () {
      for (final type in CatLandmarkType.values) {
        final map = {'type': type.name, 'x': 1.0, 'y': 2.0};
        final lm = CatLandmark.fromMap(map);
        expect(lm.type, type);
      }
    });
  });

  // ---------------------------------------------------------------------------
  // CatFace class
  // ---------------------------------------------------------------------------
  group('CatFace', () {
    CatLandmark makeLandmark(CatLandmarkType type,
        {double x = 0, double y = 0}) {
      return CatLandmark(type: type, x: x, y: y);
    }

    CatFace makeFullFace() {
      return CatFace(
        boundingBox: BoundingBox.ltrb(10.0, 20.0, 200.0, 300.0),
        landmarks: [
          makeLandmark(CatLandmarkType.noseBridgeLeft, x: 150.0, y: 180.0),
          makeLandmark(CatLandmarkType.leftEyeOuter, x: 100.0, y: 120.0),
          makeLandmark(CatLandmarkType.rightEyeOuter, x: 200.0, y: 120.0),
        ],
      );
    }

    test('constructor stores all fields', () {
      final face = makeFullFace();
      expect(face.boundingBox.left, 10.0);
      expect(face.boundingBox.top, 20.0);
      expect(face.boundingBox.right, 200.0);
      expect(face.boundingBox.bottom, 300.0);
      expect(face.landmarks.length, 3);
    });

    test('getLandmark returns correct landmark by type', () {
      final face = makeFullFace();
      final nose = face.getLandmark(CatLandmarkType.noseBridgeLeft);
      expect(nose, isNotNull);
      expect(nose!.type, CatLandmarkType.noseBridgeLeft);
      expect(nose.x, 150.0);
      expect(nose.y, 180.0);
    });

    test('getLandmark returns null for missing type', () {
      final face = makeFullFace();
      final missing = face.getLandmark(CatLandmarkType.chinCenter);
      expect(missing, isNull);
    });

    test('getLandmark returns null for empty landmarks list', () {
      final face = CatFace(
        boundingBox: BoundingBox.ltrb(0, 0, 100, 100),
        landmarks: [],
      );
      expect(face.getLandmark(CatLandmarkType.noseBridgeLeft), isNull);
    });

    test('hasLandmarks returns true when landmarks non-empty', () {
      final face = makeFullFace();
      expect(face.hasLandmarks, true);
    });

    test('hasLandmarks returns false when landmarks empty', () {
      final face = CatFace(
        boundingBox: BoundingBox.ltrb(0, 0, 100, 100),
        landmarks: [],
      );
      expect(face.hasLandmarks, false);
    });

    test('toMap serializes bbox and landmarks list', () {
      final face = makeFullFace();
      final map = face.toMap();

      expect(map.containsKey('boundingBox'), true);
      expect(map.containsKey('landmarks'), true);

      final bbox = map['boundingBox'] as Map<String, dynamic>;
      expect(bbox['left'], 10.0);
      expect(bbox['top'], 20.0);
      expect(bbox['right'], 200.0);
      expect(bbox['bottom'], 300.0);

      final landmarksList = map['landmarks'] as List;
      expect(landmarksList.length, 3);
    });

    test('fromMap deserializes correctly', () {
      final map = {
        'boundingBox': {
          'left': 5.0,
          'top': 10.0,
          'right': 200.0,
          'bottom': 300.0
        },
        'landmarks': [
          {'type': 'muzzleCenter', 'x': 100.0, 'y': 150.0},
        ],
      };
      final face = CatFace.fromMap(map);
      expect(face.boundingBox.left, 5.0);
      expect(face.landmarks.length, 1);
      expect(face.landmarks[0].type, CatLandmarkType.muzzleCenter);
    });

    test('toMap/fromMap round-trip', () {
      final original = makeFullFace();
      final restored = CatFace.fromMap(original.toMap());

      expect(restored.boundingBox.left, original.boundingBox.left);
      expect(restored.boundingBox.top, original.boundingBox.top);
      expect(restored.boundingBox.right, original.boundingBox.right);
      expect(restored.boundingBox.bottom, original.boundingBox.bottom);
      expect(restored.landmarks.length, original.landmarks.length);
      expect(restored.landmarks[0].type, CatLandmarkType.noseBridgeLeft);
    });

    test('fromMap with empty landmarks list', () {
      final map = {
        'boundingBox': {
          'left': 0.0,
          'top': 0.0,
          'right': 100.0,
          'bottom': 100.0
        },
        'landmarks': [],
      };
      final face = CatFace.fromMap(map);
      expect(face.landmarks, isEmpty);
    });

    test('toMap/fromMap round-trip with all 48 landmarks', () {
      final allLandmarks = CatLandmarkType.values
          .map((type) => CatLandmark(type: type, x: type.index * 2.0, y: type.index * 3.0))
          .toList();
      final original = CatFace(
        boundingBox: BoundingBox.ltrb(0, 0, 400, 300),
        landmarks: allLandmarks,
      );
      final restored = CatFace.fromMap(original.toMap());
      expect(restored.landmarks.length, 48);
      for (int i = 0; i < 48; i++) {
        expect(restored.landmarks[i].type, CatLandmarkType.values[i]);
        expect(restored.landmarks[i].x, CatLandmarkType.values[i].index * 2.0);
        expect(restored.landmarks[i].y, CatLandmarkType.values[i].index * 3.0);
      }
    });

    test('toString does not crash', () {
      final face = makeFullFace();
      expect(() => face.toString(), returnsNormally);
    });

    test('toString contains CatFace prefix', () {
      final face = makeFullFace();
      final str = face.toString();
      expect(str, contains('CatFace('));
    });

    test('toString with no landmarks does not crash', () {
      final face = CatFace(
        boundingBox: BoundingBox.ltrb(0, 0, 100, 100),
        landmarks: [],
      );
      expect(() => face.toString(), returnsNormally);
      final str = face.toString();
      expect(str, contains('landmarks=0'));
    });

    test('toString contains landmark count', () {
      final face = makeFullFace();
      final str = face.toString();
      expect(str, contains('landmarks=3'));
    });

    test('edge case: empty landmarks list', () {
      final face = CatFace(
        boundingBox: BoundingBox.ltrb(0, 0, 100, 100),
        landmarks: [],
      );
      expect(face.landmarks, isEmpty);
      expect(face.hasLandmarks, false);
      expect(face.getLandmark(CatLandmarkType.chinCenter), isNull);
    });

    test('edge case: single landmark', () {
      final landmark =
          makeLandmark(CatLandmarkType.mouthCornerRight, x: 55.0, y: 77.0);
      final face = CatFace(
        boundingBox: BoundingBox.ltrb(0, 0, 100, 100),
        landmarks: [landmark],
      );
      expect(face.landmarks.length, 1);
      expect(face.hasLandmarks, true);
      final found = face.getLandmark(CatLandmarkType.mouthCornerRight);
      expect(found, isNotNull);
      expect(found!.x, 55.0);
      expect(found.y, 77.0);
    });

    test('getLandmark finds all types when all are present', () {
      final landmarks =
          CatLandmarkType.values.map((type) => makeLandmark(type)).toList();
      final face = CatFace(
        boundingBox: BoundingBox.ltrb(0, 0, 100, 100),
        landmarks: landmarks,
      );

      for (final type in CatLandmarkType.values) {
        final lm = face.getLandmark(type);
        expect(lm, isNotNull, reason: 'getLandmark returned null for $type');
        expect(lm!.type, type);
      }
    });

    test('toMap serializes landmark type names as strings', () {
      final face = CatFace(
        boundingBox: BoundingBox.ltrb(0, 0, 100, 100),
        landmarks: [
          CatLandmark(type: CatLandmarkType.leftEar3, x: 10.0, y: 20.0),
        ],
      );
      final map = face.toMap();
      final landmarksList = map['landmarks'] as List;
      final firstLm = landmarksList[0] as Map<String, dynamic>;
      expect(firstLm['type'], 'leftEar3');
    });
  });

  // ---------------------------------------------------------------------------
  // Cat class
  // ---------------------------------------------------------------------------
  group('Cat', () {
    CatFace makeFace() {
      return CatFace(
        boundingBox: BoundingBox.ltrb(50.0, 60.0, 150.0, 160.0),
        landmarks: [
          CatLandmark(
              type: CatLandmarkType.noseBridgeLeft, x: 100.0, y: 110.0),
          CatLandmark(type: CatLandmarkType.leftEyeOuter, x: 80.0, y: 90.0),
        ],
      );
    }

    AnimalPose makePose() {
      return AnimalPose(landmarks: [
        AnimalPoseLandmark(
          type: AnimalPoseLandmarkType.neckBase,
          x: 100.0,
          y: 50.0,
          confidence: 0.98,
        ),
        AnimalPoseLandmark(
          type: AnimalPoseLandmarkType.tailEnd,
          x: 300.0,
          y: 200.0,
          confidence: 0.85,
        ),
      ]);
    }

    Cat makeFullCat() {
      return Cat(
        boundingBox: BoundingBox.ltrb(10.0, 20.0, 400.0, 350.0),
        score: 0.95,
        species: 'cat',
        breed: 'tabby',
        speciesConfidence: 0.92,
        face: makeFace(),
        pose: makePose(),
        imageWidth: 640,
        imageHeight: 480,
      );
    }

    Cat makeMinimalCat() {
      return Cat(
        boundingBox: BoundingBox.ltrb(5.0, 10.0, 200.0, 180.0),
        score: 0.75,
        imageWidth: 1920,
        imageHeight: 1080,
      );
    }

    test('constructor stores all required fields', () {
      final cat = makeFullCat();
      expect(cat.boundingBox.left, 10.0);
      expect(cat.boundingBox.top, 20.0);
      expect(cat.boundingBox.right, 400.0);
      expect(cat.boundingBox.bottom, 350.0);
      expect(cat.score, 0.95);
      expect(cat.imageWidth, 640);
      expect(cat.imageHeight, 480);
    });

    test('constructor stores all optional fields', () {
      final cat = makeFullCat();
      expect(cat.species, 'cat');
      expect(cat.breed, 'tabby');
      expect(cat.speciesConfidence, 0.92);
      expect(cat.face, isNotNull);
      expect(cat.pose, isNotNull);
    });

    test('optional fields default to null', () {
      final cat = makeMinimalCat();
      expect(cat.species, isNull);
      expect(cat.breed, isNull);
      expect(cat.speciesConfidence, isNull);
      expect(cat.face, isNull);
      expect(cat.pose, isNull);
    });

    test('toMap produces correct keys for full cat', () {
      final map = makeFullCat().toMap();
      expect(map.containsKey('boundingBox'), true);
      expect(map.containsKey('score'), true);
      expect(map.containsKey('species'), true);
      expect(map.containsKey('breed'), true);
      expect(map.containsKey('speciesConfidence'), true);
      expect(map.containsKey('face'), true);
      expect(map.containsKey('pose'), true);
      expect(map.containsKey('imageWidth'), true);
      expect(map.containsKey('imageHeight'), true);
    });

    test('toMap serializes boundingBox correctly', () {
      final map = makeFullCat().toMap();
      final bbox = map['boundingBox'] as Map<String, dynamic>;
      expect(bbox['left'], 10.0);
      expect(bbox['top'], 20.0);
      expect(bbox['right'], 400.0);
      expect(bbox['bottom'], 350.0);
    });

    test('toMap serializes scalar fields correctly', () {
      final map = makeFullCat().toMap();
      expect(map['score'], 0.95);
      expect(map['species'], 'cat');
      expect(map['breed'], 'tabby');
      expect(map['speciesConfidence'], 0.92);
      expect(map['imageWidth'], 640);
      expect(map['imageHeight'], 480);
    });

    test('toMap serializes face when present', () {
      final map = makeFullCat().toMap();
      expect(map['face'], isNotNull);
      final faceMap = map['face'] as Map<String, dynamic>;
      expect(faceMap.containsKey('boundingBox'), true);
      expect(faceMap.containsKey('landmarks'), true);
      final landmarks = faceMap['landmarks'] as List;
      expect(landmarks.length, 2);
    });

    test('toMap serializes pose when present', () {
      final map = makeFullCat().toMap();
      expect(map['pose'], isNotNull);
      final poseMap = map['pose'] as Map<String, dynamic>;
      expect(poseMap.containsKey('landmarks'), true);
      final landmarks = poseMap['landmarks'] as List;
      expect(landmarks.length, 2);
    });

    test('toMap has null face and pose when absent', () {
      final map = makeMinimalCat().toMap();
      expect(map['face'], isNull);
      expect(map['pose'], isNull);
      expect(map['species'], isNull);
      expect(map['breed'], isNull);
      expect(map['speciesConfidence'], isNull);
    });

    test('fromMap reconstructs full cat correctly', () {
      final map = {
        'boundingBox': {
          'left': 10.0,
          'top': 20.0,
          'right': 400.0,
          'bottom': 350.0,
        },
        'score': 0.95,
        'species': 'cat',
        'breed': 'tabby',
        'speciesConfidence': 0.92,
        'face': {
          'boundingBox': {
            'left': 50.0,
            'top': 60.0,
            'right': 150.0,
            'bottom': 160.0,
          },
          'landmarks': [
            {'type': 'noseBridgeLeft', 'x': 100.0, 'y': 110.0},
          ],
        },
        'pose': {
          'landmarks': [
            {
              'type': 'neckBase',
              'x': 100.0,
              'y': 50.0,
              'confidence': 0.98,
            },
          ],
        },
        'imageWidth': 640,
        'imageHeight': 480,
      };
      final cat = Cat.fromMap(map);
      expect(cat.boundingBox.left, 10.0);
      expect(cat.boundingBox.top, 20.0);
      expect(cat.boundingBox.right, 400.0);
      expect(cat.boundingBox.bottom, 350.0);
      expect(cat.score, 0.95);
      expect(cat.species, 'cat');
      expect(cat.breed, 'tabby');
      expect(cat.speciesConfidence, 0.92);
      expect(cat.face, isNotNull);
      expect(cat.face!.landmarks.length, 1);
      expect(cat.face!.landmarks[0].type, CatLandmarkType.noseBridgeLeft);
      expect(cat.pose, isNotNull);
      expect(cat.pose!.landmarks.length, 1);
      expect(cat.pose!.landmarks[0].type, AnimalPoseLandmarkType.neckBase);
      expect(cat.imageWidth, 640);
      expect(cat.imageHeight, 480);
    });

    test('fromMap reconstructs minimal cat (nulls) correctly', () {
      final map = {
        'boundingBox': {
          'left': 5.0,
          'top': 10.0,
          'right': 200.0,
          'bottom': 180.0,
        },
        'score': 0.75,
        'species': null,
        'breed': null,
        'speciesConfidence': null,
        'face': null,
        'pose': null,
        'imageWidth': 1920,
        'imageHeight': 1080,
      };
      final cat = Cat.fromMap(map);
      expect(cat.score, 0.75);
      expect(cat.species, isNull);
      expect(cat.breed, isNull);
      expect(cat.speciesConfidence, isNull);
      expect(cat.face, isNull);
      expect(cat.pose, isNull);
      expect(cat.imageWidth, 1920);
      expect(cat.imageHeight, 1080);
    });

    test('fromMap handles integer values for doubles', () {
      final map = {
        'boundingBox': {'left': 0, 'top': 0, 'right': 100, 'bottom': 100},
        'score': 1,
        'species': null,
        'breed': null,
        'speciesConfidence': null,
        'face': null,
        'pose': null,
        'imageWidth': 640,
        'imageHeight': 480,
      };
      final cat = Cat.fromMap(map);
      expect(cat.boundingBox.left, 0.0);
      expect(cat.score, 1.0);
    });

    test('toMap/fromMap round-trip with full cat', () {
      final original = makeFullCat();
      final restored = Cat.fromMap(original.toMap());

      expect(restored.boundingBox.left, original.boundingBox.left);
      expect(restored.boundingBox.top, original.boundingBox.top);
      expect(restored.boundingBox.right, original.boundingBox.right);
      expect(restored.boundingBox.bottom, original.boundingBox.bottom);
      expect(restored.score, original.score);
      expect(restored.species, original.species);
      expect(restored.breed, original.breed);
      expect(restored.speciesConfidence, original.speciesConfidence);
      expect(restored.imageWidth, original.imageWidth);
      expect(restored.imageHeight, original.imageHeight);

      // Face round-trip
      expect(restored.face, isNotNull);
      expect(restored.face!.boundingBox.left, original.face!.boundingBox.left);
      expect(restored.face!.landmarks.length, original.face!.landmarks.length);
      expect(
          restored.face!.landmarks[0].type, original.face!.landmarks[0].type);
      expect(restored.face!.landmarks[0].x, original.face!.landmarks[0].x);
      expect(restored.face!.landmarks[0].y, original.face!.landmarks[0].y);

      // Pose round-trip
      expect(restored.pose, isNotNull);
      expect(restored.pose!.landmarks.length, original.pose!.landmarks.length);
      expect(
          restored.pose!.landmarks[0].type, original.pose!.landmarks[0].type);
      expect(restored.pose!.landmarks[0].x, original.pose!.landmarks[0].x);
      expect(restored.pose!.landmarks[0].y, original.pose!.landmarks[0].y);
      expect(restored.pose!.landmarks[0].confidence,
          original.pose!.landmarks[0].confidence);
    });

    test('toMap/fromMap round-trip with minimal cat', () {
      final original = makeMinimalCat();
      final restored = Cat.fromMap(original.toMap());

      expect(restored.boundingBox.left, original.boundingBox.left);
      expect(restored.boundingBox.top, original.boundingBox.top);
      expect(restored.boundingBox.right, original.boundingBox.right);
      expect(restored.boundingBox.bottom, original.boundingBox.bottom);
      expect(restored.score, original.score);
      expect(restored.species, isNull);
      expect(restored.breed, isNull);
      expect(restored.speciesConfidence, isNull);
      expect(restored.face, isNull);
      expect(restored.pose, isNull);
      expect(restored.imageWidth, original.imageWidth);
      expect(restored.imageHeight, original.imageHeight);
    });

    test('toMap/fromMap round-trip with face but no pose', () {
      final original = Cat(
        boundingBox: BoundingBox.ltrb(10.0, 20.0, 300.0, 250.0),
        score: 0.88,
        species: 'cat',
        breed: 'persian',
        speciesConfidence: 0.80,
        face: makeFace(),
        imageWidth: 800,
        imageHeight: 600,
      );
      final restored = Cat.fromMap(original.toMap());
      expect(restored.face, isNotNull);
      expect(restored.pose, isNull);
      expect(restored.species, 'cat');
      expect(restored.breed, 'persian');
    });

    test('toMap/fromMap round-trip with pose but no face', () {
      final original = Cat(
        boundingBox: BoundingBox.ltrb(10.0, 20.0, 300.0, 250.0),
        score: 0.88,
        pose: makePose(),
        imageWidth: 800,
        imageHeight: 600,
      );
      final restored = Cat.fromMap(original.toMap());
      expect(restored.face, isNull);
      expect(restored.pose, isNotNull);
      expect(restored.pose!.landmarks.length, 2);
    });

    test('toString does not crash', () {
      expect(() => makeFullCat().toString(), returnsNormally);
      expect(() => makeMinimalCat().toString(), returnsNormally);
    });

    test('toString contains Cat prefix', () {
      expect(makeFullCat().toString(), contains('Cat('));
      expect(makeMinimalCat().toString(), contains('Cat('));
    });

    test('toString contains score', () {
      final str = makeFullCat().toString();
      expect(str, contains('score=0.950'));
    });

    test('toString reflects species and breed', () {
      final str = makeFullCat().toString();
      expect(str, contains('species=cat'));
      expect(str, contains('breed=tabby'));
    });

    test('toString reflects null species and breed', () {
      final str = makeMinimalCat().toString();
      expect(str, contains('species=null'));
      expect(str, contains('breed=null'));
    });

    test('toString reflects face and pose presence', () {
      final fullStr = makeFullCat().toString();
      expect(fullStr, contains('face=true'));
      expect(fullStr, contains('pose=true'));

      final minStr = makeMinimalCat().toString();
      expect(minStr, contains('face=false'));
      expect(minStr, contains('pose=false'));
    });

    test('edge case: score of 0.0', () {
      final cat = Cat(
        boundingBox: BoundingBox.ltrb(0, 0, 100, 100),
        score: 0.0,
        imageWidth: 640,
        imageHeight: 480,
      );
      expect(cat.score, 0.0);
      final restored = Cat.fromMap(cat.toMap());
      expect(restored.score, 0.0);
    });

    test('edge case: score of 1.0', () {
      final cat = Cat(
        boundingBox: BoundingBox.ltrb(0, 0, 100, 100),
        score: 1.0,
        imageWidth: 640,
        imageHeight: 480,
      );
      expect(cat.score, 1.0);
      final restored = Cat.fromMap(cat.toMap());
      expect(restored.score, 1.0);
    });

    test('edge case: empty species and breed strings', () {
      final cat = Cat(
        boundingBox: BoundingBox.ltrb(0, 0, 100, 100),
        score: 0.5,
        species: '',
        breed: '',
        imageWidth: 640,
        imageHeight: 480,
      );
      expect(cat.species, '');
      expect(cat.breed, '');
      final restored = Cat.fromMap(cat.toMap());
      expect(restored.species, '');
      expect(restored.breed, '');
    });

    test('edge case: zero-size bounding box', () {
      final cat = Cat(
        boundingBox: BoundingBox.ltrb(50.0, 50.0, 50.0, 50.0),
        score: 0.5,
        imageWidth: 640,
        imageHeight: 480,
      );
      final restored = Cat.fromMap(cat.toMap());
      expect(restored.boundingBox.left, restored.boundingBox.right);
      expect(restored.boundingBox.top, restored.boundingBox.bottom);
    });

    test('edge case: very small speciesConfidence', () {
      final cat = Cat(
        boundingBox: BoundingBox.ltrb(0, 0, 100, 100),
        score: 0.5,
        speciesConfidence: 0.001,
        imageWidth: 640,
        imageHeight: 480,
      );
      final restored = Cat.fromMap(cat.toMap());
      expect(restored.speciesConfidence, closeTo(0.001, 0.0001));
    });

    test('edge case: large image dimensions', () {
      final cat = Cat(
        boundingBox: BoundingBox.ltrb(0, 0, 7680, 4320),
        score: 0.99,
        imageWidth: 7680,
        imageHeight: 4320,
      );
      final restored = Cat.fromMap(cat.toMap());
      expect(restored.imageWidth, 7680);
      expect(restored.imageHeight, 4320);
    });

    test('edge case: face with empty landmarks list', () {
      final cat = Cat(
        boundingBox: BoundingBox.ltrb(0, 0, 100, 100),
        score: 0.5,
        face: CatFace(
          boundingBox: BoundingBox.ltrb(10, 10, 90, 90),
          landmarks: [],
        ),
        imageWidth: 640,
        imageHeight: 480,
      );
      expect(cat.face, isNotNull);
      expect(cat.face!.landmarks, isEmpty);
      final restored = Cat.fromMap(cat.toMap());
      expect(restored.face, isNotNull);
      expect(restored.face!.landmarks, isEmpty);
    });

    test('fromMap with integer speciesConfidence', () {
      final map = {
        'boundingBox': {'left': 0.0, 'top': 0.0, 'right': 100.0, 'bottom': 100.0},
        'score': 0.8,
        'species': 'cat',
        'breed': null,
        'speciesConfidence': 1,
        'face': null,
        'pose': null,
        'imageWidth': 640,
        'imageHeight': 480,
      };
      final cat = Cat.fromMap(map);
      expect(cat.speciesConfidence, 1.0);
    });
  });

  // ---------------------------------------------------------------------------
  // catLandmarkConnections constant
  // ---------------------------------------------------------------------------
  group('catLandmarkConnections constant', () {
    test('is non-empty list', () {
      expect(catLandmarkConnections, isNotEmpty);
    });

    test('each connection has exactly 2 elements', () {
      for (final connection in catLandmarkConnections) {
        expect(
          connection.length,
          2,
          reason: 'Connection does not have 2 elements: $connection',
        );
      }
    });

    test('all elements are valid CatLandmarkType values', () {
      final allTypes = CatLandmarkType.values.toSet();
      for (final connection in catLandmarkConnections) {
        expect(
          allTypes.contains(connection[0]),
          true,
          reason: 'Invalid start: ${connection[0]}',
        );
        expect(
          allTypes.contains(connection[1]),
          true,
          reason: 'Invalid end: ${connection[1]}',
        );
      }
    });

    test('right ear chain has 4 connections', () {
      final rightEarTypes = {
        CatLandmarkType.rightEar0,
        CatLandmarkType.rightEar1,
        CatLandmarkType.rightEar2,
        CatLandmarkType.rightEar3,
        CatLandmarkType.rightEar4,
      };
      final rightEarConnections = catLandmarkConnections
          .where((c) =>
              rightEarTypes.contains(c[0]) && rightEarTypes.contains(c[1]))
          .toList();
      expect(rightEarConnections.length, 4);
    });

    test('left ear chain has 4 connections', () {
      final leftEarTypes = {
        CatLandmarkType.leftEar0,
        CatLandmarkType.leftEar1,
        CatLandmarkType.leftEar2,
        CatLandmarkType.leftEar3,
        CatLandmarkType.leftEar4,
      };
      final leftEarConnections = catLandmarkConnections
          .where((c) =>
              leftEarTypes.contains(c[0]) && leftEarTypes.contains(c[1]))
          .toList();
      expect(leftEarConnections.length, 4);
    });

    test('right eye forms a loop (4 connections)', () {
      final rightEyeTypes = {
        CatLandmarkType.rightEyeOuter,
        CatLandmarkType.rightEyeTop,
        CatLandmarkType.rightEyeInner,
        CatLandmarkType.rightEyeBottom,
      };
      final rightEyeConnections = catLandmarkConnections
          .where((c) =>
              rightEyeTypes.contains(c[0]) && rightEyeTypes.contains(c[1]))
          .toList();
      expect(rightEyeConnections.length, 4);
    });

    test('left eye forms a loop (4 connections)', () {
      final leftEyeTypes = {
        CatLandmarkType.leftEyeOuter,
        CatLandmarkType.leftEyeTop,
        CatLandmarkType.leftEyeInner,
        CatLandmarkType.leftEyeBottom,
      };
      final leftEyeConnections = catLandmarkConnections
          .where((c) =>
              leftEyeTypes.contains(c[0]) && leftEyeTypes.contains(c[1]))
          .toList();
      expect(leftEyeConnections.length, 4);
    });

    test('nose bridge connections are present', () {
      final found = catLandmarkConnections.where((c) =>
          (c[0] == CatLandmarkType.noseBridgeLeft ||
              c[0] == CatLandmarkType.noseBridgeRight) ||
          (c[1] == CatLandmarkType.noseBridgeLeft ||
              c[1] == CatLandmarkType.noseBridgeRight));
      expect(found.length, greaterThanOrEqualTo(2));
    });

    test('nose ring connections are present', () {
      final noseRingTypes = {
        CatLandmarkType.noseRingLeft0,
        CatLandmarkType.noseRingLeft1,
        CatLandmarkType.noseRingRight0,
        CatLandmarkType.noseRingRight1,
      };
      final noseRingConnections = catLandmarkConnections.where((c) =>
          noseRingTypes.contains(c[0]) || noseRingTypes.contains(c[1]));
      expect(noseRingConnections.length, greaterThanOrEqualTo(3));
    });

    test('mouth connections include mouthTop and mouthBottom', () {
      final mouthTypes = {
        CatLandmarkType.mouthTop,
        CatLandmarkType.mouthBottom,
        CatLandmarkType.mouthCornerLeft,
        CatLandmarkType.mouthCornerRight,
      };
      final mouthConnections = catLandmarkConnections
          .where((c) =>
              mouthTypes.contains(c[0]) && mouthTypes.contains(c[1]))
          .toList();
      expect(mouthConnections.length, greaterThanOrEqualTo(4));
    });

    test('nose tip connections include noseTipLeft and noseTipRight', () {
      final found = catLandmarkConnections.where((c) =>
          c[0] == CatLandmarkType.noseTipLeft ||
          c[1] == CatLandmarkType.noseTipLeft ||
          c[0] == CatLandmarkType.noseTipRight ||
          c[1] == CatLandmarkType.noseTipRight);
      expect(found.length, greaterThanOrEqualTo(2));
    });

    test('total connection count is 28', () {
      // 4 (right ear) + 4 (left ear) + 4 (right eye) + 4 (left eye)
      // + 2 (nose bridge) + 3 (nose ring) + 3 (nose tips/wings)
      // + 4 (mouth) = 28
      expect(catLandmarkConnections.length, 28);
    });

    test('no connection references the same type twice', () {
      for (final connection in catLandmarkConnections) {
        expect(
          connection[0] == connection[1],
          false,
          reason: 'Self-loop found for ${connection[0]}',
        );
      }
    });
  });
}
