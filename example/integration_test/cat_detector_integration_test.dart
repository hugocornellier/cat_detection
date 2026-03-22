// Comprehensive integration tests for CatDetector.
//
// These tests cover:
// - Initialization and disposal
// - Detection from Mat (detectFromMat)
// - detect() bytes API
// - poseOnly mode
// - Landmark validation (48 landmarks, all types present, finite coordinates)
// - Error recovery after empty-result input
// - Result consistency / determinism
// - Configuration parameters (cropMargin, PerformanceConfig)
// - CatDetectorIsolate (spawn, detect, detectFromMat, re-spawn)
//
// Run with:
//   cd example && flutter test integration_test/
// or via a connected device/simulator.

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:cat_detection/cat_detection.dart';

/// Path to the test cat image bundled in example/integration_test/test_images/.
const _catImagePath = 'integration_test/test_images/cat.jpg';

/// Minimal valid 1x1 black PNG (used for error-recovery tests).
class _TestUtils {
  static Uint8List createTinyBlackPng() {
    return Uint8List.fromList([
      0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, // PNG signature
      0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52, // IHDR chunk
      0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, // Width: 1, Height: 1
      0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4, // Bit depth, color type
      0x89, 0x00, 0x00, 0x00, 0x0A, 0x49, 0x44, 0x41, // IDAT chunk
      0x54, 0x78, 0x9C, 0x63, 0x00, 0x01, 0x00, 0x00,
      0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00, // Image data
      0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, // IEND chunk
      0x42, 0x60, 0x82,
    ]);
  }
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  // ---------------------------------------------------------------------------
  // 1. CatDetector - Initialization
  // ---------------------------------------------------------------------------

  group('CatDetector - Initialization', () {
    testWidgets('should initialize successfully', (tester) async {
      final detector = CatDetector();
      await detector.initialize();
      expect(detector.isInitialized, true);
      await detector.dispose();
    });

    testWidgets('should report isInitialized as true after init',
        (tester) async {
      final detector = CatDetector();
      expect(detector.isInitialized, false);
      await detector.initialize();
      expect(detector.isInitialized, true);
      await detector.dispose();
    });

    testWidgets('should report isInitialized as false before init',
        (tester) async {
      final detector = CatDetector();
      expect(detector.isInitialized, false);
    });

    testWidgets('should throw StateError when detect called before init',
        (tester) async {
      final detector = CatDetector();
      final bytes = _TestUtils.createTinyBlackPng();

      expect(
        () => detector.detect(bytes),
        throwsA(isA<StateError>().having(
          (e) => e.message,
          'message',
          contains('not initialized'),
        )),
      );
    });

    testWidgets('should throw StateError when detectFromMat called before init',
        (tester) async {
      final detector = CatDetector();
      final mat = cv.Mat.zeros(100, 100, cv.MatType.CV_8UC3);

      try {
        expect(
          () => detector.detectFromMat(
            mat,
            imageWidth: mat.cols,
            imageHeight: mat.rows,
          ),
          throwsA(isA<StateError>().having(
            (e) => e.message,
            'message',
            contains('not initialized'),
          )),
        );
      } finally {
        mat.dispose();
      }
    });

    testWidgets('should allow re-initialization', (tester) async {
      final detector = CatDetector();
      await detector.initialize();
      expect(detector.isInitialized, true);

      await detector.initialize();
      expect(detector.isInitialized, true);

      await detector.dispose();
    });

    testWidgets('should handle multiple dispose calls', (tester) async {
      final detector = CatDetector();
      await detector.initialize();
      await detector.dispose();
      expect(detector.isInitialized, false);

      // Second dispose should not throw.
      await detector.dispose();
      expect(detector.isInitialized, false);
    });
  });

  // ---------------------------------------------------------------------------
  // 2. CatDetector - Detection from Mat
  // ---------------------------------------------------------------------------

  group('CatDetector - Detection from Mat', () {
    testWidgets('should detect cat from test image', (tester) async {
      final detector = CatDetector();
      await detector.initialize();

      final ByteData data = await rootBundle.load(_catImagePath);
      final Uint8List bytes = data.buffer.asUint8List();
      final mat = cv.imdecode(bytes, cv.IMREAD_COLOR);
      expect(mat.isEmpty, isFalse);

      try {
        final List<Cat> results = await detector.detectFromMat(
          mat,
          imageWidth: mat.cols,
          imageHeight: mat.rows,
        );

        debugPrint('detectFromMat: ${results.length} cat(s) detected');
        for (final cat in results) {
          debugPrint('  species=${cat.species}, score=${cat.score}');
          debugPrint('  bbox=${cat.boundingBox}');
          debugPrint('  face=${cat.face != null}');
          if (cat.face != null) {
            debugPrint('  face bbox=${cat.face!.boundingBox}');
            debugPrint('  landmarks=${cat.face!.landmarks.length}');
          }
        }

        expect(results, isNotEmpty, reason: 'No cats detected in test image');
      } finally {
        mat.dispose();
      }

      await detector.dispose();
    });

    testWidgets('should return valid bounding box', (tester) async {
      final detector = CatDetector();
      await detector.initialize();

      final ByteData data = await rootBundle.load(_catImagePath);
      final Uint8List bytes = data.buffer.asUint8List();
      final mat = cv.imdecode(bytes, cv.IMREAD_COLOR);

      try {
        final List<Cat> results = await detector.detectFromMat(
          mat,
          imageWidth: mat.cols,
          imageHeight: mat.rows,
        );
        expect(results, isNotEmpty);

        final cat = results.first;
        expect(cat.boundingBox.right, greaterThan(cat.boundingBox.left));
        expect(cat.boundingBox.bottom, greaterThan(cat.boundingBox.top));
        expect(cat.boundingBox.left, greaterThanOrEqualTo(0));
        expect(cat.boundingBox.top, greaterThanOrEqualTo(0));
      } finally {
        mat.dispose();
      }

      await detector.dispose();
    });

    testWidgets('should have correct imageWidth and imageHeight',
        (tester) async {
      final detector = CatDetector();
      await detector.initialize();

      final ByteData data = await rootBundle.load(_catImagePath);
      final Uint8List bytes = data.buffer.asUint8List();
      final mat = cv.imdecode(bytes, cv.IMREAD_COLOR);

      try {
        final List<Cat> results = await detector.detectFromMat(
          mat,
          imageWidth: mat.cols,
          imageHeight: mat.rows,
        );
        expect(results, isNotEmpty);

        final cat = results.first;
        expect(cat.imageWidth, mat.cols);
        expect(cat.imageHeight, mat.rows);
      } finally {
        mat.dispose();
      }

      await detector.dispose();
    });

    testWidgets('should detect face with landmarks when mode is full',
        (tester) async {
      final detector = CatDetector(
        mode: CatDetectionMode.full,
      );
      await detector.initialize();

      final ByteData data = await rootBundle.load(_catImagePath);
      final Uint8List bytes = data.buffer.asUint8List();
      final mat = cv.imdecode(bytes, cv.IMREAD_COLOR);

      try {
        final List<Cat> results = await detector.detectFromMat(
          mat,
          imageWidth: mat.cols,
          imageHeight: mat.rows,
        );

        debugPrint('Full mode: ${results.length} cat(s)');
        for (final cat in results) {
          debugPrint('  face=${cat.face != null}');
          if (cat.face != null) {
            debugPrint('  hasLandmarks=${cat.face!.hasLandmarks}');
            debugPrint('  landmark count=${cat.face!.landmarks.length}');
          }
        }

        expect(results, isNotEmpty);
        expect(results.first.face, isNotNull,
            reason: 'Face should not be null in full mode');
        expect(results.first.face!.hasLandmarks, true,
            reason: 'Face should have landmarks in full mode');
      } finally {
        mat.dispose();
      }

      await detector.dispose();
    });

    testWidgets('should return 48 landmarks', (tester) async {
      final detector = CatDetector(
        mode: CatDetectionMode.full,
      );
      await detector.initialize();

      final ByteData data = await rootBundle.load(_catImagePath);
      final Uint8List bytes = data.buffer.asUint8List();
      final mat = cv.imdecode(bytes, cv.IMREAD_COLOR);

      try {
        final List<Cat> results = await detector.detectFromMat(
          mat,
          imageWidth: mat.cols,
          imageHeight: mat.rows,
        );
        expect(results, isNotEmpty);
        expect(results.first.face, isNotNull);
        expect(results.first.face!.landmarks.length, numCatLandmarks);
      } finally {
        mat.dispose();
      }

      await detector.dispose();
    });

    testWidgets('should have all landmark types present', (tester) async {
      final detector = CatDetector(
        mode: CatDetectionMode.full,
      );
      await detector.initialize();

      final ByteData data = await rootBundle.load(_catImagePath);
      final Uint8List bytes = data.buffer.asUint8List();
      final mat = cv.imdecode(bytes, cv.IMREAD_COLOR);

      try {
        final List<Cat> results = await detector.detectFromMat(
          mat,
          imageWidth: mat.cols,
          imageHeight: mat.rows,
        );
        expect(results, isNotEmpty);

        final face = results.first.face!;
        for (final type in CatLandmarkType.values) {
          final landmark = face.getLandmark(type);
          expect(landmark, isNotNull, reason: 'Missing landmark: $type');
          expect(landmark!.type, type);
        }
      } finally {
        mat.dispose();
      }

      await detector.dispose();
    });
  });

  // ---------------------------------------------------------------------------
  // 3. CatDetector - detect() bytes API
  // ---------------------------------------------------------------------------

  group('CatDetector - detect() bytes API', () {
    testWidgets('should detect cats from image bytes', (tester) async {
      final detector = CatDetector();
      await detector.initialize();

      final ByteData data = await rootBundle.load(_catImagePath);
      final Uint8List bytes = data.buffer.asUint8List();

      final List<Cat> results = await detector.detect(bytes);

      expect(results, isNotEmpty);
      expect(results.first.boundingBox.right,
          greaterThan(results.first.boundingBox.left));
      expect(results.first.imageWidth, greaterThan(0));
      expect(results.first.imageHeight, greaterThan(0));

      await detector.dispose();
    });

    testWidgets('should produce matching results to detectFromMat',
        (tester) async {
      final detector = CatDetector();
      await detector.initialize();

      final ByteData data = await rootBundle.load(_catImagePath);
      final Uint8List bytes = data.buffer.asUint8List();
      final mat = cv.imdecode(bytes, cv.IMREAD_COLOR);
      expect(mat.isEmpty, isFalse);

      try {
        final List<Cat> fromBytes = await detector.detect(bytes);
        final List<Cat> fromMat = await detector.detectFromMat(
          mat,
          imageWidth: mat.cols,
          imageHeight: mat.rows,
        );

        expect(fromBytes.length, fromMat.length);

        for (int i = 0; i < fromBytes.length; i++) {
          expect(fromBytes[i].face?.landmarks.length,
              fromMat[i].face?.landmarks.length);
        }
      } finally {
        mat.dispose();
      }

      await detector.dispose();
    });

    testWidgets('should return empty list for invalid bytes', (tester) async {
      final detector = CatDetector();
      await detector.initialize();

      final invalidBytes = Uint8List.fromList([0, 1, 2, 3, 4, 5]);
      final results = await detector.detect(invalidBytes);

      expect(results, isEmpty);

      await detector.dispose();
    });
  });

  // ---------------------------------------------------------------------------
  // 4. CatDetector - poseOnly Mode
  // ---------------------------------------------------------------------------

  group('CatDetector - poseOnly Mode', () {
    testWidgets('should return cat with no face in poseOnly mode',
        (tester) async {
      final detector = CatDetector(mode: CatDetectionMode.poseOnly);
      await detector.initialize();

      final ByteData data = await rootBundle.load(_catImagePath);
      final Uint8List bytes = data.buffer.asUint8List();
      final mat = cv.imdecode(bytes, cv.IMREAD_COLOR);

      try {
        final List<Cat> results = await detector.detectFromMat(
          mat,
          imageWidth: mat.cols,
          imageHeight: mat.rows,
        );
        expect(results, isNotEmpty);

        final cat = results.first;
        expect(cat.face, isNull);
      } finally {
        mat.dispose();
      }

      await detector.dispose();
    });

    testWidgets('should still have valid bounding box in poseOnly mode',
        (tester) async {
      final detector = CatDetector(mode: CatDetectionMode.poseOnly);
      await detector.initialize();

      final ByteData data = await rootBundle.load(_catImagePath);
      final Uint8List bytes = data.buffer.asUint8List();
      final mat = cv.imdecode(bytes, cv.IMREAD_COLOR);

      try {
        final List<Cat> results = await detector.detectFromMat(
          mat,
          imageWidth: mat.cols,
          imageHeight: mat.rows,
        );
        expect(results, isNotEmpty);

        final bbox = results.first.boundingBox;
        expect(bbox.right, greaterThan(bbox.left));
        expect(bbox.bottom, greaterThan(bbox.top));
        expect(bbox.left, greaterThanOrEqualTo(0));
        expect(bbox.top, greaterThanOrEqualTo(0));
      } finally {
        mat.dispose();
      }

      await detector.dispose();
    });

    testWidgets('should have pose data in poseOnly mode', (tester) async {
      final detector = CatDetector(mode: CatDetectionMode.poseOnly);
      await detector.initialize();

      final ByteData data = await rootBundle.load(_catImagePath);
      final Uint8List bytes = data.buffer.asUint8List();

      final List<Cat> results = await detector.detect(bytes);
      expect(results, isNotEmpty);

      final cat = results.first;
      expect(cat.pose, isNotNull, reason: 'Pose should be present');
      expect(cat.pose!.landmarks, isNotEmpty,
          reason: 'Pose should have landmarks');

      await detector.dispose();
    });
  });

  // ---------------------------------------------------------------------------
  // 5. CatDetector - Error Recovery
  // ---------------------------------------------------------------------------

  group('CatDetector - Error Recovery', () {
    testWidgets('should recover after empty-result input (1x1 black image)',
        (tester) async {
      final detector = CatDetector();
      await detector.initialize();

      // A 1x1 black Mat is valid but produces no detections.
      final tiny = cv.Mat.zeros(1, 1, cv.MatType.CV_8UC3);
      final emptyResults = await detector.detectFromMat(
        tiny,
        imageWidth: 1,
        imageHeight: 1,
      );
      tiny.dispose();
      expect(emptyResults, isNotNull);

      // Should work normally after a no-detection run.
      final ByteData data = await rootBundle.load(_catImagePath);
      final Uint8List bytes = data.buffer.asUint8List();
      final mat = cv.imdecode(bytes, cv.IMREAD_COLOR);
      try {
        final List<Cat> results = await detector.detectFromMat(
          mat,
          imageWidth: mat.cols,
          imageHeight: mat.rows,
        );
        expect(results, isNotEmpty);
      } finally {
        mat.dispose();
      }

      await detector.dispose();
    });
  });

  // ---------------------------------------------------------------------------
  // 6. CatDetector - Result Consistency
  // ---------------------------------------------------------------------------

  group('CatDetector - Result Consistency', () {
    testWidgets('should produce deterministic results (same image twice)',
        (tester) async {
      final detector = CatDetector();
      await detector.initialize();

      final ByteData data = await rootBundle.load(_catImagePath);
      final Uint8List bytes = data.buffer.asUint8List();
      final mat1 = cv.imdecode(bytes, cv.IMREAD_COLOR);
      final mat2 = cv.imdecode(bytes, cv.IMREAD_COLOR);

      try {
        final List<Cat> first = await detector.detectFromMat(
          mat1,
          imageWidth: mat1.cols,
          imageHeight: mat1.rows,
        );
        final List<Cat> second = await detector.detectFromMat(
          mat2,
          imageWidth: mat2.cols,
          imageHeight: mat2.rows,
        );

        expect(first.length, second.length);

        for (int i = 0; i < first.length; i++) {
          expect(first[i].face?.landmarks.length,
              second[i].face?.landmarks.length);

          final firstLandmarks = first[i].face?.landmarks ?? [];
          final secondLandmarks = second[i].face?.landmarks ?? [];
          for (int j = 0; j < firstLandmarks.length; j++) {
            expect(
              firstLandmarks[j].x,
              closeTo(secondLandmarks[j].x, 1e-3),
              reason: 'Landmark x not deterministic at face=$i lm=$j',
            );
            expect(
              firstLandmarks[j].y,
              closeTo(secondLandmarks[j].y, 1e-3),
              reason: 'Landmark y not deterministic at face=$i lm=$j',
            );
          }
        }
      } finally {
        mat1.dispose();
        mat2.dispose();
      }

      await detector.dispose();
    });
  });

  // ---------------------------------------------------------------------------
  // 7. CatDetector - Configuration
  // ---------------------------------------------------------------------------

  group('CatDetector - Configuration', () {
    testWidgets('should respect cropMargin parameter', (tester) async {
      final detectorTight = CatDetector(cropMargin: 0.05);
      final detectorWide = CatDetector(cropMargin: 0.40);

      await detectorTight.initialize();
      await detectorWide.initialize();

      final ByteData data = await rootBundle.load(_catImagePath);
      final Uint8List bytes = data.buffer.asUint8List();

      // Both detectors should detect successfully with different margins.
      final tightResults = await detectorTight.detect(bytes);
      final wideResults = await detectorWide.detect(bytes);

      expect(tightResults, isNotEmpty,
          reason: 'Tight margin detector returned no results');
      expect(wideResults, isNotEmpty,
          reason: 'Wide margin detector returned no results');

      // Both should produce 48 landmarks.
      expect(tightResults.first.face, isNotNull);
      expect(wideResults.first.face, isNotNull);
      expect(tightResults.first.face!.landmarks.length, numCatLandmarks);
      expect(wideResults.first.face!.landmarks.length, numCatLandmarks);

      await detectorTight.dispose();
      await detectorWide.dispose();
    });

    testWidgets('should work with PerformanceConfig.disabled', (tester) async {
      final detector = CatDetector(
        performanceConfig: PerformanceConfig.disabled,
      );
      await detector.initialize();

      final ByteData data = await rootBundle.load(_catImagePath);
      final Uint8List bytes = data.buffer.asUint8List();
      final List<Cat> results = await detector.detect(bytes);

      expect(results, isNotEmpty);

      await detector.dispose();
    });

    testWidgets('should expose configured mode and landmarkModel',
        (tester) async {
      final detector = CatDetector(
        mode: CatDetectionMode.poseOnly,
        landmarkModel: CatLandmarkModel.full,
        cropMargin: 0.15,
      );
      await detector.initialize();

      expect(detector.mode, CatDetectionMode.poseOnly);
      expect(detector.landmarkModel, CatLandmarkModel.full);
      expect(detector.cropMargin, 0.15);

      await detector.dispose();
    });
  });

  // ---------------------------------------------------------------------------
  // 8. CatDetector - Landmark Validation
  // ---------------------------------------------------------------------------

  group('CatDetector - Landmark Validation', () {
    testWidgets('all landmarks should have finite x,y coordinates',
        (tester) async {
      final detector = CatDetector(
        mode: CatDetectionMode.full,
      );
      await detector.initialize();

      final ByteData data = await rootBundle.load(_catImagePath);
      final Uint8List bytes = data.buffer.asUint8List();
      final List<Cat> results = await detector.detect(bytes);

      expect(results, isNotEmpty);

      for (final cat in results) {
        for (final landmark in cat.face?.landmarks ?? []) {
          expect(landmark.x.isFinite, true,
              reason: 'x is not finite for ${landmark.type}');
          expect(landmark.y.isFinite, true,
              reason: 'y is not finite for ${landmark.type}');
        }
      }

      await detector.dispose();
    });

    testWidgets('landmarks should be within image bounds', (tester) async {
      final detector = CatDetector(
        mode: CatDetectionMode.full,
      );
      await detector.initialize();

      final ByteData data = await rootBundle.load(_catImagePath);
      final Uint8List bytes = data.buffer.asUint8List();
      final mat = cv.imdecode(bytes, cv.IMREAD_COLOR);

      try {
        final List<Cat> results = await detector.detectFromMat(
          mat,
          imageWidth: mat.cols,
          imageHeight: mat.rows,
        );
        expect(results, isNotEmpty);

        final cat = results.first;
        expect(cat.face, isNotNull, reason: 'Face should not be null');

        for (final landmark in cat.face!.landmarks) {
          expect(landmark.x, greaterThanOrEqualTo(0),
              reason: '${landmark.type}.x is negative: ${landmark.x}');
          expect(landmark.x, lessThanOrEqualTo(cat.imageWidth.toDouble()),
              reason:
                  '${landmark.type}.x exceeds imageWidth: ${landmark.x} > ${cat.imageWidth}');
          expect(landmark.y, greaterThanOrEqualTo(0),
              reason: '${landmark.type}.y is negative: ${landmark.y}');
          expect(landmark.y, lessThanOrEqualTo(cat.imageHeight.toDouble()),
              reason:
                  '${landmark.type}.y exceeds imageHeight: ${landmark.y} > ${cat.imageHeight}');
        }
      } finally {
        mat.dispose();
      }

      await detector.dispose();
    });

    testWidgets('normalized coordinates should be in 0.0-1.0 range',
        (tester) async {
      final detector = CatDetector(
        mode: CatDetectionMode.full,
      );
      await detector.initialize();

      final ByteData data = await rootBundle.load(_catImagePath);
      final Uint8List bytes = data.buffer.asUint8List();
      final List<Cat> results = await detector.detect(bytes);

      expect(results, isNotEmpty);

      final cat = results.first;
      expect(cat.face, isNotNull);

      for (final landmark in cat.face!.landmarks) {
        final xNorm = landmark.xNorm(cat.imageWidth);
        final yNorm = landmark.yNorm(cat.imageHeight);
        expect(xNorm, greaterThanOrEqualTo(0.0),
            reason: '${landmark.type}.xNorm < 0');
        expect(xNorm, lessThanOrEqualTo(1.0),
            reason: '${landmark.type}.xNorm > 1');
        expect(yNorm, greaterThanOrEqualTo(0.0),
            reason: '${landmark.type}.yNorm < 0');
        expect(yNorm, lessThanOrEqualTo(1.0),
            reason: '${landmark.type}.yNorm > 1');
      }

      await detector.dispose();
    });

    testWidgets('face bounding box should be valid', (tester) async {
      final detector = CatDetector(
        mode: CatDetectionMode.full,
      );
      await detector.initialize();

      final ByteData data = await rootBundle.load(_catImagePath);
      final Uint8List bytes = data.buffer.asUint8List();
      final List<Cat> results = await detector.detect(bytes);

      expect(results, isNotEmpty);

      final cat = results.first;
      expect(cat.face, isNotNull);

      final faceBbox = cat.face!.boundingBox;
      debugPrint('Face bbox: L=${faceBbox.left}, T=${faceBbox.top}, '
          'R=${faceBbox.right}, B=${faceBbox.bottom}');
      expect(faceBbox.right, greaterThan(faceBbox.left),
          reason: 'Face bbox width <= 0');
      expect(faceBbox.bottom, greaterThan(faceBbox.top),
          reason: 'Face bbox height <= 0');
    });
  });

  // ---------------------------------------------------------------------------
  // 9. CatDetectorIsolate
  // ---------------------------------------------------------------------------

  group('CatDetectorIsolate', () {
    testWidgets('should detect cats via isolate', (tester) async {
      final isolate = await CatDetectorIsolate.spawn(
        mode: CatDetectionMode.full,
      );
      expect(isolate.isReady, true);

      final ByteData data = await rootBundle.load(_catImagePath);
      final Uint8List bytes = data.buffer.asUint8List();
      final List<Cat> results = await isolate.detectCats(bytes);

      debugPrint('Isolate: ${results.length} cat(s) detected');
      for (final cat in results) {
        debugPrint('  species=${cat.species}, score=${cat.score}');
        debugPrint('  face=${cat.face != null}');
        if (cat.face != null) {
          debugPrint('  landmarks=${cat.face!.landmarks.length}');
        }
      }

      expect(results, isNotEmpty);

      final cat = results.first;
      expect(cat.boundingBox.right, greaterThan(cat.boundingBox.left));
      expect(cat.boundingBox.bottom, greaterThan(cat.boundingBox.top));
      expect(cat.imageWidth, greaterThan(0));
      expect(cat.imageHeight, greaterThan(0));

      await isolate.dispose();
    });

    testWidgets('should detect cats from Mat via isolate', (tester) async {
      final isolate = await CatDetectorIsolate.spawn(
        mode: CatDetectionMode.full,
      );

      final ByteData data = await rootBundle.load(_catImagePath);
      final Uint8List bytes = data.buffer.asUint8List();
      final mat = cv.imdecode(bytes, cv.IMREAD_COLOR);
      expect(mat.isEmpty, isFalse);

      try {
        final List<Cat> results = await isolate.detectCatsFromMat(mat);

        expect(results, isNotEmpty);

        final cat = results.first;
        expect(cat.boundingBox.right, greaterThan(cat.boundingBox.left));
        expect(cat.face, isNotNull);
        expect(cat.face!.landmarks.length, numCatLandmarks);
      } finally {
        mat.dispose();
      }

      await isolate.dispose();
    });

    testWidgets('should match main thread results', (tester) async {
      final detector = CatDetector();
      await detector.initialize();

      final isolate = await CatDetectorIsolate.spawn(
        mode: CatDetectionMode.full,
      );

      final ByteData data = await rootBundle.load(_catImagePath);
      final Uint8List bytes = data.buffer.asUint8List();

      final List<Cat> mainResults = await detector.detect(bytes);
      final List<Cat> isolateResults = await isolate.detectCats(bytes);

      expect(mainResults.length, isolateResults.length);

      for (int i = 0; i < mainResults.length; i++) {
        expect(mainResults[i].face?.landmarks.length,
            isolateResults[i].face?.landmarks.length,
            reason: 'Landmark count mismatch at index $i');
      }

      await detector.dispose();
      await isolate.dispose();
    });

    testWidgets('should support dispose and re-spawn', (tester) async {
      final first = await CatDetectorIsolate.spawn();
      expect(first.isReady, true);
      await first.dispose();
      expect(first.isReady, false);

      // Spawn a new isolate after the previous one was disposed.
      final second = await CatDetectorIsolate.spawn();
      expect(second.isReady, true);

      final ByteData data = await rootBundle.load(_catImagePath);
      final Uint8List bytes = data.buffer.asUint8List();
      final List<Cat> results = await second.detectCats(bytes);

      expect(results, isNotEmpty);

      await second.dispose();
    });

    testWidgets('should handle two sequential detectCats calls on same isolate',
        (tester) async {
      final isolate = await CatDetectorIsolate.spawn(
        mode: CatDetectionMode.full,
      );
      expect(isolate.isReady, true);

      final ByteData data = await rootBundle
          .load('packages/cat_detection/assets/samples/sample_cat_1.png');
      final Uint8List bytes = data.buffer.asUint8List();

      final List<Cat> first = await isolate.detectCats(bytes);
      expect(first, isNotEmpty);

      final List<Cat> second = await isolate.detectCats(bytes);
      expect(second, isNotEmpty);

      expect(first.length, second.length);

      await isolate.dispose();
    });

    testWidgets(
        'should handle three sequential detectCats calls on same isolate',
        (tester) async {
      final isolate = await CatDetectorIsolate.spawn(
        mode: CatDetectionMode.full,
      );
      expect(isolate.isReady, true);

      final ByteData data = await rootBundle
          .load('packages/cat_detection/assets/samples/sample_cat_1.png');
      final Uint8List bytes = data.buffer.asUint8List();

      final List<Cat> first = await isolate.detectCats(bytes);
      expect(first, isNotEmpty);

      final List<Cat> second = await isolate.detectCats(bytes);
      expect(second, isNotEmpty);

      final List<Cat> third = await isolate.detectCats(bytes);
      expect(third, isNotEmpty);

      expect(first.length, second.length);
      expect(second.length, third.length);

      await isolate.dispose();
    });

    testWidgets(
        'should handle two sequential detectCatsFromMat calls on same isolate',
        (tester) async {
      final isolate = await CatDetectorIsolate.spawn(
        mode: CatDetectionMode.full,
      );
      expect(isolate.isReady, true);

      final ByteData data = await rootBundle
          .load('packages/cat_detection/assets/samples/sample_cat_1.png');
      final Uint8List bytes = data.buffer.asUint8List();
      final mat = cv.imdecode(bytes, cv.IMREAD_COLOR);
      expect(mat.isEmpty, isFalse);

      try {
        final List<Cat> first = await isolate.detectCatsFromMat(mat);
        expect(first, isNotEmpty);

        final List<Cat> second = await isolate.detectCatsFromMat(mat);
        expect(second, isNotEmpty);

        expect(first.length, second.length);
      } finally {
        mat.dispose();
      }

      await isolate.dispose();
    });
  });

  // ---------------------------------------------------------------------------
  // 10. CatDetector - Dispose
  // ---------------------------------------------------------------------------

  group('CatDetector - Dispose', () {
    testWidgets('should dispose cleanly', (tester) async {
      final detector = CatDetector();
      await detector.initialize();
      expect(detector.isInitialized, true);

      await detector.dispose();
      expect(detector.isInitialized, false);
    });

    testWidgets('should not be usable after dispose', (tester) async {
      final detector = CatDetector();
      await detector.initialize();
      await detector.dispose();

      final bytes = _TestUtils.createTinyBlackPng();
      expect(
        () => detector.detect(bytes),
        throwsA(isA<StateError>().having(
          (e) => e.message,
          'message',
          contains('not initialized'),
        )),
      );
    });

    testWidgets('Cat.toString() should not crash', (tester) async {
      final detector = CatDetector();
      await detector.initialize();

      final ByteData data = await rootBundle.load(_catImagePath);
      final Uint8List bytes = data.buffer.asUint8List();
      final List<Cat> results = await detector.detect(bytes);

      expect(results, isNotEmpty);

      final catString = results.first.toString();
      expect(catString, isNotEmpty);
      expect(catString, contains('Cat('));
      expect(catString, contains('score='));

      await detector.dispose();
    });
  });
}
