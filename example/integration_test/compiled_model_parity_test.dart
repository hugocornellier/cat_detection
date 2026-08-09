import 'package:cat_detection/cat_detection.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

const _imagePath = 'integration_test/test_images/cat.jpg';

Future<List<Cat>> _detect({
  required String label,
  required bool useCompiledModel,
  Set<Accelerator> accelerators = const {
    Accelerator.gpu,
    Accelerator.cpu,
  },
}) async {
  final detector = CatDetector(mode: CatDetectionMode.full);
  await detector.initialize(
    useCompiledModel: useCompiledModel,
    accelerators: accelerators,
    precision: Precision.fp32,
  );
  try {
    final ByteData data = await rootBundle.load(_imagePath);
    final Uint8List bytes = data.buffer.asUint8List();
    final result = await detector.detect(bytes);
    debugPrint(
      '$label: ${result.length} cat(s), '
      'species=${result.isEmpty ? "-" : result.first.species}, '
      'faceLandmarks=${result.isEmpty ? 0 : result.first.face?.landmarks.length}, '
      'poseLandmarks=${result.isEmpty ? 0 : result.first.pose?.landmarks.length}',
    );
    return result;
  } finally {
    await detector.dispose();
  }
}

double _boxDelta(BoundingBox a, BoundingBox b) => [
      (a.left - b.left).abs(),
      (a.top - b.top).abs(),
      (a.right - b.right).abs(),
      (a.bottom - b.bottom).abs(),
    ].reduce((x, y) => x > y ? x : y);

void _expectParity(List<Cat> expected, List<Cat> actual, String label) {
  expect(actual.length, expected.length, reason: '$label animal count');
  for (var i = 0; i < expected.length; i++) {
    final a = expected[i];
    final b = actual[i];
    expect(b.species, a.species, reason: '$label species[$i]');
    expect(b.breed, a.breed, reason: '$label breed[$i]');
    expect(b.imageWidth, a.imageWidth, reason: '$label image width[$i]');
    expect(b.imageHeight, a.imageHeight, reason: '$label image height[$i]');
    expect(b.speciesConfidence, isNotNull);
    expect(a.speciesConfidence, isNotNull);
    expect(
      (b.speciesConfidence! - a.speciesConfidence!).abs(),
      lessThan(0.02),
      reason: '$label species confidence[$i]',
    );
    expect(
      (b.score - a.score).abs(),
      lessThan(0.02),
      reason: '$label body score[$i]',
    );
    expect(
      _boxDelta(a.boundingBox, b.boundingBox),
      lessThan(10.0),
      reason: '$label body box[$i]',
    );

    expect(b.pose, isNotNull, reason: '$label pose[$i]');
    final expectedPose = a.pose!.landmarks;
    final actualPose = b.pose!.landmarks;
    expect(actualPose.length, expectedPose.length, reason: '$label pose count');
    for (var j = 0; j < expectedPose.length; j++) {
      expect(actualPose[j].type, expectedPose[j].type);
      expect(
        (actualPose[j].confidence - expectedPose[j].confidence).abs(),
        lessThan(0.02),
        reason: '$label pose confidence[$i][$j]',
      );
      expect(
        (actualPose[j].x - expectedPose[j].x).abs(),
        lessThan(10.0),
        reason: '$label pose x[$i][$j]',
      );
      expect(
        (actualPose[j].y - expectedPose[j].y).abs(),
        lessThan(10.0),
        reason: '$label pose y[$i][$j]',
      );
    }

    expect(b.face, isNotNull, reason: '$label face[$i]');
    final expectedFace = a.face!;
    final actualFace = b.face!;
    expect(
      _boxDelta(expectedFace.boundingBox, actualFace.boundingBox),
      lessThan(10.0),
      reason: '$label face box[$i]',
    );
    expect(
      actualFace.landmarks.length,
      expectedFace.landmarks.length,
      reason: '$label face landmark count[$i]',
    );
    for (var j = 0; j < expectedFace.landmarks.length; j++) {
      final expectedLandmark = expectedFace.landmarks[j];
      final actualLandmark = actualFace.landmarks[j];
      expect(actualLandmark.type, expectedLandmark.type);
      expect(
        (actualLandmark.x - expectedLandmark.x).abs(),
        lessThan(10.0),
        reason: '$label face x[$i][$j]',
      );
      expect(
        (actualLandmark.y - expectedLandmark.y).abs(),
        lessThan(10.0),
        reason: '$label face y[$i][$j]',
      );
    }
  }
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('full pipeline matches across all native backends',
      (tester) async {
    final interpreter = await _detect(
      label: 'Interpreter',
      useCompiledModel: false,
    );
    final compiledCpu = await _detect(
      label: 'CompiledModel CPU',
      useCompiledModel: true,
      accelerators: const {Accelerator.cpu},
    );
    final compiledAuto = await _detect(
      label: 'CompiledModel GPU+CPU',
      useCompiledModel: true,
    );

    expect(interpreter, isNotEmpty);
    _expectParity(interpreter, compiledCpu, 'CompiledModel CPU');
    _expectParity(interpreter, compiledAuto, 'CompiledModel GPU+CPU');
  }, timeout: const Timeout(Duration(minutes: 10)));
}
