// ignore_for_file: avoid_print

// Species gate, exercised on a frame holding a man, a cat and a dog.
//
// Before 3.1.0 the body detector's every hit was returned as a [Cat] with
// cat face landmarks run on it, whatever the species classifier said. This
// asserts the gate: only the cat survives.

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:cat_detection/cat_detection.dart';

// The full man+cat+dog frame is deliberately not used for the cat side: the
// SSD body detector does not resolve the cat at that scale (it is ~340x500 in a
// 1122x1402 frame and is missed at every threshold down to 0.15). This closeup
// is a crop of that same frame, so the man's leg is still present and the gate
// still has something to reject.
const _imagePath = 'integration_test/test_images/cat_closeup.png';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('returns only the cat from a frame holding a cat and a person', (
    tester,
  ) async {
    final data = await rootBundle.load(_imagePath);
    final mat = cv.imdecode(data.buffer.asUint8List(), cv.IMREAD_COLOR);
    addTearDown(mat.dispose);

    final detector = CatDetector(mode: CatDetectionMode.full);
    await detector.initialize();
    addTearDown(detector.dispose);

    final results = await detector.detectFromMat(
      mat,
      imageWidth: mat.cols,
      imageHeight: mat.rows,
    );

    print(
      'GATE image ${mat.cols}x${mat.rows}, cats returned: ${results.length}',
    );
    for (final r in results) {
      print(
        'GATE   species=${r.species} breed=${r.breed} '
        'conf=${r.speciesConfidence?.toStringAsFixed(3)} '
        'score=${r.score.toStringAsFixed(3)} '
        'face=${r.face != null} landmarks=${r.face?.landmarks.length ?? 0}',
      );
    }

    expect(results, isNotEmpty, reason: 'the cat should be found');
    expect(
      results.length,
      1,
      reason: 'the person must be dropped by the species gate',
    );

    final only = results.single;
    expect(only.species, 'cat');
    expect(only.face, isNotNull, reason: 'face stage should run on a real cat');
  });
}
