// Per-change performance harness for cat_detection.
//
// Measures whole-pipeline latency under different PerformanceConfig settings so
// the effect of a single change can be attributed rather than estimated.
// Run in profile mode for AOT-representative numbers:
//
//   flutter test integration_test/perf_benchmark_test.dart -d macos --profile
//
// full mode  = body detect + species + pose + localizer + landmarks
// poseOnly   = body detect + species + pose
// the difference isolates the face stages.
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:cat_detection/cat_detection.dart';

const _catImagePath = 'integration_test/test_images/cat.jpg';
const _warmup = 3;
const _iters = 15;

Future<double> _bench(
  cv.Mat mat,
  CatDetectionMode mode,
  PerformanceConfig config,
) async {
  final detector = CatDetector(mode: mode, performanceConfig: config);
  await detector.initialize();
  try {
    for (int i = 0; i < _warmup; i++) {
      await detector.detectFromMat(mat,
          imageWidth: mat.cols, imageHeight: mat.rows);
    }
    final sw = Stopwatch()..start();
    for (int i = 0; i < _iters; i++) {
      await detector.detectFromMat(mat,
          imageWidth: mat.cols, imageHeight: mat.rows);
    }
    sw.stop();
    return sw.elapsedMicroseconds / _iters / 1000.0;
  } finally {
    await detector.dispose();
  }
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('PerformanceConfig A/B across the pipeline', (tester) async {
    final data = await rootBundle.load(_catImagePath);
    final mat = cv.imdecode(data.buffer.asUint8List(), cv.IMREAD_COLOR);
    addTearDown(mat.dispose);
    debugPrint('PERF image ${mat.cols}x${mat.rows}, '
        '$_iters iters after $_warmup warmup');

    const configs = <String, PerformanceConfig>{
      'disabled (isolate default)': PerformanceConfig.disabled,
      'auto (CatDetector default)': PerformanceConfig(),
      'xnnpack (explicit)': PerformanceConfig.xnnpack(),
    };

    final full = <String, double>{};
    final pose = <String, double>{};

    for (final e in configs.entries) {
      full[e.key] = await _bench(mat, CatDetectionMode.full, e.value);
      debugPrint('PERF full     ${e.key.padRight(28)} '
          '${full[e.key]!.toStringAsFixed(1)} ms/frame');
    }
    for (final e in configs.entries) {
      pose[e.key] = await _bench(mat, CatDetectionMode.poseOnly, e.value);
      debugPrint('PERF poseOnly ${e.key.padRight(28)} '
          '${pose[e.key]!.toStringAsFixed(1)} ms/frame');
    }

    final base = full['disabled (isolate default)']!;
    debugPrint('PERF ---- summary (full pipeline) ----');
    for (final k in configs.keys) {
      final v = full[k]!;
      final face = v - pose[k]!;
      debugPrint('PERF ${k.padRight(28)} total=${v.toStringAsFixed(1)}ms  '
          'body=${pose[k]!.toStringAsFixed(1)}ms  '
          'face=${face.toStringAsFixed(1)}ms  '
          'speedup vs disabled=${(base / v).toStringAsFixed(2)}x');
    }
  }, timeout: const Timeout(Duration(minutes: 10)));
}
