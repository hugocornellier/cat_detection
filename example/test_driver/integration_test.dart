// Host-side driver so integration tests can run in profile mode (AOT), which
// `flutter test` cannot do. Needed for representative performance numbers:
//
//   flutter drive --profile \
//     --driver=test_driver/integration_test.dart \
//     --target=integration_test/perf_benchmark_test.dart -d macos
import 'package:integration_test/integration_test_driver.dart';

Future<void> main() => integrationDriver();
