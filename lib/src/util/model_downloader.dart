import 'dart:typed_data';
import 'package:animal_detection/animal_detection.dart';

/// Cat-specific model downloader wrapping [SpeciesModelDownloader].
class CatModelDownloader {
  static const _downloader = SpeciesModelDownloader(
    releaseBaseUrl:
        'https://github.com/hugocornellier/cat_detection/releases/download/v0.0.1-models',
    cacheSubdir: 'cat_detection/models',
    model256Name: 'cat_face_landmarks_256_float16.tflite',
    model320Name: 'cat_face_landmarks_320_float16.tflite',
  );

  static Future<(Uint8List, Uint8List)> getEnsembleModels({
    void Function(String model, int received, int total)? onProgress,
  }) =>
      _downloader.getEnsembleModels(onProgress: onProgress);

  static Future<bool> isEnsembleCached() => _downloader.isEnsembleCached();

  static Future<void> clearCache() => _downloader.clearCache();
}
