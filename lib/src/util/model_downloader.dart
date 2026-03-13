import 'dart:io';
import 'dart:typed_data';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';

class CatModelDownloader {
  static const String _releaseBaseUrl =
      'https://github.com/hugocornellier/cat_detection/releases/download/v0.0.1-models';

  static const String model256 = 'cat_face_landmarks_256_float16.tflite';
  static const String model320 = 'cat_face_landmarks_320_float16.tflite';

  static Future<Directory> _cacheDir() async {
    final appDir = await getApplicationSupportDirectory();
    final dir = Directory('${appDir.path}/cat_detection/models');
    if (!await dir.exists()) {
      await dir.create(recursive: true);
    }
    return dir;
  }

  static Future<Uint8List> getModel(
    String fileName, {
    void Function(int received, int total)? onProgress,
  }) async {
    final dir = await _cacheDir();
    final file = File('${dir.path}/$fileName');

    if (await file.exists()) {
      return await file.readAsBytes();
    }

    final url = '$_releaseBaseUrl/$fileName';
    final request = http.Request('GET', Uri.parse(url));
    final response = await http.Client().send(request);

    if (response.statusCode != 200) {
      throw HttpException(
        'Failed to download $fileName: HTTP ${response.statusCode}',
        uri: Uri.parse(url),
      );
    }

    final totalBytes = response.contentLength ?? -1;
    final chunks = <List<int>>[];
    int received = 0;

    await for (final chunk in response.stream) {
      chunks.add(chunk);
      received += chunk.length;
      onProgress?.call(received, totalBytes);
    }

    final bytes = Uint8List(received);
    int offset = 0;
    for (final chunk in chunks) {
      bytes.setRange(offset, offset + chunk.length, chunk);
      offset += chunk.length;
    }

    await file.writeAsBytes(bytes, flush: true);
    return bytes;
  }

  static Future<(Uint8List, Uint8List)> getEnsembleModels({
    void Function(String model, int received, int total)? onProgress,
  }) async {
    final results = await Future.wait([
      getModel(
        model256,
        onProgress:
            onProgress != null ? (r, t) => onProgress(model256, r, t) : null,
      ),
      getModel(
        model320,
        onProgress:
            onProgress != null ? (r, t) => onProgress(model320, r, t) : null,
      ),
    ]);
    return (results[0], results[1]);
  }

  static Future<bool> isEnsembleCached() async {
    final dir = await _cacheDir();
    final f256 = File('${dir.path}/$model256');
    final f320 = File('${dir.path}/$model320');
    return await f256.exists() && await f320.exists();
  }

  static Future<void> clearCache() async {
    final dir = await _cacheDir();
    if (await dir.exists()) {
      await dir.delete(recursive: true);
    }
  }
}
