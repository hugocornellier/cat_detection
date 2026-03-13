import 'dart:typed_data';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:flutter_litert/flutter_litert.dart';
import 'package:animal_detection/animal_detection.dart';

class ImageUtils {
  static const List<double> imagenetMean = [0.485, 0.456, 0.406];
  static const List<double> imagenetStd = [0.229, 0.224, 0.225];

  static (cv.Mat, LetterboxParams) letterboxResize(cv.Mat src, int targetSize) {
    final params = computeLetterboxParams(
      srcWidth: src.cols,
      srcHeight: src.rows,
      targetWidth: targetSize,
      targetHeight: targetSize,
    );
    final resized = cv.resize(src, (params.newWidth, params.newHeight));
    final padded = cv.copyMakeBorder(
      resized,
      params.padTop,
      params.padBottom,
      params.padLeft,
      params.padRight,
      cv.BORDER_CONSTANT,
      value: cv.Scalar.black,
    );
    resized.dispose();
    return (padded, params);
  }

  static (cv.Mat, CropMetadata) cropAndResize(
    cv.Mat src,
    BoundingBox bbox,
    double margin,
    int targetSize,
  ) {
    final bw = bbox.right - bbox.left;
    final bh = bbox.bottom - bbox.top;
    final cx1 = (bbox.left - bw * margin).clamp(0.0, src.cols.toDouble());
    final cy1 = (bbox.top - bh * margin).clamp(0.0, src.rows.toDouble());
    final cx2 = (bbox.right + bw * margin).clamp(0.0, src.cols.toDouble());
    final cy2 = (bbox.bottom + bh * margin).clamp(0.0, src.rows.toDouble());
    final cropW = cx2 - cx1;
    final cropH = cy2 - cy1;

    final cropped = src.region(
      cv.Rect(cx1.toInt(), cy1.toInt(), cropW.toInt(), cropH.toInt()),
    );
    final resized = cv.resize(cropped, (targetSize, targetSize));
    cropped.dispose();

    return (
      resized,
      CropMetadata(cx1: cx1, cy1: cy1, cropW: cropW, cropH: cropH),
    );
  }

  static Float32List matToFloat32(cv.Mat mat) {
    return bgrBytesToRgbFloat32(
      bytes: mat.data,
      totalPixels: mat.rows * mat.cols,
    );
  }

  static Float32List matToFloat32ImageNet(cv.Mat mat) {
    final int totalPixels = mat.rows * mat.cols;
    final Uint8List bytes = mat.data;
    final result = Float32List(totalPixels * 3);

    for (int i = 0; i < totalPixels; i++) {
      final int bgr = i * 3;
      final int rgb = i * 3;
      final double r = bytes[bgr + 2] / 255.0;
      final double g = bytes[bgr + 1] / 255.0;
      final double b = bytes[bgr + 0] / 255.0;
      result[rgb + 0] = (r - imagenetMean[0]) / imagenetStd[0];
      result[rgb + 1] = (g - imagenetMean[1]) / imagenetStd[1];
      result[rgb + 2] = (b - imagenetMean[2]) / imagenetStd[2];
    }

    return result;
  }

  static (Float32List, LetterboxParams) letterboxAndNormalizeImageNet(
    cv.Mat src,
    int targetSize,
  ) {
    final (padded, params) = letterboxResize(src, targetSize);
    final normalized = matToFloat32ImageNet(padded);
    padded.dispose();
    return (normalized, params);
  }
}
