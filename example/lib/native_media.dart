import 'dart:async';
import 'dart:io';
import 'dart:math' as math;

import 'package:cat_detection/cat_detection.dart';
import 'package:camera/camera.dart';
import 'package:file_selector/file_selector.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:path_provider/path_provider.dart';
import 'package:sensors_plus/sensors_plus.dart';
import 'package:video_player/video_player.dart';

const Color _bodyColor = Color(0xFFFF9800);
const Color _poseColor = Color(0xFF00E676);
const int _cameraMaxDimension = 640;

class LiveCameraScreen extends StatefulWidget {
  const LiveCameraScreen({super.key});

  @override
  State<LiveCameraScreen> createState() => _LiveCameraScreenState();
}

class _LiveCameraScreenState extends State<LiveCameraScreen> {
  CameraController? _cameraController;
  List<CameraDescription> _availableCameras = const [];
  CatDetector? _detector;
  List<Cat> _cats = const [];
  Size? _imageSize;
  int? _sensorOrientation;
  bool _isFrontCamera = false;
  bool _isSwitchingCamera = false;
  bool _streamStarted = false;
  bool _isInitialized = false;
  bool _useCompiledModel = true;
  CatDetectionMode _detectionMode = CatDetectionMode.full;
  AnimalPoseModel _poseModel = AnimalPoseModel.rtmpose;
  DeviceOrientation _deviceOrientation = DeviceOrientation.portraitUp;
  StreamSubscription<AccelerometerEvent>? _accelerometerSub;
  final FrameThrottle _throttle = FrameThrottle();
  final FpsCounter _fpsCounter = FpsCounter();
  int _fps = 0;
  int _detectionTimeMs = 0;

  @override
  void initState() {
    super.initState();
    _initCamera();
    if (!kIsWeb && (Platform.isAndroid || Platform.isIOS)) {
      _accelerometerSub = accelerometerEventStream().listen((event) {
        final next = event.x.abs() > event.y.abs()
            ? (event.x > 0
                ? DeviceOrientation.landscapeLeft
                : DeviceOrientation.landscapeRight)
            : (event.y > 0
                ? DeviceOrientation.portraitUp
                : DeviceOrientation.portraitDown);
        if (next == DeviceOrientation.portraitDown &&
            (_deviceOrientation == DeviceOrientation.landscapeLeft ||
                _deviceOrientation == DeviceOrientation.landscapeRight)) {
          return;
        }
        if (next != _deviceOrientation && mounted) {
          setState(() => _deviceOrientation = next);
        }
      });
    }
  }

  Future<CatDetector> _createDetector(bool useCompiledModel) {
    return CatDetector.create(
      mode: _detectionMode,
      poseModel: _poseModel,
      landmarkModel: CatLandmarkModel.full,
      performanceConfig: const PerformanceConfig.xnnpack(),
      useCompiledModel: useCompiledModel,
    );
  }

  Future<void> _reinitDetector() async {
    final old = _detector;
    _detector = null;
    await old?.dispose();
    try {
      _detector = await _createDetector(_useCompiledModel);
    } catch (error) {
      if (!_useCompiledModel) rethrow;
      debugPrint(
        'Live camera CompiledModel init failed; falling back to XNNPACK: '
        '$error',
      );
      _useCompiledModel = false;
      _detector = await _createDetector(false);
    }
  }

  Future<void> _toggleBackend() async {
    setState(() {
      _isInitialized = false;
      _useCompiledModel = !_useCompiledModel;
    });
    try {
      await _reinitDetector();
    } catch (error) {
      if (mounted) _showError('Could not initialize detector: $error');
    }
    if (mounted) setState(() => _isInitialized = _detector?.isReady ?? false);
  }

  Future<void> _updateDetector(VoidCallback update) async {
    setState(() {
      _isInitialized = false;
      update();
      _cats = const [];
    });
    try {
      await _reinitDetector();
    } catch (error) {
      if (mounted) _showError('Could not update detector: $error');
    }
    if (mounted) setState(() => _isInitialized = _detector?.isReady ?? false);
  }

  Future<void> _initCamera() async {
    try {
      await _reinitDetector();
      if (mounted) setState(() => _isInitialized = true);
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        if (mounted) _showError('No cameras available');
        return;
      }
      _availableCameras = cameras;
      final camera = cameras.firstWhere(
        (value) => value.lensDirection == CameraLensDirection.front,
        orElse: () => cameras.first,
      );
      await _startController(camera);
    } catch (error, stack) {
      debugPrint('Camera init failed: $error\n$stack');
      if (mounted) _showError('Error initializing camera: $error');
    }
  }

  Future<void> _startController(CameraDescription camera) async {
    final controller = CameraController(
      camera,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );
    await controller.initialize();
    if (!mounted) {
      await controller.dispose();
      return;
    }
    setState(() {
      _cameraController = controller;
      _sensorOrientation = camera.sensorOrientation;
      _isFrontCamera = camera.lensDirection == CameraLensDirection.front;
    });
    await controller.startImageStream(_processCameraImage);
    _streamStarted = true;
  }

  bool get _canSwitchCamera {
    if (kIsWeb || !(Platform.isAndroid || Platform.isIOS)) return false;
    final front = _availableCameras.any(
      (value) => value.lensDirection == CameraLensDirection.front,
    );
    final back = _availableCameras.any(
      (value) => value.lensDirection == CameraLensDirection.back,
    );
    return front && back;
  }

  Future<void> _switchCamera() async {
    if (_isSwitchingCamera || !_canSwitchCamera) return;
    final target =
        _isFrontCamera ? CameraLensDirection.back : CameraLensDirection.front;
    final next = _availableCameras.firstWhere(
      (value) => value.lensDirection == target,
    );
    final previous = _cameraController;
    setState(() {
      _isSwitchingCamera = true;
      _cameraController = null;
      _cats = const [];
      _imageSize = null;
    });
    try {
      if (_streamStarted) {
        await previous?.stopImageStream();
        _streamStarted = false;
      }
      await previous?.dispose();
      await _startController(next);
    } catch (error) {
      if (mounted) _showError('Error switching camera: $error');
    } finally {
      if (mounted) setState(() => _isSwitchingCamera = false);
    }
  }

  DeviceOrientation _effectiveOrientation(BuildContext context) {
    return _cameraController?.value.deviceOrientation ??
        (MediaQuery.orientationOf(context) == Orientation.portrait
            ? DeviceOrientation.portraitUp
            : DeviceOrientation.landscapeLeft);
  }

  Future<void> _processCameraImage(CameraImage image) async {
    if (_fpsCounter.tick() && mounted) setState(() => _fps = _fpsCounter.fps);
    await _throttle.run(() async {
      final detector = _detector;
      if (detector == null || !_isInitialized || !mounted) return;
      try {
        final started = DateTime.now();
        final sensor = _sensorOrientation;
        final rotation = sensor == null
            ? null
            : rotationForFrame(
                width: image.width,
                height: image.height,
                sensorOrientation: sensor,
                isFrontCamera: _isFrontCamera,
                deviceOrientation: _effectiveOrientation(context),
              );
        final size = detectionSize(
          width: image.width,
          height: image.height,
          rotation: rotation,
          maxDim: _cameraMaxDimension,
        );
        final cats = await detector.detectFromCameraImage(
          image,
          rotation: rotation,
          isBgra: Platform.isMacOS,
          maxDim: _cameraMaxDimension,
        );
        if (mounted) {
          setState(() {
            _cats = cats;
            _imageSize = size;
            _detectionTimeMs =
                DateTime.now().difference(started).inMilliseconds;
          });
        }
      } catch (error) {
        debugPrint('Camera frame failed: $error');
      }
    });
  }

  void _showError(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message)),
    );
  }

  @override
  void dispose() {
    _accelerometerSub?.cancel();
    if (_streamStarted) _cameraController?.stopImageStream();
    _cameraController?.dispose();
    _detector?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final controller = _cameraController;
    if (!_isInitialized || controller == null) {
      return Scaffold(
        appBar: AppBar(title: const Text('Live Cat Detection')),
        body: const Center(child: CircularProgressIndicator()),
      );
    }
    final effective = _effectiveOrientation(context);
    final portrait = effective == DeviceOrientation.portraitUp ||
        effective == DeviceOrientation.portraitDown;
    final aspect = portrait
        ? 1.0 / controller.value.aspectRatio
        : controller.value.aspectRatio;
    final turns = barQuarterTurns(_deviceOrientation);
    final mirror = (Platform.isAndroid && _isFrontCamera) || Platform.isWindows;
    return Scaffold(
      body: Stack(
        fit: StackFit.expand,
        children: [
          Center(
            child: AspectRatio(
              aspectRatio: aspect,
              child: Stack(
                fit: StackFit.expand,
                children: [
                  CameraPreview(controller),
                  if (_imageSize != null)
                    CustomPaint(
                      painter: _CatPainter(
                        cats: _cats,
                        imageSize: _imageSize!,
                        mirrorHorizontally: mirror,
                      ),
                    ),
                ],
              ),
            ),
          ),
          _positionedTopBar(turns),
        ],
      ),
    );
  }

  Widget _positionedTopBar(int turns) {
    final padding = MediaQuery.paddingOf(context);
    final bar = _cameraTopBar();
    if (turns == 0) {
      return Positioned(
        top: padding.top,
        left: padding.left,
        right: padding.right,
        child: bar,
      );
    }
    return Positioned(
      top: padding.top,
      bottom: padding.bottom,
      left: turns == 3 ? padding.left : null,
      right: turns == 1 ? padding.right : null,
      width: kToolbarHeight,
      child: RotatedBox(quarterTurns: turns, child: bar),
    );
  }

  Widget _cameraTopBar() {
    final mobile = !kIsWeb && (Platform.isAndroid || Platform.isIOS);
    final fps = SizedBox(
      width: 70,
      child: Text(
        'FPS: $_fps',
        textAlign: mobile ? TextAlign.left : TextAlign.right,
        style: const TextStyle(color: Colors.white, fontSize: 14),
      ),
    );
    final milliseconds = SizedBox(
      width: 70,
      child: Text(
        '${_detectionTimeMs}ms',
        style: const TextStyle(color: Colors.white, fontSize: 14),
      ),
    );
    return Material(
      color: Colors.black.withAlpha(179),
      elevation: 4,
      child: SizedBox(
        height: kToolbarHeight,
        child: Row(
          children: [
            IconButton(
              tooltip: 'Back',
              color: Colors.white,
              onPressed: () => Navigator.maybePop(context),
              icon: const Icon(Icons.arrow_back),
            ),
            if (mobile) ...[
              fps,
              const Text(' | ', style: TextStyle(color: Colors.white)),
              milliseconds,
              const Spacer(),
            ] else
              const Expanded(
                child: Padding(
                  padding: EdgeInsets.symmetric(horizontal: 8),
                  child: Text(
                    'Live Cat Detection',
                    style: TextStyle(color: Colors.white, fontSize: 18),
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
              ),
            if (_canSwitchCamera)
              IconButton(
                tooltip: 'Switch camera',
                color: Colors.white,
                onPressed: _isSwitchingCamera ? null : _switchCamera,
                icon: Icon(
                  Platform.isIOS
                      ? Icons.flip_camera_ios
                      : Icons.flip_camera_android,
                ),
              ),
            TextButton(
              onPressed: _isInitialized ? _toggleBackend : null,
              style: TextButton.styleFrom(minimumSize: const Size(92, 36)),
              child: Text(
                _useCompiledModel ? 'CM' : 'Interpreter',
                style: const TextStyle(
                  color: Colors.amberAccent,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
            PopupMenuButton<void>(
              tooltip: 'Settings',
              icon: const Icon(Icons.settings, color: Colors.white),
              color: Colors.blueGrey,
              itemBuilder: (context) => [
                PopupMenuItem<void>(
                  enabled: false,
                  child: SizedBox(
                    width: 250,
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        DropdownButtonFormField<CatDetectionMode>(
                          initialValue: _detectionMode,
                          dropdownColor: Colors.blueGrey[800],
                          style: const TextStyle(color: Colors.white),
                          decoration: const InputDecoration(
                            labelText: 'Detection mode',
                            labelStyle: TextStyle(color: Colors.white70),
                          ),
                          items: CatDetectionMode.values
                              .map(
                                (value) => DropdownMenuItem(
                                  value: value,
                                  child: Text(value.name),
                                ),
                              )
                              .toList(),
                          onChanged: (value) {
                            if (value != null) {
                              _updateDetector(
                                () => _detectionMode = value,
                              );
                            }
                          },
                        ),
                        DropdownButtonFormField<AnimalPoseModel>(
                          initialValue: _poseModel,
                          dropdownColor: Colors.blueGrey[800],
                          style: const TextStyle(color: Colors.white),
                          decoration: const InputDecoration(
                            labelText: 'Pose model',
                            labelStyle: TextStyle(color: Colors.white70),
                          ),
                          items: AnimalPoseModel.values
                              .map(
                                (value) => DropdownMenuItem(
                                  value: value,
                                  child: Text(value.name),
                                ),
                              )
                              .toList(),
                          onChanged: _detectionMode != CatDetectionMode.faceOnly
                              ? (value) {
                                  if (value != null) {
                                    _updateDetector(() => _poseModel = value);
                                  }
                                }
                              : null,
                        ),
                      ],
                    ),
                  ),
                ),
              ],
            ),
            if (!mobile) ...[
              fps,
              const Text(' | ', style: TextStyle(color: Colors.white)),
              milliseconds,
              const SizedBox(width: 8),
            ],
          ],
        ),
      ),
    );
  }
}

class _CatPainter extends CustomPainter {
  const _CatPainter({
    required this.cats,
    required this.imageSize,
    this.mirrorHorizontally = false,
  });

  final List<Cat> cats;
  final Size imageSize;
  final bool mirrorHorizontally;

  @override
  void paint(Canvas canvas, Size size) {
    if (imageSize.isEmpty) return;
    canvas.save();
    if (mirrorHorizontally) {
      canvas.translate(size.width, 0);
      canvas.scale(-1, 1);
    }
    final scaleX = size.width / imageSize.width;
    final scaleY = size.height / imageSize.height;
    final bodyPaint = Paint()
      ..color = _bodyColor
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2;
    final posePaint = Paint()
      ..color = _poseColor
      ..strokeWidth = 3
      ..strokeCap = StrokeCap.round;
    for (final cat in cats) {
      final box = cat.boundingBox;
      canvas.drawRect(
        Rect.fromLTRB(
          box.left * scaleX,
          box.top * scaleY,
          box.right * scaleX,
          box.bottom * scaleY,
        ),
        bodyPaint,
      );
      final pose = cat.pose;
      if (pose != null) {
        for (final connection in animalPoseConnections) {
          final a = pose.getLandmark(connection[0]);
          final b = pose.getLandmark(connection[1]);
          if (a == null ||
              b == null ||
              a.confidence < 0.3 ||
              b.confidence < 0.3) {
            continue;
          }
          canvas.drawLine(
            Offset(a.x * scaleX, a.y * scaleY),
            Offset(b.x * scaleX, b.y * scaleY),
            posePaint,
          );
        }
        for (final landmark in pose.landmarks) {
          if (landmark.confidence >= 0.3) {
            canvas.drawCircle(
              Offset(landmark.x * scaleX, landmark.y * scaleY),
              3,
              posePaint,
            );
          }
        }
      }
      final face = cat.face;
      if (face != null) {
        final facePaint = Paint()
          ..color = Colors.lightBlueAccent
          ..style = PaintingStyle.stroke
          ..strokeWidth = 2;
        final faceBox = face.boundingBox;
        canvas.drawRect(
          Rect.fromLTRB(
            faceBox.left * scaleX,
            faceBox.top * scaleY,
            faceBox.right * scaleX,
            faceBox.bottom * scaleY,
          ),
          facePaint,
        );
        for (final connection in catLandmarkConnections) {
          final a = face.getLandmark(connection[0]);
          final b = face.getLandmark(connection[1]);
          if (a == null || b == null) continue;
          canvas.drawLine(
            Offset(a.x * scaleX, a.y * scaleY),
            Offset(b.x * scaleX, b.y * scaleY),
            facePaint,
          );
        }
        for (final landmark in face.landmarks) {
          canvas.drawCircle(
            Offset(landmark.x * scaleX, landmark.y * scaleY),
            2,
            facePaint,
          );
        }
      }
      _paintLabel(canvas, cat, scaleX, scaleY);
    }
    canvas.restore();
  }

  void _paintLabel(Canvas canvas, Cat cat, double scaleX, double scaleY) {
    final species = cat.species ?? 'Cat';
    final confidence = (cat.score * 100).toStringAsFixed(0);
    final painter = TextPainter(
      text: TextSpan(
        text: '$species $confidence%',
        style: const TextStyle(
          color: Colors.white,
          fontSize: 13,
          fontWeight: FontWeight.bold,
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout();
    final left = cat.boundingBox.left * scaleX;
    final top = math.max(
      0.0,
      cat.boundingBox.top * scaleY - painter.height - 6,
    );
    canvas.drawRect(
      Rect.fromLTWH(left, top, painter.width + 8, painter.height + 6),
      Paint()..color = _bodyColor,
    );
    painter.paint(canvas, Offset(left + 4, top + 3));
  }

  @override
  bool shouldRepaint(covariant _CatPainter oldDelegate) =>
      oldDelegate.cats != cats ||
      oldDelegate.imageSize != imageSize ||
      oldDelegate.mirrorHorizontally != mirrorHorizontally;
}

class VideoFileScreen extends StatefulWidget {
  const VideoFileScreen({super.key});

  @override
  State<VideoFileScreen> createState() => _VideoFileScreenState();
}

class _VideoFileScreenState extends State<VideoFileScreen> {
  CatDetector? _detector;
  bool _isInitialized = false;
  bool _isProcessing = false;
  bool _cancelRequested = false;
  bool _useCompiledModel = true;
  CatDetectionMode _detectionMode = CatDetectionMode.full;
  AnimalPoseModel _poseModel = AnimalPoseModel.rtmpose;
  bool _smoothingEnabled = true;
  final _CatSmoother _smoother = _CatSmoother();
  String? _inputPath;
  String? _outputPath;
  String? _errorMessage;
  String? _statusMessage;
  int _totalFrames = 0;
  int _processedFrames = 0;
  double _videoFps = 0;
  int _videoWidth = 0;
  int _videoHeight = 0;
  Duration _elapsed = Duration.zero;
  final Stopwatch _wallClock = Stopwatch();
  VideoPlayerController? _playerController;
  bool _playerReady = false;
  String? _playerError;

  bool get _supportsInAppPlayer =>
      kIsWeb || Platform.isAndroid || Platform.isIOS || Platform.isMacOS;

  @override
  void initState() {
    super.initState();
    _initDetector();
  }

  Future<CatDetector> _createDetector(bool useCompiledModel) {
    return CatDetector.create(
      mode: _detectionMode,
      poseModel: _poseModel,
      landmarkModel: CatLandmarkModel.full,
      performanceConfig: const PerformanceConfig.xnnpack(),
      useCompiledModel: useCompiledModel,
    );
  }

  Future<void> _initDetector() async {
    try {
      _detector = await _createDetector(_useCompiledModel);
    } catch (error) {
      if (!_useCompiledModel) {
        if (mounted) setState(() => _errorMessage = '$error');
        return;
      }
      _useCompiledModel = false;
      try {
        _detector = await _createDetector(false);
      } catch (fallbackError) {
        if (mounted) setState(() => _errorMessage = '$fallbackError');
        return;
      }
    }
    if (mounted) setState(() => _isInitialized = true);
  }

  Future<void> _reinitDetector() async {
    setState(() => _isInitialized = false);
    final old = _detector;
    _detector = null;
    await old?.dispose();
    await _initDetector();
  }

  Future<void> _toggleBackend() async {
    if (_isProcessing) return;
    _useCompiledModel = !_useCompiledModel;
    await _reinitDetector();
  }

  @override
  void dispose() {
    _cancelRequested = true;
    _detector?.dispose();
    _playerController?.dispose();
    super.dispose();
  }

  Future<void> _pickVideo() async {
    const group = XTypeGroup(
      label: 'Videos',
      extensions: ['mp4', 'mov', 'm4v'],
    );
    final file = await openFile(acceptedTypeGroups: [group]);
    if (file != null) await _processVideo(file.path);
  }

  Future<void> _processVideo(String path) async {
    final detector = _detector;
    if (detector == null) return;
    final cap = cv.VideoCapture.fromFile(path);
    if (!cap.isOpened) {
      cap.release();
      setState(() => _errorMessage = 'Could not open video: $path');
      return;
    }
    final fps = cap.get(cv.CAP_PROP_FPS);
    final width = cap.get(cv.CAP_PROP_FRAME_WIDTH).toInt();
    final height = cap.get(cv.CAP_PROP_FRAME_HEIGHT).toInt();
    final total = cap.get(cv.CAP_PROP_FRAME_COUNT).toInt();
    final docs = await getApplicationDocumentsDirectory();
    final output =
        '${docs.path}/cat_${DateTime.now().millisecondsSinceEpoch}.mp4';
    final writer = cv.VideoWriter.fromFile(
      output,
      'avc1',
      fps > 0 ? fps : 30,
      (width, height),
    );
    if (!writer.isOpened) {
      cap.release();
      setState(() {
        _errorMessage = 'Could not create H.264 output video at $output';
      });
      return;
    }
    await _disposePlayer();
    if (!mounted) {
      cap.release();
      writer.release();
      return;
    }
    setState(() {
      _inputPath = path;
      _outputPath = output;
      _videoFps = fps;
      _videoWidth = width;
      _videoHeight = height;
      _totalFrames = total;
      _processedFrames = 0;
      _isProcessing = true;
      _cancelRequested = false;
      _errorMessage = null;
      _statusMessage = 'Processing...';
      _elapsed = Duration.zero;
    });
    _smoother.reset();
    _wallClock
      ..reset()
      ..start();
    cv.Mat? frame;
    var index = 0;
    try {
      while (mounted && !_cancelRequested) {
        final read = cap.read(m: frame);
        if (!read.$1 || read.$2.isEmpty) break;
        frame = read.$2;
        final raw = await detector.detectFromMat(
          frame,
          imageWidth: frame.cols,
          imageHeight: frame.rows,
        );
        final time = fps > 0 ? index / fps : index / 30;
        final cats = _smoother.apply(raw, time);
        _drawCats(frame, cats);
        writer.write(frame);
        index++;
        if (index % 4 == 0) {
          if (!mounted) break;
          setState(() {
            _processedFrames = index;
            _elapsed = _wallClock.elapsed;
          });
          await Future<void>.delayed(Duration.zero);
        }
      }
      if (mounted) {
        setState(() {
          _processedFrames = index;
          _elapsed = _wallClock.elapsed;
          _statusMessage = _cancelRequested
              ? 'Cancelled after $index frames.'
              : 'Done. Wrote $index frames to:\n$output';
        });
      }
    } catch (error) {
      if (mounted) setState(() => _errorMessage = 'Processing failed: $error');
    } finally {
      _wallClock.stop();
      cap.release();
      writer.release();
      frame?.dispose();
      if (mounted) setState(() => _isProcessing = false);
      if (mounted && !_cancelRequested) await _initPlayer(output);
    }
  }

  void _drawCats(cv.Mat frame, List<Cat> cats) {
    final body = cv.Scalar(0, 152, 255);
    final pose = cv.Scalar(118, 230, 0);
    final black = cv.Scalar(0, 0, 0);
    for (final cat in cats) {
      final box = cat.boundingBox;
      final left = box.left.round().clamp(0, frame.cols - 1);
      final top = box.top.round().clamp(0, frame.rows - 1);
      final right = box.right.round().clamp(0, frame.cols - 1);
      final bottom = box.bottom.round().clamp(0, frame.rows - 1);
      cv.rectangle(
        frame,
        cv.Rect(
          left,
          top,
          math.max(1, right - left),
          math.max(1, bottom - top),
        ),
        body,
        thickness: 2,
      );
      final catPose = cat.pose;
      if (catPose != null) {
        for (final connection in animalPoseConnections) {
          final a = catPose.getLandmark(connection[0]);
          final b = catPose.getLandmark(connection[1]);
          if (a == null ||
              b == null ||
              a.confidence < 0.3 ||
              b.confidence < 0.3) {
            continue;
          }
          cv.line(
            frame,
            cv.Point(a.x.round(), a.y.round()),
            cv.Point(b.x.round(), b.y.round()),
            pose,
            thickness: 3,
          );
        }
        for (final landmark in catPose.landmarks) {
          if (landmark.confidence >= 0.3) {
            cv.circle(
              frame,
              cv.Point(landmark.x.round(), landmark.y.round()),
              3,
              pose,
              thickness: -1,
            );
          }
        }
      }
      final face = cat.face;
      if (face != null) {
        final faceColor = cv.Scalar(255, 191, 0);
        final box = face.boundingBox;
        final faceLeft = box.left.round().clamp(0, frame.cols - 1);
        final faceTop = box.top.round().clamp(0, frame.rows - 1);
        final faceRight = box.right.round().clamp(0, frame.cols - 1);
        final faceBottom = box.bottom.round().clamp(0, frame.rows - 1);
        cv.rectangle(
          frame,
          cv.Rect(
            faceLeft,
            faceTop,
            math.max(1, faceRight - faceLeft),
            math.max(1, faceBottom - faceTop),
          ),
          faceColor,
          thickness: 2,
        );
        for (final connection in catLandmarkConnections) {
          final a = face.getLandmark(connection[0]);
          final b = face.getLandmark(connection[1]);
          if (a == null || b == null) continue;
          cv.line(
            frame,
            cv.Point(a.x.round(), a.y.round()),
            cv.Point(b.x.round(), b.y.round()),
            faceColor,
            thickness: 2,
          );
        }
        for (final landmark in face.landmarks) {
          cv.circle(
            frame,
            cv.Point(landmark.x.round(), landmark.y.round()),
            2,
            faceColor,
            thickness: -1,
          );
        }
      }
      final label = '${cat.species ?? 'Cat'} ${(cat.score * 100).round()}%';
      final (textSize, _) = cv.getTextSize(
        label,
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        2,
      );
      final labelTop = math.max(0, top - textSize.height - 8);
      cv.rectangle(
        frame,
        cv.Rect(
          left,
          labelTop,
          math.min(frame.cols - left, textSize.width + 8),
          textSize.height + 8,
        ),
        body,
        thickness: -1,
      );
      cv.putText(
        frame,
        label,
        cv.Point(left + 4, labelTop + textSize.height + 2),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        black,
        thickness: 2,
      );
    }
  }

  Future<void> _disposePlayer() async {
    final player = _playerController;
    _playerController = null;
    _playerReady = false;
    _playerError = null;
    await player?.dispose();
  }

  Future<void> _initPlayer(String path) async {
    await _disposePlayer();
    if (!_supportsInAppPlayer) return;
    final player = VideoPlayerController.file(File(path));
    _playerController = player;
    try {
      await player.initialize();
      await player.setLooping(true);
      if (!mounted) return;
      setState(() => _playerReady = true);
      await player.play();
    } catch (error) {
      if (mounted) {
        setState(() => _playerError = 'Could not load video: $error');
      }
    }
  }

  Future<void> _openOutput() async {
    final path = _outputPath;
    if (path == null) return;
    if (Platform.isMacOS) {
      await Process.run('open', [path]);
    } else if (Platform.isLinux) {
      await Process.run('xdg-open', [path]);
    } else if (Platform.isWindows) {
      await Process.run('cmd', ['/c', 'start', '', path]);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Video File - Cat Detection'),
        actions: [
          TextButton(
            onPressed: _isInitialized && !_isProcessing ? _toggleBackend : null,
            child: Text(
              _useCompiledModel ? 'CM' : 'Interpreter',
              style: const TextStyle(
                color: Colors.amber,
                fontWeight: FontWeight.bold,
              ),
            ),
          ),
          PopupMenuButton<void>(
            tooltip: 'Settings',
            icon: const Icon(Icons.tune),
            itemBuilder: (context) => [
              PopupMenuItem<void>(
                enabled: false,
                child: SizedBox(
                  width: 260,
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      DropdownButtonFormField<CatDetectionMode>(
                        initialValue: _detectionMode,
                        decoration: const InputDecoration(
                          labelText: 'Detection mode',
                        ),
                        items: CatDetectionMode.values
                            .map(
                              (value) => DropdownMenuItem(
                                value: value,
                                child: Text(value.name),
                              ),
                            )
                            .toList(),
                        onChanged: _isProcessing
                            ? null
                            : (value) {
                                if (value != null) {
                                  _detectionMode = value;
                                  _reinitDetector();
                                }
                              },
                      ),
                      DropdownButtonFormField<AnimalPoseModel>(
                        initialValue: _poseModel,
                        decoration: const InputDecoration(
                          labelText: 'Pose model',
                        ),
                        items: AnimalPoseModel.values
                            .map(
                              (value) => DropdownMenuItem(
                                value: value,
                                child: Text(value.name),
                              ),
                            )
                            .toList(),
                        onChanged: _isProcessing ||
                                _detectionMode == CatDetectionMode.faceOnly
                            ? null
                            : (value) {
                                if (value != null) {
                                  _poseModel = value;
                                  _reinitDetector();
                                }
                              },
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
      body: _buildBody(),
      floatingActionButton: _isInitialized && !_isProcessing
          ? FloatingActionButton.extended(
              onPressed: _pickVideo,
              icon: const Icon(Icons.video_file),
              label: const Text('Pick Video'),
            )
          : (_isProcessing
              ? FloatingActionButton.extended(
                  onPressed: () => setState(() => _cancelRequested = true),
                  icon: const Icon(Icons.cancel),
                  label: const Text('Cancel'),
                )
              : null),
    );
  }

  Widget _buildBody() {
    if (!_isInitialized && _errorMessage == null) {
      return const Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            CircularProgressIndicator(),
            SizedBox(height: 16),
            Text('Initializing detector...'),
          ],
        ),
      );
    }
    final progress = _totalFrames > 0
        ? (_processedFrames / _totalFrames).clamp(0.0, 1.0)
        : 0.0;
    final processedFps = _elapsed.inMilliseconds > 0
        ? _processedFrames * 1000 / _elapsed.inMilliseconds
        : 0.0;
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          if (_errorMessage != null)
            Card(
              color: Colors.red[50],
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Text(_errorMessage!),
              ),
            ),
          if (_inputPath != null) ...[
            _infoRow('Input', _inputPath!),
            _infoRow(
              'Source',
              '$_videoWidth×$_videoHeight @ '
                  '${_videoFps.toStringAsFixed(2)} fps · $_totalFrames frames',
            ),
          ],
          if (!_isProcessing)
            SwitchListTile(
              contentPadding: EdgeInsets.zero,
              title: const Text('Smoothing (One-Euro filter)'),
              subtitle: Text(
                _smoothingEnabled
                    ? 'On: pose landmarks filtered across frames'
                    : 'Off: raw per-frame detections',
              ),
              value: _smoothingEnabled,
              onChanged: (value) {
                setState(() {
                  _smoothingEnabled = value;
                  _smoother.enabled = value;
                  _smoother.reset();
                });
              },
            ),
          if (!_isProcessing && _inputPath != null)
            Align(
              alignment: Alignment.centerLeft,
              child: OutlinedButton.icon(
                onPressed: () => _processVideo(_inputPath!),
                icon: const Icon(Icons.refresh),
                label: const Text('Re-run with current settings'),
              ),
            ),
          const SizedBox(height: 16),
          if (_isProcessing) ...[
            LinearProgressIndicator(value: _totalFrames > 0 ? progress : null),
            const SizedBox(height: 8),
            Text(
              'Frame $_processedFrames / $_totalFrames · '
              '${(progress * 100).toStringAsFixed(1)}% · '
              '${processedFps.toStringAsFixed(1)} fps · '
              'elapsed ${_formatDuration(_elapsed)}',
              style: const TextStyle(
                fontFeatures: [FontFeature.tabularFigures()],
              ),
            ),
          ] else if (_outputPath != null && _statusMessage != null)
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        const Icon(Icons.check_circle, color: Colors.green),
                        const SizedBox(width: 8),
                        Expanded(child: Text(_statusMessage!)),
                      ],
                    ),
                    const SizedBox(height: 8),
                    Text(
                      'Total time: ${_formatDuration(_elapsed)} '
                      '(${processedFps.toStringAsFixed(1)} fps avg)',
                    ),
                    const SizedBox(height: 12),
                    _buildPreview(),
                    if (Platform.isMacOS ||
                        Platform.isLinux ||
                        Platform.isWindows) ...[
                      const SizedBox(height: 12),
                      ElevatedButton.icon(
                        onPressed: _openOutput,
                        icon: const Icon(Icons.play_circle_outline),
                        label: const Text('Open output video'),
                      ),
                    ],
                  ],
                ),
              ),
            )
          else
            Center(
              child: Column(
                children: [
                  const SizedBox(height: 32),
                  Icon(
                    Icons.movie_creation_outlined,
                    size: 96,
                    color: Colors.grey[400],
                  ),
                  const SizedBox(height: 16),
                  const Text(
                    'Pick an MP4 to run cat detection on every frame.\n'
                    'Output is written to the app documents directory.',
                    textAlign: TextAlign.center,
                  ),
                ],
              ),
            ),
        ],
      ),
    );
  }

  Widget _infoRow(String label, String value) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 2),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            SizedBox(
              width: 70,
              child: Text(
                '$label:',
                style: const TextStyle(fontWeight: FontWeight.w600),
              ),
            ),
            Expanded(child: SelectableText(value)),
          ],
        ),
      );

  String _formatDuration(Duration duration) {
    final minutes = duration.inMinutes.remainder(60).toString().padLeft(2, '0');
    final seconds = duration.inSeconds.remainder(60).toString().padLeft(2, '0');
    return duration.inHours > 0
        ? '${duration.inHours}:$minutes:$seconds'
        : '$minutes:$seconds';
  }

  Widget _buildPreview() {
    if (!_supportsInAppPlayer) return const SizedBox.shrink();
    if (_playerError != null) {
      return Text(_playerError!, style: const TextStyle(color: Colors.red));
    }
    final player = _playerController;
    if (player == null || !_playerReady) {
      return const SizedBox(
        height: 64,
        child: Center(child: CircularProgressIndicator()),
      );
    }
    return _OutputVideoPlayer(controller: player);
  }
}

class _OutputVideoPlayer extends StatefulWidget {
  const _OutputVideoPlayer({required this.controller});

  final VideoPlayerController controller;

  @override
  State<_OutputVideoPlayer> createState() => _OutputVideoPlayerState();
}

class _OutputVideoPlayerState extends State<_OutputVideoPlayer> {
  void _tick() {
    if (mounted) setState(() {});
  }

  @override
  void initState() {
    super.initState();
    widget.controller.addListener(_tick);
  }

  @override
  void dispose() {
    widget.controller.removeListener(_tick);
    super.dispose();
  }

  String _format(Duration duration) {
    final minutes = duration.inMinutes.remainder(60).toString().padLeft(2, '0');
    final seconds = duration.inSeconds.remainder(60).toString().padLeft(2, '0');
    return '$minutes:$seconds';
  }

  @override
  Widget build(BuildContext context) {
    final value = widget.controller.value;
    final maxHeight = math.max(120.0, MediaQuery.sizeOf(context).height * 0.45);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Align(
          alignment: Alignment.centerLeft,
          child: ConstrainedBox(
            constraints: BoxConstraints(maxHeight: maxHeight),
            child: AspectRatio(
              aspectRatio: value.aspectRatio == 0 ? 16 / 9 : value.aspectRatio,
              child: ClipRRect(
                borderRadius: BorderRadius.circular(8),
                child: VideoPlayer(widget.controller),
              ),
            ),
          ),
        ),
        Row(
          children: [
            IconButton(
              icon: Icon(value.isPlaying ? Icons.pause : Icons.play_arrow),
              onPressed: () => value.isPlaying
                  ? widget.controller.pause()
                  : widget.controller.play(),
            ),
            Expanded(
              child: VideoProgressIndicator(
                widget.controller,
                allowScrubbing: true,
                padding: const EdgeInsets.symmetric(vertical: 12),
              ),
            ),
            const SizedBox(width: 8),
            Text(
              '${_format(value.position)} / ${_format(value.duration)}',
              style: const TextStyle(
                fontFeatures: [FontFeature.tabularFigures()],
              ),
            ),
          ],
        ),
      ],
    );
  }
}

class _CatSmoother {
  bool enabled = true;
  final List<_CatTrack> _tracks = [];

  void reset() => _tracks.clear();

  List<Cat> apply(List<Cat> cats, double time) {
    if (!enabled) {
      _tracks.clear();
      return cats;
    }
    final unmatched = List<int>.generate(_tracks.length, (index) => index);
    final output = <Cat>[];
    for (final cat in cats) {
      var bestTrack = -1;
      var bestIou = 0.2;
      for (final index in unmatched) {
        final iou = _tracks[index].iou(cat.boundingBox);
        if (iou > bestIou) {
          bestIou = iou;
          bestTrack = index;
        }
      }
      final track = bestTrack < 0 ? _CatTrack() : _tracks[bestTrack];
      if (bestTrack < 0) {
        _tracks.add(track);
      } else {
        unmatched.remove(bestTrack);
      }
      track.update(cat.boundingBox);
      output.add(track.smooth(cat, time));
    }
    for (final index in unmatched) {
      _tracks[index].missed++;
    }
    _tracks.removeWhere((track) => track.missed > 5);
    return output;
  }
}

class _CatTrack {
  final Map<AnimalPoseLandmarkType, List<OneEuroFilter>> filters = {};
  final Map<CatLandmarkType, List<OneEuroFilter>> faceFilters = {};
  double left = 0;
  double top = 0;
  double right = 0;
  double bottom = 0;
  bool hasBox = false;
  int missed = 0;

  double iou(BoundingBox box) => hasBox
      ? iouLTRB(
          box.left,
          box.top,
          box.right,
          box.bottom,
          left,
          top,
          right,
          bottom,
        )
      : 0;

  void update(BoundingBox box) {
    left = box.left;
    top = box.top;
    right = box.right;
    bottom = box.bottom;
    hasBox = true;
    missed = 0;
  }

  Cat smooth(Cat cat, double time) {
    final pose = cat.pose;
    final smoothedPose = pose == null
        ? null
        : AnimalPose(
            landmarks: pose.landmarks.map((landmark) {
              final pair = filters.putIfAbsent(
                landmark.type,
                () => [
                  OneEuroFilter(minCutoff: 1, beta: 0.1, dCutoff: 1),
                  OneEuroFilter(minCutoff: 1, beta: 0.1, dCutoff: 1),
                ],
              );
              return AnimalPoseLandmark(
                type: landmark.type,
                x: pair[0].filter(landmark.x, time),
                y: pair[1].filter(landmark.y, time),
                confidence: landmark.confidence,
              );
            }).toList(),
          );
    final face = cat.face;
    final smoothedFace = face == null
        ? null
        : CatFace(
            boundingBox: face.boundingBox,
            landmarks: face.landmarks.map((landmark) {
              final pair = faceFilters.putIfAbsent(
                landmark.type,
                () => [
                  OneEuroFilter(minCutoff: 1, beta: 0.1, dCutoff: 1),
                  OneEuroFilter(minCutoff: 1, beta: 0.1, dCutoff: 1),
                ],
              );
              return CatLandmark(
                type: landmark.type,
                x: pair[0].filter(landmark.x, time),
                y: pair[1].filter(landmark.y, time),
              );
            }).toList(),
          );
    return Cat(
      boundingBox: cat.boundingBox,
      score: cat.score,
      species: cat.species,
      breed: cat.breed,
      speciesConfidence: cat.speciesConfidence,
      face: smoothedFace,
      pose: smoothedPose,
      imageWidth: cat.imageWidth,
      imageHeight: cat.imageHeight,
    );
  }
}
