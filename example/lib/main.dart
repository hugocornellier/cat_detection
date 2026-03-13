import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:file_selector/file_selector.dart';
import 'package:cat_detection/cat_detection.dart';

void main() {
  runApp(const CatDetectionApp());
}

class CatDetectionApp extends StatelessWidget {
  const CatDetectionApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Cat Detection Demo',
      theme: ThemeData(
        colorSchemeSeed: Colors.purple,
        useMaterial3: true,
      ),
      home: const CatDetectionHome(),
    );
  }
}

class CatDetectionHome extends StatelessWidget {
  const CatDetectionHome({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Cat Detection Demo'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.pets, size: 100, color: Colors.purple[300]),
            const SizedBox(height: 48),
            Text(
              'Cat Detection',
              style: Theme.of(context).textTheme.headlineMedium,
            ),
            const SizedBox(height: 16),
            Text(
              'Detect cats with body pose and 48 facial landmarks',
              style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                    color: Colors.grey[600],
                  ),
            ),
            const SizedBox(height: 48),
            SizedBox(
              width: 400,
              child: Card(
                elevation: 4,
                child: InkWell(
                  onTap: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (context) => const StillImageScreen(),
                      ),
                    );
                  },
                  borderRadius: BorderRadius.circular(12),
                  child: Padding(
                    padding: const EdgeInsets.all(24),
                    child: Row(
                      children: [
                        const Icon(Icons.image, size: 64, color: Colors.purple),
                        const SizedBox(width: 24),
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                'Still Image',
                                style: Theme.of(context).textTheme.titleLarge,
                              ),
                              const SizedBox(height: 8),
                              Text(
                                'Detect cats in photos from gallery or camera',
                                style: Theme.of(context)
                                    .textTheme
                                    .bodyMedium
                                    ?.copyWith(color: Colors.grey[600]),
                              ),
                            ],
                          ),
                        ),
                        const Icon(Icons.arrow_forward_ios),
                      ],
                    ),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class StillImageScreen extends StatefulWidget {
  const StillImageScreen({super.key});

  @override
  State<StillImageScreen> createState() => _StillImageScreenState();
}

class _StillImageScreenState extends State<StillImageScreen> {
  CatDetectorIsolate? _detector;
  final ImagePicker _picker = ImagePicker();

  bool _useEnsemble = false;
  CatDetectionMode _detectionMode = CatDetectionMode.full;
  bool _isInitialized = false;
  bool _isProcessing = false;
  bool _isDownloading = false;
  String _downloadStatus = '';
  Uint8List? _imageBytes;
  int _imageWidth = 0;
  int _imageHeight = 0;
  List<Cat> _results = [];
  String? _errorMessage;

  @override
  void initState() {
    super.initState();
    _initializeDetector();
  }

  Future<void> _initializeDetector() async {
    setState(() {
      _isProcessing = true;
      _isInitialized = false;
      _errorMessage = null;
    });

    try {
      await _detector?.dispose();

      if (_useEnsemble) {
        final cached = await CatDetector.isEnsembleCached();
        if (!cached) {
          setState(() {
            _isDownloading = true;
            _downloadStatus = 'Downloading ensemble models...';
          });
        }
      }

      _detector = await CatDetectorIsolate.spawn(
        mode: _detectionMode,
        landmarkModel:
            _useEnsemble ? CatLandmarkModel.ensemble : CatLandmarkModel.full,
        performanceConfig: PerformanceConfig.disabled,
        onDownloadProgress: (model, received, total) {
          if (!mounted) return;
          final mb = (received / 1024 / 1024).toStringAsFixed(1);
          final totalMb =
              total > 0 ? (total / 1024 / 1024).toStringAsFixed(1) : '?';
          final name = model.contains('256') ? '256px' : '320px';
          setState(() {
            _downloadStatus = 'Downloading $name model: $mb / $totalMb MB';
          });
        },
      );

      if (!mounted) return;
      setState(() {
        _isInitialized = true;
        _isProcessing = false;
        _isDownloading = false;
        _downloadStatus = '';
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _isProcessing = false;
        _isDownloading = false;
        _downloadStatus = '';
        _errorMessage = 'Failed to initialize: $e';
      });
    }
  }

  Future<void> _toggleEnsemble(bool value) async {
    if (value == _useEnsemble) return;
    setState(() {
      _useEnsemble = value;
      _results = [];
    });
    await _initializeDetector();
    if (_imageBytes != null && _isInitialized) {
      await _runDetection(_imageBytes!);
    }
  }

  Future<void> _changeDetectionMode(CatDetectionMode mode) async {
    if (mode == _detectionMode) return;
    setState(() {
      _detectionMode = mode;
      _results = [];
    });
    await _initializeDetector();
    if (_imageBytes != null && _isInitialized) {
      await _runDetection(_imageBytes!);
    }
  }

  Future<void> _pickImage(ImageSource source) async {
    try {
      final XFile? pickedFile = await _picker.pickImage(source: source);
      if (pickedFile == null) return;

      final Uint8List bytes = await pickedFile.readAsBytes();
      await _runDetection(bytes);
    } catch (e) {
      setState(() {
        _isProcessing = false;
        _errorMessage = 'Error: $e';
      });
    }
  }

  Future<void> _pickFileFromSystem() async {
    try {
      const XTypeGroup typeGroup = XTypeGroup(
        label: 'images',
        extensions: ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'],
      );
      final XFile? file = await openFile(acceptedTypeGroups: [typeGroup]);
      if (file == null) return;

      final Uint8List bytes = await File(file.path).readAsBytes();
      await _runDetection(bytes);
    } catch (e) {
      setState(() {
        _isProcessing = false;
        _errorMessage = 'Error: $e';
      });
    }
  }

  bool get _isDesktop =>
      !kIsWeb && (Platform.isMacOS || Platform.isWindows || Platform.isLinux);

  Future<void> _runDetection(Uint8List bytes) async {
    setState(() {
      _isProcessing = true;
      _errorMessage = null;
      _results = [];
    });

    try {
      final List<Cat> results = await _detector!.detectCats(bytes);

      int imgW = 0;
      int imgH = 0;
      if (results.isNotEmpty) {
        imgW = results.first.imageWidth;
        imgH = results.first.imageHeight;
      } else {
        final decoded = await decodeImageFromList(bytes);
        imgW = decoded.width;
        imgH = decoded.height;
      }

      setState(() {
        _imageBytes = bytes;
        _imageWidth = imgW;
        _imageHeight = imgH;
        _results = results;
        _isProcessing = false;
        if (results.isEmpty) {
          _errorMessage = 'No cats detected in image';
        }
      });
    } catch (e) {
      setState(() {
        _isProcessing = false;
        _errorMessage = 'Error: $e';
      });
    }
  }

  @override
  void dispose() {
    _detector?.dispose();
    _detector = null;
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Cat Detection'),
        actions: [
          if (_isInitialized && _results.isNotEmpty)
            IconButton(
              icon: const Icon(Icons.info_outline),
              onPressed: _showDetectionInfo,
            ),
          IconButton(
            icon: const Icon(Icons.settings),
            onPressed: _showSettings,
          ),
        ],
      ),
      body: _buildBody(),
    );
  }

  Widget _buildBody() {
    if (_isDownloading) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 48),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const CircularProgressIndicator(),
              const SizedBox(height: 24),
              Text(
                _downloadStatus,
                textAlign: TextAlign.center,
                style: Theme.of(context).textTheme.bodyLarge,
              ),
              const SizedBox(height: 8),
              Text(
                'This is a one-time download. Models will be cached for future use.',
                textAlign: TextAlign.center,
                style: Theme.of(context)
                    .textTheme
                    .bodySmall
                    ?.copyWith(color: Colors.grey[600]),
              ),
            ],
          ),
        ),
      );
    }

    if (!_isInitialized && _isProcessing) {
      return const Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            CircularProgressIndicator(),
            SizedBox(height: 16),
            Text('Initializing cat detector...'),
          ],
        ),
      );
    }

    if (_errorMessage != null && _imageBytes == null) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.error_outline, size: 64, color: Colors.red),
            const SizedBox(height: 16),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 32),
              child: Text(
                _errorMessage!,
                textAlign: TextAlign.center,
                style: const TextStyle(color: Colors.red),
              ),
            ),
            const SizedBox(height: 16),
            ElevatedButton(
              onPressed: _initializeDetector,
              child: const Text('Retry'),
            ),
          ],
        ),
      );
    }

    if (_imageBytes == null) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.pets, size: 100, color: Colors.grey[400]),
            const SizedBox(height: 24),
            Text(
              'Select an image to detect cats',
              style: TextStyle(fontSize: 18, color: Colors.grey[600]),
            ),
            if (_useEnsemble)
              Padding(
                padding: const EdgeInsets.only(top: 8),
                child: Chip(
                  avatar: const Icon(Icons.auto_awesome, size: 16),
                  label: const Text('Ensemble mode'),
                  backgroundColor: Colors.amber[50],
                ),
              ),
            const SizedBox(height: 24),
            _buildActionButtons(),
          ],
        ),
      );
    }

    return SingleChildScrollView(
      child: Column(
        children: [
          CatVisualizerWidget(
            imageBytes: _imageBytes!,
            imageWidth: _imageWidth,
            imageHeight: _imageHeight,
            results: _results,
          ),
          if (_isProcessing)
            const Padding(
              padding: EdgeInsets.all(16),
              child: Column(
                children: [
                  CircularProgressIndicator(),
                  SizedBox(height: 8),
                  Text('Detecting cats...'),
                ],
              ),
            ),
          if (_errorMessage != null && !_isProcessing)
            Padding(
              padding: const EdgeInsets.all(16),
              child: Card(
                color: Colors.orange[50],
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Row(
                    children: [
                      const Icon(Icons.info_outline, color: Colors.orange),
                      const SizedBox(width: 8),
                      Expanded(child: Text(_errorMessage!)),
                    ],
                  ),
                ),
              ),
            ),
          if (_results.isNotEmpty)
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              child: Card(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          Expanded(
                            child: Text(
                              'Detected: ${_results.length} cat${_results.length > 1 ? 's' : ''}',
                              style: Theme.of(context)
                                  .textTheme
                                  .titleLarge
                                  ?.copyWith(
                                    color: Colors.green,
                                    fontWeight: FontWeight.bold,
                                  ),
                            ),
                          ),
                          if (_useEnsemble)
                            Chip(
                              avatar: const Icon(Icons.auto_awesome, size: 14),
                              label: const Text('Ensemble'),
                              visualDensity: VisualDensity.compact,
                              backgroundColor: Colors.amber[50],
                            ),
                        ],
                      ),
                      const SizedBox(height: 8),
                      for (final cat in _results) ...[
                        Text(
                          'Score: ${(cat.score * 100).toStringAsFixed(1)}%',
                          style: Theme.of(context).textTheme.bodyMedium,
                        ),
                        if (cat.species != null)
                          Text(
                            'Species: ${cat.species}',
                            style: Theme.of(context).textTheme.bodyMedium,
                          ),
                        if (cat.pose != null)
                          Text(
                            'Pose landmarks: ${cat.pose!.landmarks.length}',
                            style: Theme.of(context).textTheme.bodyMedium,
                          ),
                        if (cat.face != null && cat.face!.hasLandmarks)
                          Text(
                            'Face landmarks: ${cat.face!.landmarks.length}',
                            style: Theme.of(context).textTheme.bodyMedium,
                          ),
                        if (_results.length > 1 && cat != _results.last)
                          const Divider(),
                      ],
                    ],
                  ),
                ),
              ),
            ),
          Padding(
            padding: const EdgeInsets.all(16),
            child: _buildActionButtons(),
          ),
        ],
      ),
    );
  }

  Widget _buildActionButtons() {
    return Wrap(
      spacing: 12,
      runSpacing: 12,
      alignment: WrapAlignment.center,
      children: [
        ElevatedButton.icon(
          onPressed: _isInitialized && !_isProcessing
              ? () => _isDesktop
                  ? _pickFileFromSystem()
                  : _pickImage(ImageSource.gallery)
              : null,
          icon: const Icon(Icons.photo_library),
          label: Text(_isDesktop ? 'Open File' : 'Gallery'),
        ),
        if (!_isDesktop)
          ElevatedButton.icon(
            onPressed: _isInitialized && !_isProcessing
                ? () => _pickImage(ImageSource.camera)
                : null,
            icon: const Icon(Icons.camera_alt),
            label: const Text('Camera'),
          ),
      ],
    );
  }

  void _showSettings() {
    showModalBottomSheet(
      context: context,
      builder: (context) => StatefulBuilder(
        builder: (context, setSheetState) => Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'Settings',
                style: Theme.of(context).textTheme.headlineSmall,
              ),
              const SizedBox(height: 24),
              Text(
                'Detection Mode',
                style: Theme.of(context).textTheme.titleMedium,
              ),
              const SizedBox(height: 8),
              RadioGroup<CatDetectionMode>(
                groupValue: _detectionMode,
                onChanged: _isDownloading
                    ? (_) {}
                    : (value) {
                        if (value == null) return;
                        setSheetState(() {});
                        Navigator.pop(context);
                        _changeDetectionMode(value);
                      },
                child: Column(
                  children: [
                    for (final mode in CatDetectionMode.values)
                      RadioListTile<CatDetectionMode>(
                        title: Text(_modeLabel(mode)),
                        subtitle: Text(_modeDescription(mode)),
                        value: mode,
                      ),
                  ],
                ),
              ),
              const Divider(),
              SwitchListTile(
                title: const Text('Ensemble mode'),
                subtitle: Text(
                  _useEnsemble
                      ? '3-model ensemble (256+320+384px) for ~8% better accuracy'
                      : 'Single 384px model (default)',
                ),
                secondary: const Icon(Icons.auto_awesome),
                value: _useEnsemble,
                onChanged: (_detectionMode == CatDetectionMode.poseOnly ||
                        _isDownloading)
                    ? null
                    : (value) {
                        setSheetState(() {});
                        Navigator.pop(context);
                        _toggleEnsemble(value);
                      },
              ),
              Padding(
                padding: const EdgeInsets.only(left: 72, top: 4),
                child: Text(
                  _useEnsemble
                      ? 'Extra models are cached locally after first download.'
                      : 'Enable to download two extra models (~110 MB total) for improved landmark accuracy.',
                  style: Theme.of(context)
                      .textTheme
                      .bodySmall
                      ?.copyWith(color: Colors.grey[600]),
                ),
              ),
              const SizedBox(height: 24),
            ],
          ),
        ),
      ),
    );
  }

  String _modeLabel(CatDetectionMode mode) {
    switch (mode) {
      case CatDetectionMode.full:
        return 'Full';
      case CatDetectionMode.poseOnly:
        return 'Pose Only';
    }
  }

  String _modeDescription(CatDetectionMode mode) {
    switch (mode) {
      case CatDetectionMode.full:
        return 'Body detection + species + pose + face landmarks';
      case CatDetectionMode.poseOnly:
        return 'Body detection + species + body pose only';
    }
  }

  void _showDetectionInfo() {
    if (_results.isEmpty) return;

    showModalBottomSheet(
      context: context,
      builder: (context) => DraggableScrollableSheet(
        initialChildSize: 0.6,
        minChildSize: 0.4,
        maxChildSize: 0.95,
        expand: false,
        builder: (context, scrollController) => ListView(
          controller: scrollController,
          padding: const EdgeInsets.all(16),
          children: [
            Text(
              'Detection Details',
              style: Theme.of(context).textTheme.headlineSmall,
            ),
            const SizedBox(height: 16),
            for (final cat in _results) ...[
              Text(
                'Cat (score: ${(cat.score * 100).toStringAsFixed(1)}%)',
                style: Theme.of(context).textTheme.titleMedium?.copyWith(
                      fontWeight: FontWeight.bold,
                    ),
              ),
              if (cat.species != null)
                Padding(
                  padding: const EdgeInsets.only(top: 4, bottom: 4),
                  child: Text('Species: ${cat.species}'),
                ),
              if (cat.pose != null && cat.pose!.hasLandmarks) ...[
                const SizedBox(height: 8),
                Text(
                  'Body Pose (${cat.pose!.landmarks.length} keypoints)',
                  style: Theme.of(context).textTheme.titleSmall,
                ),
                ...cat.pose!.landmarks.map((lm) => Card(
                      margin: const EdgeInsets.only(bottom: 4),
                      child: ListTile(
                        dense: true,
                        leading: CircleAvatar(
                          radius: 14,
                          backgroundColor: Colors.red,
                          child: Text(
                            lm.type.index.toString(),
                            style: const TextStyle(
                                fontSize: 9, color: Colors.white),
                          ),
                        ),
                        title: Text(
                          lm.type.name,
                          style: const TextStyle(fontWeight: FontWeight.w500),
                        ),
                        subtitle: Text(
                          'Position: (${lm.x.toStringAsFixed(1)}, ${lm.y.toStringAsFixed(1)})  conf: ${(lm.confidence * 100).toStringAsFixed(0)}%',
                        ),
                      ),
                    )),
              ],
              if (cat.face != null && cat.face!.hasLandmarks) ...[
                const SizedBox(height: 8),
                Text(
                  'Face Landmarks (${cat.face!.landmarks.length} keypoints)',
                  style: Theme.of(context).textTheme.titleSmall,
                ),
                ...cat.face!.landmarks.map((lm) => Card(
                      margin: const EdgeInsets.only(bottom: 4),
                      child: ListTile(
                        dense: true,
                        leading: CircleAvatar(
                          radius: 14,
                          backgroundColor: _landmarkColor(lm.type),
                          child: Text(
                            lm.type.index.toString(),
                            style: const TextStyle(
                                fontSize: 9, color: Colors.white),
                          ),
                        ),
                        title: Text(
                          lm.type.name,
                          style: const TextStyle(fontWeight: FontWeight.w500),
                        ),
                        subtitle: Text(
                          'Position: (${lm.x.toStringAsFixed(1)}, ${lm.y.toStringAsFixed(1)})',
                        ),
                      ),
                    )),
              ],
              if (cat != _results.last) const Divider(height: 24),
            ],
          ],
        ),
      ),
    );
  }

  Color _landmarkColor(CatLandmarkType type) {
    final String name = type.name;
    if (name.startsWith('leftEar') || name.startsWith('rightEar')) {
      return Colors.blue;
    } else if (name.startsWith('leftEye') || name.startsWith('rightEye')) {
      return Colors.green;
    } else if (name.startsWith('nose') || name.startsWith('noseRing') ||
        name.startsWith('noseBridge') || name.startsWith('noseTip') ||
        name.startsWith('noseWing')) {
      return Colors.orange;
    } else {
      return Colors.yellow;
    }
  }
}

class CatVisualizerWidget extends StatelessWidget {
  final Uint8List imageBytes;
  final int imageWidth;
  final int imageHeight;
  final List<Cat> results;

  const CatVisualizerWidget({
    super.key,
    required this.imageBytes,
    required this.imageWidth,
    required this.imageHeight,
    required this.results,
  });

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(builder: (context, constraints) {
      return Stack(
        children: [
          Image.memory(imageBytes, fit: BoxFit.contain),
          Positioned.fill(
            child: CustomPaint(
              painter: CatOverlayPainter(
                results: results,
                imageWidth: imageWidth,
                imageHeight: imageHeight,
              ),
            ),
          ),
        ],
      );
    });
  }
}

class CatOverlayPainter extends CustomPainter {
  final List<Cat> results;
  final int imageWidth;
  final int imageHeight;

  CatOverlayPainter({
    required this.results,
    required this.imageWidth,
    required this.imageHeight,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (results.isEmpty || imageWidth == 0 || imageHeight == 0) return;

    final double imageAspect = imageWidth / imageHeight;
    final double canvasAspect = size.width / size.height;
    double scaleX, scaleY;
    double offsetX = 0, offsetY = 0;

    if (canvasAspect > imageAspect) {
      scaleY = size.height / imageHeight;
      scaleX = scaleY;
      offsetX = (size.width - imageWidth * scaleX) / 2;
    } else {
      scaleX = size.width / imageWidth;
      scaleY = scaleX;
      offsetY = (size.height - imageHeight * scaleY) / 2;
    }

    for (final cat in results) {
      _drawBodyBoundingBox(canvas, cat, scaleX, scaleY, offsetX, offsetY);
      _drawSpeciesLabel(canvas, cat, scaleX, scaleY, offsetX, offsetY);

      if (cat.pose != null && cat.pose!.hasLandmarks) {
        _drawBodySkeleton(canvas, cat, scaleX, scaleY, offsetX, offsetY);
        _drawBodyKeypoints(canvas, cat, scaleX, scaleY, offsetX, offsetY);
      }

      if (cat.face != null) {
        _drawFaceBoundingBox(
            canvas, cat.face!, scaleX, scaleY, offsetX, offsetY);
        if (cat.face!.hasLandmarks) {
          _drawFaceConnections(
              canvas, cat.face!, scaleX, scaleY, offsetX, offsetY);
          _drawFaceLandmarks(
              canvas, cat.face!, scaleX, scaleY, offsetX, offsetY);
        }
      }
    }
  }

  void _drawBodyBoundingBox(Canvas canvas, Cat cat, double scaleX,
      double scaleY, double offsetX, double offsetY) {
    final Paint strokePaint = Paint()
      ..color = Colors.orange.withValues(alpha: 0.9)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3;

    final Paint fillPaint = Paint()
      ..color = Colors.orange.withValues(alpha: 0.08)
      ..style = PaintingStyle.fill;

    final double x1 = cat.boundingBox.left * scaleX + offsetX;
    final double y1 = cat.boundingBox.top * scaleY + offsetY;
    final double x2 = cat.boundingBox.right * scaleX + offsetX;
    final double y2 = cat.boundingBox.bottom * scaleY + offsetY;
    final Rect rect = Rect.fromLTRB(x1, y1, x2, y2);
    canvas.drawRect(rect, fillPaint);
    canvas.drawRect(rect, strokePaint);
  }

  void _drawSpeciesLabel(Canvas canvas, Cat cat, double scaleX, double scaleY,
      double offsetX, double offsetY) {
    if (cat.species == null) return;

    final double x1 = cat.boundingBox.left * scaleX + offsetX;
    final double y1 = cat.boundingBox.top * scaleY + offsetY;

    final String breedInfo = cat.breed != null && cat.speciesConfidence != null
        ? ' (${cat.breed}, ${(cat.speciesConfidence! * 100).toStringAsFixed(0)}%)'
        : '';
    final String label = '${cat.species}$breedInfo';
    final TextPainter textPainter = TextPainter(
      text: TextSpan(
        text: label,
        style: const TextStyle(
          color: Colors.white,
          fontSize: 12,
          fontWeight: FontWeight.bold,
        ),
      ),
      textDirection: TextDirection.ltr,
    );
    textPainter.layout();

    final double padding = 4;
    final double labelY = y1 - textPainter.height - padding * 2;
    final Rect bgRect = Rect.fromLTWH(
      x1,
      labelY,
      textPainter.width + padding * 2,
      textPainter.height + padding * 2,
    );

    canvas.drawRect(
      bgRect,
      Paint()..color = Colors.green.withValues(alpha: 0.85),
    );
    textPainter.paint(canvas, Offset(x1 + padding, labelY + padding));
  }

  void _drawBodySkeleton(Canvas canvas, Cat cat, double scaleX, double scaleY,
      double offsetX, double offsetY) {
    final Paint posePaint = Paint()
      ..color = Colors.orange.withValues(alpha: 0.8)
      ..strokeWidth = 2.5
      ..strokeCap = StrokeCap.round;

    for (final bone in animalPoseConnections) {
      final start = cat.pose!.getLandmark(bone[0]);
      final end = cat.pose!.getLandmark(bone[1]);
      if (start != null && end != null) {
        canvas.drawLine(
          Offset(start.x * scaleX + offsetX, start.y * scaleY + offsetY),
          Offset(end.x * scaleX + offsetX, end.y * scaleY + offsetY),
          posePaint,
        );
      }
    }
  }

  void _drawBodyKeypoints(Canvas canvas, Cat cat, double scaleX, double scaleY,
      double offsetX, double offsetY) {
    for (final lm in cat.pose!.landmarks) {
      final Offset center =
          Offset(lm.x * scaleX + offsetX, lm.y * scaleY + offsetY);
      canvas.drawCircle(center, 5, Paint()..color = Colors.red);
      canvas.drawCircle(center, 2, Paint()..color = Colors.white);
    }
  }

  void _drawFaceBoundingBox(Canvas canvas, CatFace face, double scaleX,
      double scaleY, double offsetX, double offsetY) {
    final Paint strokePaint = Paint()
      ..color = Colors.cyan.withValues(alpha: 0.9)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2;

    final double x1 = face.boundingBox.left * scaleX + offsetX;
    final double y1 = face.boundingBox.top * scaleY + offsetY;
    final double x2 = face.boundingBox.right * scaleX + offsetX;
    final double y2 = face.boundingBox.bottom * scaleY + offsetY;
    canvas.drawRect(Rect.fromLTRB(x1, y1, x2, y2), strokePaint);
  }

  void _drawFaceConnections(Canvas canvas, CatFace face, double scaleX,
      double scaleY, double offsetX, double offsetY) {
    final Paint paint = Paint()
      ..color = Colors.white.withValues(alpha: 0.7)
      ..strokeWidth = 2
      ..strokeCap = StrokeCap.round;

    for (final connection in catLandmarkConnections) {
      final CatLandmark? start = face.getLandmark(connection[0]);
      final CatLandmark? end = face.getLandmark(connection[1]);
      if (start != null && end != null) {
        canvas.drawLine(
          Offset(start.x * scaleX + offsetX, start.y * scaleY + offsetY),
          Offset(end.x * scaleX + offsetX, end.y * scaleY + offsetY),
          paint,
        );
      }
    }
  }

  void _drawFaceLandmarks(Canvas canvas, CatFace face, double scaleX,
      double scaleY, double offsetX, double offsetY) {
    for (final lm in face.landmarks) {
      final Offset center =
          Offset(lm.x * scaleX + offsetX, lm.y * scaleY + offsetY);
      final Color color = _landmarkColor(lm.type);

      final Paint glowPaint = Paint()..color = color.withValues(alpha: 0.3);
      final Paint dotPaint = Paint()..color = color;
      final Paint centerPaint = Paint()..color = Colors.white;

      canvas.drawCircle(center, 7, glowPaint);
      canvas.drawCircle(center, 4, dotPaint);
      canvas.drawCircle(center, 1.5, centerPaint);
    }
  }

  Color _landmarkColor(CatLandmarkType type) {
    final String name = type.name;
    if (name.startsWith('leftEar') || name.startsWith('rightEar')) {
      return Colors.blue;
    } else if (name.startsWith('leftEye') || name.startsWith('rightEye')) {
      return Colors.green;
    } else if (name.startsWith('nose') || name.startsWith('noseRing') ||
        name.startsWith('noseBridge') || name.startsWith('noseTip') ||
        name.startsWith('noseWing')) {
      return Colors.orange;
    } else {
      return Colors.yellow;
    }
  }

  @override
  bool shouldRepaint(CatOverlayPainter oldDelegate) => true;
}
