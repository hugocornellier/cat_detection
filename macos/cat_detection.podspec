Pod::Spec.new do |s|
  s.name                  = 'cat_detection'
  s.version               = '0.0.1'
  s.summary               = 'Cat detection via TensorFlow Lite (macOS)'
  s.description           = 'Flutter plugin for on-device cat face detection using TensorFlow Lite.'
  s.homepage              = 'https://github.com/hugocornellier/cat_detection'
  s.license               = { :type => 'MIT' }
  s.authors               = { 'Hugo Cornellier' => 'hugo@example.com' }
  s.source                = { :path => '.' }

  s.platform              = :osx, '11.0'
  s.swift_version         = '5.0'

  s.source_files          = 'cat_detection/Sources/cat_detection/**/*'
  s.resource_bundles      = { 'cat_detection_privacy' => ['cat_detection/Sources/cat_detection/PrivacyInfo.xcprivacy'] }

  s.dependency            'FlutterMacOS'
  s.static_framework      = true
end
