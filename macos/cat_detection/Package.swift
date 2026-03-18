// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "cat_detection",
    platforms: [
        .macOS("11.0")
    ],
    products: [
        .library(name: "cat-detection", targets: ["cat_detection"])
    ],
    dependencies: [],
    targets: [
        .target(
            name: "cat_detection",
            dependencies: [],
            resources: [
                .process("PrivacyInfo.xcprivacy"),
            ]
        )
    ]
)
