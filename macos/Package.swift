// swift-tools-version:5.10
import PackageDescription

let package = Package(
    name: "SubjectMatting",
    platforms: [.macOS(.v14)],
    targets: [
        // Worker protocol, process management and the job queue: no UI, unit tested.
        .target(name: "MattingCore"),
        // The SwiftUI app. scripts/build_macos_app.sh wraps the binary in SubjectMatting.app.
        .executableTarget(name: "SubjectMatting", dependencies: ["MattingCore"]),
        .testTarget(name: "MattingCoreTests", dependencies: ["MattingCore"]),
    ]
)
