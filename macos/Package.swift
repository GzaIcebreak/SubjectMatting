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
        // The matting process the app starts: JSON lines over stdin/stdout, libsubject_matting (cpp/) via dlopen.
        .executableTarget(name: "matting-worker", dependencies: ["MattingCore"], path: "Sources/MattingWorker"),
        .testTarget(name: "MattingCoreTests", dependencies: ["MattingCore"]),

        // SubjectMatting Q: Qwen-VL picks the subjects, libsubject_matting_q (cpp/) cuts them out. A separate app;
        // the SAM2 targets above are untouched.
        .target(name: "MattingQ", dependencies: ["MattingCore"]),
        .executableTarget(name: "matting-worker-q", dependencies: ["MattingCore", "MattingQ"], path: "Sources/MattingWorkerQ"),
        .executableTarget(name: "SubjectMattingQ", dependencies: ["MattingCore", "MattingQ"]),
        .testTarget(name: "MattingQTests", dependencies: ["MattingQ"]),
    ]
)
