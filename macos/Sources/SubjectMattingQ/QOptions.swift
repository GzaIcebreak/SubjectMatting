import Foundation
import MattingQ
import Observation

/// The instruction typed in the toolbar, shared by drop, the open panel, Dock drops and "重新抠图".
@Observable
final class QOptions {
    static let shared = QOptions()
    var instruction = ""
}

enum QKey {
    /// The worker's environment: the Keychain key, else whatever DASHSCOPE_API_KEY the app was started with (dev).
    static func environment() -> [String: String] {
        if let key = APIKeyStore.load() { return ["DASHSCOPE_API_KEY": key] }
        return [:]
    }

    static var isConfigured: Bool {
        APIKeyStore.load() != nil || !(ProcessInfo.processInfo.environment["DASHSCOPE_API_KEY"] ?? "").isEmpty
    }
}
