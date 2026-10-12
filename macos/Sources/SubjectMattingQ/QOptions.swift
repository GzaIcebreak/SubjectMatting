import Foundation
import MattingQ
import Observation

/// The instruction typed in the toolbar, shared by drop, the open panel, Dock drops and "重新抠图".
@Observable
final class QOptions {
    static let shared = QOptions()
    var instruction = ""
}

/// The DashScope key, read from the Keychain once, off the main thread.
///
/// The app is ad-hoc signed, so after every update macOS asks again before letting it read the Keychain item
/// saved by the previous build. A synchronous read on the main thread would freeze the app behind that prompt.
@Observable
final class QKey {
    static let shared = QKey()

    private(set) var key: String?
    private(set) var loaded = false

    /// Reads the Keychain in the background, then calls `then` on the main thread.
    func load(then: @escaping () -> Void) {
        // Development: a key in the environment wins and the Keychain is not touched (no prompt).
        if let env = ProcessInfo.processInfo.environment["DASHSCOPE_API_KEY"], !env.isEmpty {
            key = env
            loaded = true
            then()
            return
        }
        DispatchQueue.global(qos: .userInitiated).async {
            let key = APIKeyStore.load()
            DispatchQueue.main.async {
                self.key = key
                self.loaded = true
                then()
            }
        }
    }

    /// Saves in the background; `then(ok)` on the main thread.
    func save(_ newKey: String, then: @escaping (Bool) -> Void) {
        DispatchQueue.global(qos: .userInitiated).async {
            let ok = APIKeyStore.save(newKey)
            DispatchQueue.main.async {
                if ok {
                    let t = newKey.trimmingCharacters(in: .whitespacesAndNewlines)
                    self.key = t.isEmpty ? nil : t
                }
                then(ok)
            }
        }
    }

    /// The worker's environment: the Keychain key, else whatever DASHSCOPE_API_KEY the app was started with (dev).
    func environment() -> [String: String] {
        key.map { ["DASHSCOPE_API_KEY": $0] } ?? [:]
    }

    var isConfigured: Bool {
        key != nil || !(ProcessInfo.processInfo.environment["DASHSCOPE_API_KEY"] ?? "").isEmpty
    }
}
