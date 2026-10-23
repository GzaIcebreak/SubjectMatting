import Foundation
import Security

/// The DashScope API key in the user's login Keychain (generic password). Never written to files or logs.
public enum APIKeyStore {
    static let service = "com.gzaicebreak.subjectmatting.q"
    static let account = "dashscope-api-key"

    public static func load() -> String? {
        let query: [CFString: Any] = [
            kSecClass: kSecClassGenericPassword,
            kSecAttrService: service,
            kSecAttrAccount: account,
            kSecReturnData: true,
            kSecMatchLimit: kSecMatchLimitOne,
        ]
        var item: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &item) == errSecSuccess, let data = item as? Data else { return nil }
        let key = String(decoding: data, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
        return key.isEmpty ? nil : key
    }

    /// Saves (or with an empty key, deletes) the key. Returns false when the Keychain refused.
    @discardableResult
    public static func save(_ key: String) -> Bool {
        let base: [CFString: Any] = [kSecClass: kSecClassGenericPassword, kSecAttrService: service, kSecAttrAccount: account]
        SecItemDelete(base as CFDictionary)
        let trimmed = key.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { return true }
        var add = base
        add[kSecValueData] = Data(trimmed.utf8)
        add[kSecAttrLabel] = "抠图demo：阿里云百炼 API Key"
        return SecItemAdd(add as CFDictionary, nil) == errSecSuccess
    }
}
