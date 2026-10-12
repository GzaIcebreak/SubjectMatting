import Foundation

/// Subject detection with Qwen-VL over DashScope's OpenAI-compatible API. Port of matting/qwen.py: same prompts,
/// same lenient parsing, boxes in Qwen's 0~1000 coordinates converted to pixels of the original image.
public struct QwenSubject: Equatable, Sendable {
    public var label: String
    public var box: [Double]  // x1, y1, x2, y2 in pixels

    public init(label: String, box: [Double]) {
        self.label = label
        self.box = box
    }
}

public struct QwenError: Error, Equatable, CustomStringConvertible {
    public var description: String
    public init(_ description: String) { self.description = description }
}

public final class QwenClient: @unchecked Sendable {
    public static let defaultBaseURL = URL(string: "https://dashscope.aliyuncs.com/compatible-mode/v1")!
    public static let defaultModel = "qwen3-vl-plus"

    static let autoPrompt = """
    找出这张图里，一般人想抠出来的前景主体。可以有多个（比如人和他牵着的狗），但背景、地面、桌面、墙、天空不算主体。
    只输出 JSON，不要任何解释：
    {"subjects": [{"label": "...", "box_2d": [x1, y1, x2, y2]}]}
    - label：主体的英文名词短语，1~3 个词，单数，例如 "cat"、"red car"、"teddy bear"。
    - box_2d：主体的外接框，坐标是相对图片宽高的 0~1000 整数，左上角 (x1, y1)，右下角 (x2, y2)。框要完整包住整个主体。
    """

    static func instructionPrompt(_ instruction: String) -> String {
        """
        用户想从这张图里抠出：「\(instruction)」。
        找出图中符合这个描述的对象（可能有多个，也可能没有）。只输出 JSON，不要任何解释：
        {"subjects": [{"label": "...", "box_2d": [x1, y1, x2, y2]}]}
        - label：对象的英文名词短语，1~3 个词，单数，例如 "cat"、"shopping bag"。不要把位置写进 label（"左边的猫" 写成 "cat"，靠 box_2d 区分是哪一只）。
        - box_2d：对象的外接框，坐标是相对图片宽高的 0~1000 整数，左上角 (x1, y1)，右下角 (x2, y2)。框要完整包住整个对象。
        - 图中没有符合描述的对象时输出 {"subjects": []}。
        """
    }

    let apiKey: String
    let model: String
    let baseURL: URL
    let timeout: TimeInterval

    public init(apiKey: String, model: String = defaultModel, baseURL: URL = defaultBaseURL, timeout: TimeInterval = 60) {
        self.apiKey = apiKey
        self.model = model
        self.baseURL = baseURL
        self.timeout = timeout
    }

    /// jpeg: the image (already scaled down for upload). width/height: the original image, for the pixel boxes.
    /// Blocks until the reply arrives (the worker processes one image at a time).
    public func detect(jpeg: Data, width: Int, height: Int, instruction: String?) throws -> [QwenSubject] {
        let prompt = instruction.map(Self.instructionPrompt) ?? Self.autoPrompt
        let body: [String: Any] = [
            "model": model,
            "messages": [[
                "role": "user",
                "content": [
                    ["type": "image_url", "image_url": ["url": "data:image/jpeg;base64," + jpeg.base64EncodedString()]],
                    ["type": "text", "text": prompt],
                ],
            ]],
            "temperature": 0,
            // Hybrid-thinking models would otherwise spend seconds reasoning about a bounding box.
            "enable_thinking": false,
        ]
        var request = URLRequest(url: baseURL.appendingPathComponent("chat/completions"), timeoutInterval: timeout)
        request.httpMethod = "POST"
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, status) = try Self.send(request)
        guard status == 200 else { throw Self.httpError(status, data) }
        guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let choices = obj["choices"] as? [[String: Any]],
              let message = choices.first?["message"] as? [String: Any],
              let content = message["content"] as? String
        else { throw QwenError("Qwen 接口返回格式异常：\(String(decoding: data.prefix(300), as: UTF8.self))") }
        return try Self.parseSubjects(content, width: width, height: height)
    }

    static func send(_ request: URLRequest) throws -> (Data, Int) {
        let done = DispatchSemaphore(value: 0)
        var result: Result<(Data, Int), Error> = .failure(QwenError("无法连接阿里云百炼"))
        URLSession.shared.dataTask(with: request) { data, response, error in
            if let error {
                result = .failure(QwenError("无法连接阿里云百炼：\(error.localizedDescription)"))
            } else {
                result = .success((data ?? Data(), (response as? HTTPURLResponse)?.statusCode ?? 0))
            }
            done.signal()
        }.resume()
        done.wait()
        return try result.get()
    }

    static func httpError(_ status: Int, _ data: Data) -> QwenError {
        let detail = String(decoding: data.prefix(300), as: UTF8.self)
        switch status {
        case 401, 403: return QwenError("阿里云百炼 API Key 无效或没有权限（\(status)），请在「设置」里检查")
        case 429: return QwenError("阿里云百炼请求太频繁或额度不足（429）：\(detail)")
        default: return QwenError("Qwen 接口返回 \(status)：\(detail)")
        }
    }

    /// The model's reply -> subjects with pixel boxes. Accepts {"subjects": [...]} as asked, or the bare
    /// [{"bbox_2d": ..., "label": ...}] list some Qwen models fall back to; ```json fences / surrounding text are ok.
    public static func parseSubjects(_ text: String, width: Int, height: Int) throws -> [QwenSubject] {
        guard let start = text.firstIndex(where: { $0 == "{" || $0 == "[" }) else {
            throw QwenError("模型没有返回 JSON：\(text.prefix(200))")
        }
        let tail = String(text[start...])
        guard let json = firstJSONValue(tail), let data = json.data(using: .utf8),
              let value = try? JSONSerialization.jsonObject(with: data)
        else { throw QwenError("模型返回的 JSON 无法解析：\(tail.prefix(200))") }
        let items: [Any]
        if let list = value as? [Any] {
            items = list
        } else if let dict = value as? [String: Any] {
            items = dict["subjects"] as? [Any] ?? []
        } else {
            items = []
        }
        var subjects: [QwenSubject] = []
        for case let s as [String: Any] in items {
            let label = (s["label"] as? String ?? "").trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            let raw = (s["box_2d"] ?? s["bbox_2d"] ?? s["box"]) as? [Any] ?? []
            let nums = raw.compactMap { ($0 as? NSNumber)?.doubleValue }
            guard !label.isEmpty, nums.count == 4 else { continue }
            let clamp = { (v: Double) in min(max(v, 0), 1000) }
            let x1 = clamp(min(nums[0], nums[2])), x2 = clamp(max(nums[0], nums[2]))
            let y1 = clamp(min(nums[1], nums[3])), y2 = clamp(max(nums[1], nums[3]))
            let px = [x1 / 1000 * Double(width), y1 / 1000 * Double(height), x2 / 1000 * Double(width), y2 / 1000 * Double(height)]
            if px[2] - px[0] >= 2, px[3] - px[1] >= 2 {
                subjects.append(QwenSubject(label: label, box: px))
            }
        }
        return subjects
    }

    /// The first complete JSON object/array at the start of `s` (bracket matching that respects strings).
    static func firstJSONValue(_ s: String) -> String? {
        var depth = 0
        var inString = false
        var escaped = false
        for (i, c) in s.enumerated() {
            if inString {
                if escaped { escaped = false } else if c == "\\" { escaped = true } else if c == "\"" { inString = false }
                continue
            }
            switch c {
            case "\"": inString = true
            case "{", "[": depth += 1
            case "}", "]":
                depth -= 1
                if depth == 0 { return String(s.prefix(i + 1)) }
            default: break
            }
        }
        return nil
    }
}
