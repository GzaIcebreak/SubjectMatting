import Foundation

/// Conservative support check, independent of the visual reviewer's opinion.
/// Enclosed holes are filled ONLY for measuring support; output alpha is never changed.
enum ExteriorGrowth {
    struct Evidence {
        let addedShare: Double
        let addedMass: Double
        let tolerancePixels: Int
        let veto: Bool
    }

    static func measure(original: [UInt8], detail: [UInt8], width: Int, height: Int) -> Evidence {
        // Bound scratch memory for full-resolution photographs. Any-foreground pooling
        // supports local edges; all-foreground pooling preserves open background channels.
        // Averaging alpha measures coverage rather than thresholded binary area.
        let scale = min(1.0, 1024.0 / Double(max(width, height)))
        let w = max(1, Int(Double(width) * scale)), h = max(1, Int(Double(height) * scale))
        let n = w * h
        var support = [Bool](repeating: false, count: n)
        var fullSupport = [Bool](repeating: true, count: n)
        var a = [Float](repeating: 0, count: n), b = a
        var counts = [Int](repeating: 0, count: n)
        for y in 0..<height {
            let gy = y * h / height
            for x in 0..<width {
                let g = gy * w + x * w / width, k = (y * width + x) * 4 + 3
                support[g] = support[g] || original[k] >= 32
                fullSupport[g] = fullSupport[g] && original[k] >= 32
                a[g] += Float(original[k]); b[g] += Float(detail[k]); counts[g] += 1
            }
        }
        for i in 0..<n {
            a[i] /= Float(max(1, counts[i])) * 255
            b[i] /= Float(max(1, counts[i])) * 255
        }

        var exterior = [Bool](repeating: false, count: n)
        var queue = [Int](); queue.reserveCapacity(n)
        func visitExterior(_ i: Int) {
            if !fullSupport[i] && !exterior[i] { exterior[i] = true; queue.append(i) }
        }
        for x in 0..<w { visitExterior(x); visitExterior((h - 1) * w + x) }
        for y in 0..<h { visitExterior(y * w); visitExterior(y * w + w - 1) }
        var head = 0
        while head < queue.count {
            let i = queue[head]; head += 1
            let x = i % w, y = i / w
            if x > 0 { visitExterior(i - 1) }
            if x + 1 < w { visitExterior(i + 1) }
            if y > 0 { visitExterior(i - w) }
            if y + 1 < h { visitExterior(i + w) }
        }

        // A bounded Chebyshev band tolerates registration/edge changes, not whole new parts.
        let radius = max(2, Int((0.01 * Double(min(w, h))).rounded()))
        var distance = [Int16](repeating: Int16.max, count: n)
        queue.removeAll(keepingCapacity: true)
        for i in 0..<n where support[i] || !exterior[i] { distance[i] = 0; queue.append(i) }
        head = 0
        while head < queue.count {
            let i = queue[head]; head += 1
            guard distance[i] < radius else { continue }
            let x = i % w, y = i / w, next = distance[i] + 1
            for yy in max(0, y - 1)...min(h - 1, y + 1) {
                for xx in max(0, x - 1)...min(w - 1, x + 1) {
                    let j = yy * w + xx
                    if distance[j] > next { distance[j] = next; queue.append(j) }
                }
            }
        }
        var originalMass = 0.0, added = 0.0
        for i in 0..<n {
            originalMass += Double(a[i])
            if distance[i] > radius { added += max(0, Double(b[i] - a[i])) }
        }
        let share = added / max(1, originalMass)
        return Evidence(addedShare: share, addedMass: added, tolerancePixels: radius,
                        veto: added > 4 && share > 0.02)
    }
}
