#include "pipeline.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace sm {

namespace {

constexpr int kSamSize = 1024;
constexpr int kSamMaskSize = 256;
constexpr double kMinInside = 0.6;
constexpr float kMinStability = 0.4f;
constexpr double kSafetyPad = 0.15;
const float kMean[3] = {0.485f, 0.456f, 0.406f};
const float kStd[3] = {0.229f, 0.224f, 0.225f};

double now() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// (1,3,h,w) float32: PIL bilinear resize, /255, ImageNet normalization.
std::vector<float> imagenet_tensor(const ImageU8 &img, int w, int h) {
    ImageU8 r = pil_resize(img, w, h, Filter::Bilinear);
    std::vector<float> t(size_t(3) * w * h);
    const size_t plane = size_t(w) * h;
    for (size_t i = 0; i < plane; i++) {
        for (int c = 0; c < 3; c++) {
            float x = float(r.data[i * 3 + c]) / 255.f;
            t[c * plane + i] = (x - kMean[c]) / kStd[c];
        }
    }
    return t;
}

bool touches_border(const Mask &m) {
    for (int x = 0; x < m.w; x++)
        if (m.at(x, 0) || m.at(x, m.h - 1)) return true;
    for (int y = 0; y < m.h; y++)
        if (m.at(0, y) || m.at(m.w - 1, y)) return true;
    return false;
}

// np.linspace(0, n - 1, num).round().astype(int)
std::vector<long> linspace_round(long n, long num) {
    std::vector<long> out(num);
    if (num == 1) {
        out[0] = 0;
        return out;
    }
    double step = double(n - 1) / double(num - 1);
    for (long i = 0; i < num; i++) out[i] = round_half_even(double(i) * step);
    out[num - 1] = n - 1;
    return out;
}

#ifdef _WIN32
std::wstring to_ort_path(const std::string &s) { return std::wstring(s.begin(), s.end()); }
#else
const std::string &to_ort_path(const std::string &s) { return s; }
#endif

}  // namespace

// ---------------------------------------------------------------------------------------------------------------
// Prompt and mask logic (matting/pipeline_a.py, matting/common.py)

std::vector<Mask> largest_components(const Mask &m, double min_ratio) {
    std::vector<int32_t> labels;
    std::vector<int64_t> areas;
    int n = connected_components(m, labels, &areas);
    if (n <= 1) return {};
    int64_t max_area = *std::max_element(areas.begin() + 1, areas.end());
    std::vector<int> order(n - 1);
    std::iota(order.begin(), order.end(), 1);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return areas[a] > areas[b]; });
    std::vector<Mask> out;
    for (int l : order) {
        if (double(areas[l]) < min_ratio * double(max_area)) continue;
        Mask c(m.w, m.h);
        for (size_t i = 0; i < labels.size(); i++) c.data[i] = labels[i] == l;
        out.push_back(std::move(c));
    }
    return out;
}

std::vector<std::array<float, 2>> sample_points(const Mask &mask, int max_points, double erode_frac, double grid_frac) {
    const int w = mask.w, h = mask.h;
    const int long_side = std::max(w, h);
    std::vector<int32_t> labels;
    std::vector<int64_t> areas;
    int n = connected_components(mask, labels, &areas);
    std::vector<int> order(std::max(n - 1, 0));
    std::iota(order.begin(), order.end(), 1);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return areas[a] > areas[b]; });

    // Bounding boxes, so each component's distance transform runs on a small crop. A 2px margin keeps the
    // result identical: pixels outside the component are zeros either way, and the image border is preserved.
    std::vector<std::array<int, 4>> bb(n, {INT32_MAX, INT32_MAX, -1, -1});
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            int l = labels[size_t(y) * w + x];
            if (!l) continue;
            auto &b = bb[l];
            b = {std::min(b[0], x), std::min(b[1], y), std::max(b[2], x), std::max(b[3], y)};
        }
    std::vector<std::array<float, 2>> peaks;
    for (int l : order) {
        int x0 = std::max(bb[l][0] - 2, 0), y0 = std::max(bb[l][1] - 2, 0);
        int x1 = std::min(bb[l][2] + 2, w - 1), y1 = std::min(bb[l][3] + 2, h - 1);
        Mask crop(x1 - x0 + 1, y1 - y0 + 1);
        for (int y = y0; y <= y1; y++)
            for (int x = x0; x <= x1; x++) crop.at(x - x0, y - y0) = labels[size_t(y) * w + x] == l;
        std::vector<float> dist = distance_transform(crop, 5);
        size_t best = std::max_element(dist.begin(), dist.end()) - dist.begin();
        peaks.push_back({float(x0 + int(best % crop.w)), float(y0 + int(best / crop.w))});
    }

    int k = std::max(1L, round_half_even(erode_frac * long_side));
    Mask core = erode(mask, k);
    int step = int(std::max(8L, round_half_even(grid_frac * long_side)));
    std::vector<std::array<float, 2>> grid;
    for (int y = step / 2; y < h; y += step)
        for (int x = step / 2; x < w; x += step)
            if (core.at(x, y)) grid.push_back({float(x), float(y)});
    long budget = long(max_points) - long(peaks.size());
    if (budget <= 0) {
        peaks.resize(max_points);
        return peaks;
    }
    if (long(grid.size()) > budget) {
        std::vector<std::array<float, 2>> picked;
        for (long i : linspace_round(long(grid.size()), budget)) picked.push_back(grid[i]);
        grid = std::move(picked);
    }
    peaks.insert(peaks.end(), grid.begin(), grid.end());
    return peaks;
}

Prompts prompts_from_saliency(const std::vector<uint8_t> &sal_u8, int w, int h, int max_points) {
    Mask bin(w, h);
    for (size_t i = 0; i < sal_u8.size(); i++) bin.data[i] = sal_u8[i] >= 128;  // u8 / 255 > 0.5
    std::vector<Mask> comps = largest_components(bin);
    // Secondary components cut by the image border are scenery peeking into the frame, not a second subject.
    std::vector<Mask> kept;
    for (size_t i = 0; i < comps.size(); i++)
        if (i == 0 || !touches_border(comps[i])) kept.push_back(std::move(comps[i]));
    Prompts p;
    if (kept.empty()) {
        p.box = {0.1 * w, 0.1 * h, 0.9 * w, 0.9 * h};
        p.points = {{float(w / 2.0), float(h / 2.0)}};
        p.sal_mask = Mask(w, h);
        return p;
    }
    p.sal_mask = Mask(w, h);
    for (const Mask &c : kept)
        for (size_t i = 0; i < c.data.size(); i++) p.sal_mask.data[i] |= c.data[i];
    int x1 = w, y1 = h, x2 = -1, y2 = -1;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            if (p.sal_mask.at(x, y)) {
                x1 = std::min(x1, x), y1 = std::min(y1, y), x2 = std::max(x2, x), y2 = std::max(y2, y);
            }
    p.box = {double(x1), double(y1), double(x2), double(y2)};
    p.points = sample_points(p.sal_mask, max_points);
    return p;
}

std::array<double, 4> pad_box(const std::array<double, 4> &box, long pad, int w, int h) {
    return {std::max(0.0, box[0] - pad), std::max(0.0, box[1] - pad), std::min(double(w - 1), box[2] + pad),
            std::min(double(h - 1), box[3] + pad)};
}

Mask fuse_masks(const Mask &sam, const Mask &sal, const std::array<double, 4> &box) {
    std::vector<int32_t> labels;
    int n = connected_components(sam, labels);
    std::vector<uint8_t> touching(n, 0);
    for (size_t i = 0; i < labels.size(); i++)
        if (sal.data[i] && sam.data[i]) touching[labels[i]] = 1;
    touching[0] = 0;
    int x1 = int(box[0]), y1 = int(box[1]), x2 = int(box[2]), y2 = int(box[3]);
    Mask out(sam.w, sam.h);
    for (int y = std::max(y1, 0); y <= std::min(y2, sam.h - 1); y++)
        for (int x = std::max(x1, 0); x <= std::min(x2, sam.w - 1); x++) {
            size_t i = size_t(y) * sam.w + x;
            out.data[i] = touching[labels[i]] || sal.data[i];
        }
    return out;
}

std::vector<uint8_t> make_trimap(const Mask &m) {
    int k = int(std::max(5L, round_half_even(0.012 * std::max(m.w, m.h))));
    Mask fg = erode(m, k);
    Mask maybe = dilate(m, k);
    // Components thinner than the band (text strokes, wires) would vanish from fg and become all "unknown",
    // which ViTMatte may erase: give each such component its medial core as definite foreground.
    std::vector<int32_t> labels;
    int n = connected_components(m, labels);
    if (n > 1) {
        std::vector<float> dist = distance_transform(m, 3);
        std::vector<uint8_t> has_fg(n, 0);
        std::vector<float> peak(n, 0.f);
        for (size_t i = 0; i < labels.size(); i++) {
            if (fg.data[i]) has_fg[labels[i]] = 1;
            peak[labels[i]] = std::max(peak[labels[i]], dist[i]);
        }
        for (size_t i = 0; i < labels.size(); i++) {
            int l = labels[i];
            if (l && !has_fg[l] && m.data[i] && dist[i] >= 0.5f * peak[l]) fg.data[i] = 1;
        }
    }
    std::vector<uint8_t> trimap(m.data.size(), 0);
    for (size_t i = 0; i < trimap.size(); i++) trimap[i] = fg.data[i] ? 255 : maybe.data[i] ? 128 : 0;
    return trimap;
}

ImageU8 resize_long_side(const ImageU8 &img, int long_side) {
    double scale = double(long_side) / std::max(img.w, img.h);
    if (scale >= 1.0) return img;
    return pil_resize(img, int(round_half_even(img.w * scale)), int(round_half_even(img.h * scale)), Filter::Bicubic);
}

// ---------------------------------------------------------------------------------------------------------------
// Models

struct Pipeline::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_ERROR, "subject_matting"};
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::unique_ptr<Ort::Session> birefnet, sam_encoder, sam_decoder, vitmatte;

    std::vector<Ort::Value> run(Ort::Session &s, const std::vector<const char *> &in_names, std::vector<Ort::Value> &inputs,
                                const std::vector<const char *> &out_names) {
        return s.Run(Ort::RunOptions{nullptr}, in_names.data(), inputs.data(), inputs.size(), out_names.data(),
                     out_names.size());
    }

    Ort::Value tensor(std::vector<float> &data, std::vector<int64_t> shape) {
        return Ort::Value::CreateTensor<float>(mem, data.data(), data.size(), shape.data(), shape.size());
    }

    // BiRefNet_lite at `size` -> saliency as uint8 at the image size.
    std::vector<uint8_t> saliency_at(const ImageU8 &img, int size) {
        std::vector<float> x = imagenet_tensor(img, size, size);
        std::vector<Ort::Value> in;
        in.push_back(tensor(x, {1, 3, size, size}));
        auto out = run(*birefnet, {"pixel_values"}, in, {"saliency"});
        const float *pred = out[0].GetTensorData<float>();
        ImageU8 small(size, size, 1);
        for (size_t i = 0; i < small.data.size(); i++) small.data[i] = uint8_t(pred[i] * 255.f);
        return pil_resize(small, img.w, img.h, Filter::Bilinear).data;
    }

    std::vector<uint8_t> saliency(const ImageU8 &img) {
        std::vector<uint8_t> sal = saliency_at(img, 768);
        size_t on = std::count_if(sal.begin(), sal.end(), [](uint8_t v) { return v >= 128; });
        // An (almost) empty map at 768 happens on some images; 1024 recovers them.
        if (double(on) / double(sal.size()) < 0.005) sal = saliency_at(img, 1024);
        return sal;
    }

    struct SamState {
        std::vector<float> feat0, feat1, embed;
        int w, h;
    };

    SamState sam_encode(const ImageU8 &img) {
        std::vector<float> x = imagenet_tensor(img, kSamSize, kSamSize);
        std::vector<Ort::Value> in;
        in.push_back(tensor(x, {1, 3, kSamSize, kSamSize}));
        auto out = run(*sam_encoder, {"pixel_values"}, in, {"high_res_feat0", "high_res_feat1", "image_embed"});
        auto copy = [](Ort::Value &v) {
            size_t n = v.GetTensorTypeAndShapeInfo().GetElementCount();
            const float *p = v.GetTensorData<float>();
            return std::vector<float>(p, p + n);
        };
        return {copy(out[0]), copy(out[1]), copy(out[2]), img.w, img.h};
    }

    // points (N, P, 2) in image pixels, labels (N, P) -> logits (N, 3, 256, 256), scores (N, 3).
    void sam_decode(SamState &s, const std::vector<std::array<float, 2>> &points, const std::vector<int64_t> &labels,
                    int64_t n, int64_t p, std::vector<float> &logits, std::vector<float> &scores) {
        const float sx = float(double(kSamSize) / s.w), sy = float(double(kSamSize) / s.h);
        std::vector<float> pts(points.size() * 2);
        for (size_t i = 0; i < points.size(); i++) {
            pts[i * 2] = points[i][0] * sx;
            pts[i * 2 + 1] = points[i][1] * sy;
        }
        std::vector<int64_t> lab = labels;
        std::vector<int64_t> pts_shape{1, n, p, 2}, lab_shape{1, n, p};
        std::vector<Ort::Value> in;
        in.push_back(tensor(s.feat0, {1, 32, 256, 256}));
        in.push_back(tensor(s.feat1, {1, 64, 128, 128}));
        in.push_back(tensor(s.embed, {1, 256, 64, 64}));
        in.push_back(tensor(pts, pts_shape));
        in.push_back(Ort::Value::CreateTensor<int64_t>(mem, lab.data(), lab.size(), lab_shape.data(), lab_shape.size()));
        auto out = run(*sam_decoder, {"high_res_feat0", "high_res_feat1", "image_embed", "points", "labels"}, in,
                       {"masks", "iou_scores"});
        const float *m = out[0].GetTensorData<float>();
        logits.assign(m, m + out[0].GetTensorTypeAndShapeInfo().GetElementCount());
        const float *sc = out[1].GetTensorData<float>();
        scores.assign(sc, sc + out[1].GetTensorTypeAndShapeInfo().GetElementCount());
    }

    // Each point prompts its own object; per point keep the largest of the 3 levels that lies mostly inside the
    // salient region and is stable (complete_mask in pipeline_a.py). Union of the kept masks.
    Mask complete_from_points(SamState &s, const std::vector<std::array<float, 2>> &points, const Mask &sal) {
        const int64_t n = int64_t(points.size());
        std::vector<float> logits, scores;
        sam_decode(s, points, std::vector<int64_t>(n, 1), n, 1, logits, scores);
        const size_t npx = size_t(s.w) * s.h, lowres = size_t(kSamMaskSize) * kSamMaskSize;
        std::vector<float> up(npx);
        std::vector<Mask> levels(3, Mask(s.w, s.h));
        Mask out(s.w, s.h);
        for (int64_t i = 0; i < n; i++) {
            int best = -1;
            int64_t best_area = -1;
            for (int j = 0; j < 3; j++) {
                resize_linear(&logits[(size_t(i) * 3 + j) * lowres], kSamMaskSize, kSamMaskSize, up.data(), s.w, s.h);
                int64_t area = 0, inside = 0, hi = 0, lo = 0;
                uint8_t *lv = levels[j].data.data();
                for (size_t k = 0; k < npx; k++) {
                    float v = up[k];
                    lv[k] = v > 0;
                    area += lv[k];
                    inside += lv[k] & sal.data[k];
                    hi += v > 1;
                    lo += v > -1;
                }
                float stability = float(double(hi) / double(std::max<int64_t>(lo, 1)));
                bool ok = area > 0 && double(inside) / double(std::max<int64_t>(area, 1)) >= kMinInside &&
                          stability >= kMinStability;
                if (ok && area > best_area) best = j, best_area = area;
            }
            if (best >= 0)
                for (size_t k = 0; k < npx; k++) out.data[k] |= levels[best].data[k];
        }
        return out;
    }

    // Nothing salient: one object from the central box + centre point, highest-scoring level.
    Mask from_box(SamState &s, const std::array<double, 4> &box, const std::vector<std::array<float, 2>> &points) {
        std::vector<std::array<float, 2>> pts = points;
        pts.push_back({float(box[0]), float(box[1])});
        pts.push_back({float(box[2]), float(box[3])});
        std::vector<int64_t> labels(points.size(), 1);
        labels.push_back(2);
        labels.push_back(3);
        std::vector<float> logits, scores;
        sam_decode(s, pts, labels, 1, int64_t(pts.size()), logits, scores);
        int best = int(std::max_element(scores.begin(), scores.begin() + 3) - scores.begin());
        std::vector<float> up(size_t(s.w) * s.h);
        resize_linear(&logits[size_t(best) * kSamMaskSize * kSamMaskSize], kSamMaskSize, kSamMaskSize, up.data(), s.w, s.h);
        Mask m(s.w, s.h);
        for (size_t k = 0; k < up.size(); k++) m.data[k] = up[k] > 0;
        return m;
    }

    std::vector<float> matte(const ImageU8 &img, const std::vector<uint8_t> &trimap) {
        const int w = img.w, h = img.h, pw = (w + 31) / 32 * 32, ph = (h + 31) / 32 * 32;
        const size_t plane = size_t(pw) * ph;
        std::vector<float> x(4 * plane, 0.f);
        for (int y = 0; y < h; y++)
            for (int xx = 0; xx < w; xx++) {
                size_t src = size_t(y) * w + xx, dst = size_t(y) * pw + xx;
                for (int c = 0; c < 3; c++) x[c * plane + dst] = float(img.data[src * 3 + c]) / 255.f * 2.f - 1.f;
                x[3 * plane + dst] = float(trimap[src]) / 255.f;
            }
        std::vector<Ort::Value> in;
        in.push_back(tensor(x, {1, 4, ph, pw}));
        auto out = run(*vitmatte, {"pixel_values"}, in, {"alphas"});
        const float *a = out[0].GetTensorData<float>();
        std::vector<float> alpha(size_t(w) * h);
        for (int y = 0; y < h; y++)
            for (int xx = 0; xx < w; xx++) {
                size_t i = size_t(y) * w + xx;
                uint8_t t = trimap[i];
                alpha[i] = t == 255 ? 1.f : t == 0 ? 0.f : a[size_t(y) * pw + xx];
            }
        return alpha;
    }
};

Pipeline::Pipeline(const std::string &models_dir, const std::string &precision, int num_threads)
    : impl_(std::make_unique<Impl>()) {
    // ONNX Runtime builds with telemetry would otherwise report usage to Microsoft from a background thread,
    // which embedding apps must not do silently (and which races with process exit).
    impl_->env.DisableTelemetryEvents();
    Ort::SessionOptions opts;
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (num_threads > 0) opts.SetIntraOpNumThreads(num_threads);
    auto load = [&](const char *name) {
        std::string path = models_dir + "/" + name + "." + precision + ".onnx";
        try {
            return std::make_unique<Ort::Session>(impl_->env, to_ort_path(path).c_str(), opts);
        } catch (const Ort::Exception &e) {
            throw std::runtime_error("cannot load " + path + ": " + e.what());
        }
    };
    impl_->birefnet = load("birefnet_lite");
    impl_->sam_encoder = load("sam2_encoder");
    impl_->sam_decoder = load("sam2_decoder");
    impl_->vitmatte = load("vitmatte");
}

Pipeline::~Pipeline() = default;

ImageU8 Pipeline::run(const ImageU8 &rgb, Timings *timings) {
    if (rgb.channels != 3 || rgb.w <= 0 || rgb.h <= 0) throw std::runtime_error("expected a non-empty RGB image");
    Timings t;
    try {
        double t0 = now();
        ImageU8 work = resize_long_side(rgb);
        const int w = work.w, h = work.h;
        std::vector<uint8_t> sal = impl_->saliency(work);
        Prompts prompts = prompts_from_saliency(sal, w, h);
        double t1 = now();

        Impl::SamState state = impl_->sam_encode(work);
        Mask mask;
        if (prompts.sal_mask.any()) {
            Mask completed = impl_->complete_from_points(state, prompts.points, prompts.sal_mask);
            auto safety = pad_box(prompts.box, round_half_even(kSafetyPad * std::max(w, h)), w, h);
            mask = fuse_masks(completed, prompts.sal_mask, safety);
        } else {
            mask = impl_->from_box(state, prompts.box, prompts.points);
        }
        double t2 = now();

        std::vector<uint8_t> trimap = make_trimap(mask);
        std::vector<float> alpha = impl_->matte(work, trimap);
        std::vector<float> full(size_t(rgb.w) * rgb.h);
        resize_linear(alpha.data(), w, h, full.data(), rgb.w, rgb.h);
        ImageU8 out(rgb.w, rgb.h, 4);
        for (size_t i = 0; i < full.size(); i++) {
            float a = std::min(std::max(full[i], 0.f), 1.f);
            out.data[i * 4] = rgb.data[i * 3];
            out.data[i * 4 + 1] = rgb.data[i * 3 + 1];
            out.data[i * 4 + 2] = rgb.data[i * 3 + 2];
            out.data[i * 4 + 3] = uint8_t(a * 255.f);
        }
        double t3 = now();
        t = {t1 - t0, t2 - t1, t3 - t2, t3 - t0};
        if (timings) *timings = t;
        return out;
    } catch (const Ort::Exception &e) {
        throw std::runtime_error(std::string("onnxruntime: ") + e.what());
    }
}

}  // namespace sm
