#include "pipeline_q.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <numeric>

#include "clip_tokenizer.h"

namespace sm {

namespace {

constexpr int kSam3Size = 1008;
constexpr int kContext = 16;
constexpr float kConfidence = 0.3f;  // matting/sam3.py
constexpr float kCoreProb = 0.5f;
constexpr float kExtendProb = 0.3f;
constexpr double kCoreMinRatio = 0.05;
constexpr double kSafetyPad = 0.15;  // pipeline A's SAFETY_PAD
// matting/plain_background.py
constexpr double kRingFrac = 0.01;
constexpr double kUniformTol = 16;
constexpr double kUniformShare = 0.95;
constexpr double kKeyTol = 24;
constexpr double kMinArea = 0.0005;
constexpr int kMatteMargin = 64;  // matting/pipeline_vlm.py MATTE_MARGIN
constexpr double kRemoveMinCover = 0.1;  // REMOVE_MIN_COVER
constexpr double kRemoveMaxArea = 0.25;  // REMOVE_MAX_AREA
constexpr int kGrowQuant = 8;            // GROW_QUANT
constexpr int kGrowMinCount = 5;         // GROW_MIN_COUNT
constexpr double kGrowReach = 1.0;       // GROW_REACH

double now() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

inline float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }

double box_iou(const std::array<double, 4> &a, const std::array<double, 4> &b) {
    double ix = std::max(0.0, std::min(a[2], b[2]) - std::max(a[0], b[0]));
    double iy = std::max(0.0, std::min(a[3], b[3]) - std::max(a[1], b[1]));
    double inter = ix * iy;
    double uni = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
    return uni > 0 ? inter / uni : 0.0;
}

#ifdef _WIN32
std::wstring to_ort_path(const std::string &s) { return std::wstring(s.begin(), s.end()); }
#else
const std::string &to_ort_path(const std::string &s) { return s; }
#endif

}  // namespace

// ---------------------------------------------------------------------------------------------------------------
// Selection and cleanup (matting/pipeline_vlm.py)

double inside_fraction(const std::vector<float> &prob, int w, int h, const std::array<double, 4> &box, double margin) {
    size_t area = 0;
    for (float p : prob) area += p > kCoreProb;
    if (!area) return 0.0;
    double mx = (box[2] - box[0]) * margin, my = (box[3] - box[1]) * margin;
    int xa = std::max(0, int(box[0] - mx)), ya = std::max(0, int(box[1] - my));
    int xb = std::min(w, int(std::ceil(box[2] + mx))), yb = std::min(h, int(std::ceil(box[3] + my)));
    size_t inside = 0;
    for (int y = ya; y < yb; y++)
        for (int x = xa; x < xb; x++) inside += prob[size_t(y) * w + x] > kCoreProb;
    return double(inside) / double(area);
}

std::vector<int> select_instances(const Instances &inst, const std::array<double, 4> &box, double min_inside,
                                  double fallback_iou) {
    std::vector<int> hits;
    for (int i = 0; i < int(inst.size()); i++)
        if (inside_fraction(inst.probs[i], inst.w, inst.h, box) >= min_inside) hits.push_back(i);
    if (hits.empty() && inst.size()) {
        int best = 0;
        double best_iou = -1;
        for (int i = 0; i < int(inst.size()); i++) {
            double iou = box_iou(inst.boxes[i], box);
            if (iou > best_iou) best = i, best_iou = iou;
        }
        if (best_iou >= fallback_iou) hits.push_back(best);
    }
    return hits;
}

std::vector<int> drop_duplicates(const Instances &inst, const std::vector<int> &idx, double share) {
    std::vector<int> order = idx;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return inst.scores[a] > inst.scores[b]; });
    std::vector<int> kept;
    const size_t n = size_t(inst.w) * inst.h;
    for (int i : order) {
        const std::vector<float> &pi = inst.probs[i];
        size_t area = 0;
        for (float p : pi) area += p > kCoreProb;
        if (!area) continue;
        bool dup = false;
        for (int j : kept) {
            size_t both = 0;
            const std::vector<float> &pj = inst.probs[j];
            for (size_t k = 0; k < n; k++) both += (pi[k] > kCoreProb) && (pj[k] > kCoreProb);
            if (double(both) >= share * double(area)) {
                dup = true;
                break;
            }
        }
        if (!dup) kept.push_back(i);
    }
    std::sort(kept.begin(), kept.end());
    return kept;
}

Mask complete_instance(const std::vector<float> &prob, int w, int h, const std::array<double, 4> &box) {
    Mask core_in(w, h), extend(w, h);
    for (size_t k = 0; k < prob.size(); k++) {
        core_in.data[k] = prob[k] > kCoreProb;
        extend.data[k] = prob[k] > kExtendProb;
    }
    std::vector<Mask> comps = largest_components(core_in, kCoreMinRatio);
    if (comps.empty()) return Mask(w, h);
    Mask core(w, h);
    for (const Mask &c : comps)
        for (size_t k = 0; k < c.data.size(); k++) core.data[k] |= c.data[k];
    auto safety = pad_box(box, round_half_even(kSafetyPad * std::max(w, h)), w, h);
    return fuse_masks(extend, core, safety);
}

// ---------------------------------------------------------------------------------------------------------------
// Plain background (matting/plain_background.py)

bool plain_background(const ImageU8 &work, std::array<double, 3> &bg) {
    const int w = work.w, h = work.h;
    const int m = int(std::max(2L, round_half_even(kRingFrac * std::max(w, h))));
    std::vector<const uint8_t *> ring;
    auto add = [&](int x, int y) { ring.push_back(work.data.data() + (size_t(y) * w + x) * 3); };
    for (int y = 0; y < h; y++) {
        bool edge_row = y < m || y >= h - m;
        for (int x = 0; x < w; x++)
            if (edge_row || x < m || x >= w - m) add(x, y);
    }
    if (ring.empty()) return false;
    // numpy.median per channel: the middle value, or the mean of the two middle values for an even count.
    for (int c = 0; c < 3; c++) {
        size_t hist[256] = {0};
        for (const uint8_t *p : ring) hist[p[c]]++;
        auto kth = [&](size_t k) {
            size_t acc = 0;
            for (int v = 0; v < 256; v++) {
                acc += hist[v];
                if (acc > k) return double(v);
            }
            return 255.0;
        };
        size_t n = ring.size();
        bg[c] = n % 2 ? kth(n / 2) : (kth(n / 2 - 1) + kth(n / 2)) / 2.0;
    }
    size_t uniform = 0;
    for (const uint8_t *p : ring) {
        double d = std::max({std::fabs(p[0] - bg[0]), std::fabs(p[1] - bg[1]), std::fabs(p[2] - bg[2])});
        uniform += d <= kUniformTol;
    }
    return double(uniform) / double(ring.size()) >= kUniformShare;
}

Mask key_mask(const ImageU8 &work, const std::array<double, 3> &bg) {
    Mask fg(work.w, work.h);
    for (size_t i = 0; i < fg.data.size(); i++) {
        const uint8_t *p = work.data.data() + i * 3;
        double d = std::max({std::fabs(p[0] - bg[0]), std::fabs(p[1] - bg[1]), std::fabs(p[2] - bg[2])});
        fg.data[i] = d > kKeyTol;
    }
    std::vector<int32_t> labels;
    std::vector<int64_t> areas;
    int n = connected_components(fg, labels, &areas);
    std::vector<uint8_t> keep(n, 0);
    for (int l = 1; l < n; l++) keep[l] = double(areas[l]) >= kMinArea * double(fg.data.size());
    for (size_t i = 0; i < fg.data.size(); i++) fg.data[i] = keep[labels[i]];
    return fg;
}

bool is_plain(const ImageU8 &rgb) {
    ImageU8 work = resize_long_side(rgb);
    std::array<double, 3> bg{};
    return plain_background(work, bg) && key_mask(work, bg).any();
}

Mask fill_holes(const Mask &mask) {
    const int w = mask.w, h = mask.h;
    std::vector<uint8_t> outside(mask.data.size(), 0);
    std::vector<int> stack;
    auto push = [&](int x, int y) {
        size_t i = size_t(y) * w + x;
        if (!mask.data[i] && !outside[i]) outside[i] = 1, stack.push_back(int(i));
    };
    for (int x = 0; x < w; x++) push(x, 0), push(x, h - 1);
    for (int y = 0; y < h; y++) push(0, y), push(w - 1, y);
    while (!stack.empty()) {
        int i = stack.back();
        stack.pop_back();
        int x = i % w, y = i / w;
        if (x > 0) push(x - 1, y);
        if (x < w - 1) push(x + 1, y);
        if (y > 0) push(x, y - 1);
        if (y < h - 1) push(x, y + 1);
    }
    Mask out(w, h);
    for (size_t i = 0; i < out.data.size(); i++) out.data[i] = !outside[i];
    return out;
}

std::vector<Mask> key_candidates(const ImageU8 &work, const std::array<double, 3> &bg) {
    Mask fg = key_mask(work, bg);
    std::vector<Mask> out;
    for (Mask &piece : largest_components(fg, 0.0))
        if (double(piece.count()) >= kMinArea * double(fg.data.size())) out.push_back(fill_holes(piece));
    return out;
}

Mask grow_like(const Mask &seed, const Mask &within, const ImageU8 &work, const std::array<double, 4> &box) {
    if (!seed.any()) return seed;
    const int w = seed.w, h = seed.h, n = 256 / kGrowQuant;
    auto key = [&](size_t i) {
        const uint8_t *p = work.data.data() + i * 3;
        return (p[0] / kGrowQuant * n + p[1] / kGrowQuant) * n + p[2] / kGrowQuant;
    };
    std::vector<int> counts(size_t(n) * n * n, 0);
    for (size_t i = 0; i < seed.data.size(); i++)
        if (seed.data[i]) counts[key(i)]++;
    std::vector<uint8_t> palette(counts.size(), 0);
    for (int v = 0; v < int(counts.size()); v++) {
        if (counts[v] < kGrowMinCount) continue;
        int r = v / (n * n), g = (v / n) % n, b = v % n;
        for (int dr = -1; dr <= 1; dr++)
            for (int dg = -1; dg <= 1; dg++)
                for (int db = -1; db <= 1; db++) {
                    // numpy keys may run past a channel's range and alias a neighbour; the same arithmetic here.
                    long k = (long(r + dr) * n + g + dg) * n + b + db;
                    if (k >= 0 && k < long(palette.size())) palette[k] = 1;
                }
    }
    const double mx = (box[2] - box[0]) * kGrowReach, my = (box[3] - box[1]) * kGrowReach;
    const int xa = std::max(0, int(box[0] - mx)), ya = std::max(0, int(box[1] - my));
    const int xb = std::min(w, int(std::ceil(box[2] + mx))), yb = std::min(h, int(std::ceil(box[3] + my)));
    Mask alike(w, h);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            size_t i = size_t(y) * w + x;
            bool near = x >= xa && x < xb && y >= ya && y < yb;
            alike.data[i] = seed.data[i] || (near && within.data[i] && palette[key(i)]);
        }
    std::vector<int32_t> labels;
    int nl = connected_components(alike, labels);
    std::vector<uint8_t> hit(nl, 0);
    for (size_t i = 0; i < labels.size(); i++)
        if (seed.data[i]) hit[labels[i]] = 1;
    Mask out(w, h);
    for (size_t i = 0; i < labels.size(); i++) out.data[i] = labels[i] > 0 && hit[labels[i]];
    return out;
}

double box_cover(const Mask &mask, const std::array<double, 4> &box) {
    // Python: int(round(v)) with banker's rounding, then clipped.
    int x1 = std::max(0, int(round_half_even(box[0]))), y1 = std::max(0, int(round_half_even(box[1])));
    int x2 = std::min(mask.w, int(round_half_even(box[2]))), y2 = std::min(mask.h, int(round_half_even(box[3])));
    double area = std::max(1, (x2 - x1) * (y2 - y1));
    size_t covered = 0;
    for (int y = y1; y < y2; y++)
        for (int x = x1; x < x2; x++) covered += mask.at(x, y) != 0;
    return double(covered) / area;
}

namespace {

// Cheap identity check for prepare(): size plus a sample of the pixels.
uint64_t fingerprint(const ImageU8 &img) {
    uint64_t h = 1469598103934665603ull ^ (uint64_t(img.w) << 32 | uint64_t(img.h));
    const size_t step = std::max<size_t>(1, img.data.size() / 4096);
    for (size_t i = 0; i < img.data.size(); i += step) h = (h ^ img.data[i]) * 1099511628211ull;
    return h;
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------------
// Models

struct PipelineQ::Feats {
    std::vector<float> fpn0, fpn1, fpn2, pos2;
};

struct PipelineQ::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_ERROR, "subject_matting_q"};
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::unique_ptr<Ort::Session> image_encoder, text_encoder, decoder, vitmatte;
    std::vector<std::string> decoder_inputs;
    std::unique_ptr<ClipTokenizer> tokenizer;
    std::map<std::string, std::pair<std::vector<float>, std::vector<int64_t>>> text_cache;  // label -> text, tokens

    Ort::Value tensor(std::vector<float> &data, std::vector<int64_t> shape) {
        return Ort::Value::CreateTensor<float>(mem, data.data(), data.size(), shape.data(), shape.size());
    }

    static std::vector<float> copy(Ort::Value &v) {
        const float *p = v.GetTensorData<float>();
        return std::vector<float>(p, p + v.GetTensorTypeAndShapeInfo().GetElementCount());
    }

    using Feats = PipelineQ::Feats;

    // prepare() result, used once by the next run() on the same pixels.
    bool prepared = false;
    uint64_t prepared_id = 0;
    ImageU8 prepared_work;
    Feats prepared_feats;

    Feats encode_image(const ImageU8 &work) {
        ImageU8 r = pil_resize(work, kSam3Size, kSam3Size, Filter::Bilinear);
        const size_t plane = size_t(kSam3Size) * kSam3Size;
        std::vector<float> x(3 * plane);
        for (size_t i = 0; i < plane; i++)
            for (int c = 0; c < 3; c++) x[c * plane + i] = (float(r.data[i * 3 + c]) / 255.f - 0.5f) / 0.5f;
        std::vector<Ort::Value> in;
        in.push_back(tensor(x, {1, 3, kSam3Size, kSam3Size}));
        const char *in_names[] = {"image"};
        const char *out_names[] = {"fpn0", "fpn1", "fpn2", "pos2"};
        auto out = image_encoder->Run(Ort::RunOptions{nullptr}, in_names, in.data(), 1, out_names, 4);
        return {copy(out[0]), copy(out[1]), copy(out[2]), copy(out[3])};
    }

    std::pair<std::vector<float>, std::vector<int64_t>> &text(const std::string &label) {
        auto it = text_cache.find(label);
        if (it != text_cache.end()) return it->second;
        std::vector<int64_t> tokens = tokenizer->tokenize(label, kContext);
        std::vector<int64_t> shape{1, kContext};
        std::vector<Ort::Value> in;
        in.push_back(Ort::Value::CreateTensor<int64_t>(mem, tokens.data(), tokens.size(), shape.data(), shape.size()));
        const char *in_names[] = {"tokens"};
        const char *out_names[] = {"text"};
        auto out = text_encoder->Run(Ort::RunOptions{nullptr}, in_names, in.data(), 1, out_names, 1);
        return text_cache[label] = {copy(out[0]), tokens};
    }

    // One decoder pass with `box` (work pixels) as the exemplar -> the confident instances at work size.
    Instances find(Feats &f, const std::string &label, const std::array<double, 4> &box, int w, int h) {
        auto &[text_feats, tokens] = text(label);
        std::vector<float> cxcywh{float((box[0] + box[2]) / 2 / w), float((box[1] + box[3]) / 2 / h),
                                  float((box[2] - box[0]) / w), float((box[3] - box[1]) / h)};
        // ONNX bool tensors: one byte per element.
        std::unique_ptr<bool[]> text_mask(new bool[kContext]);
        for (int i = 0; i < kContext; i++) text_mask[i] = tokens[i] == 0;
        bool box_label = true;
        std::vector<int64_t> s_mask{1, kContext}, s_box{1, 1, 4}, s_label{1, 1};
        std::vector<std::pair<std::string, Ort::Value>> all;
        all.emplace_back("fpn0", tensor(f.fpn0, {1, 256, 288, 288}));
        all.emplace_back("fpn1", tensor(f.fpn1, {1, 256, 144, 144}));
        all.emplace_back("fpn2", tensor(f.fpn2, {1, 256, 72, 72}));
        all.emplace_back("pos2", tensor(f.pos2, {1, 256, 72, 72}));
        all.emplace_back("text", tensor(text_feats, {kContext, 1, 256}));
        all.emplace_back("text_mask", Ort::Value::CreateTensor<bool>(mem, text_mask.get(), kContext, s_mask.data(), 2));
        all.emplace_back("boxes", tensor(cxcywh, s_box));
        all.emplace_back("box_labels", Ort::Value::CreateTensor<bool>(mem, &box_label, 1, s_label.data(), 2));
        // The exporter prunes inputs the graph doesn't read: feed only the declared ones.
        std::vector<const char *> names;
        std::vector<Ort::Value> values;
        for (auto &[n, v] : all) {
            if (std::find(decoder_inputs.begin(), decoder_inputs.end(), n) != decoder_inputs.end()) {
                names.push_back(n.c_str());
                values.push_back(std::move(v));
            }
        }
        const char *out_names[] = {"logits", "pred_boxes", "masks", "presence"};
        auto out = decoder->Run(Ort::RunOptions{nullptr}, names.data(), values.data(), values.size(), out_names, 4);
        const float *logits = out[0].GetTensorData<float>();
        const float *boxes = out[1].GetTensorData<float>();
        const float *masks = out[2].GetTensorData<float>();
        const float presence = sigmoid(out[3].GetTensorData<float>()[0]);
        const int queries = int(out[0].GetTensorTypeAndShapeInfo().GetShape()[1]);
        const int mh = int(out[2].GetTensorTypeAndShapeInfo().GetShape()[2]);
        const int mw = int(out[2].GetTensorTypeAndShapeInfo().GetShape()[3]);

        Instances inst;
        inst.w = w;
        inst.h = h;
        for (int q = 0; q < queries; q++) {
            float score = sigmoid(logits[q]) * presence;
            if (!(score > kConfidence)) continue;
            const float *b = boxes + size_t(q) * 4;
            float cx = b[0], cy = b[1], bw = b[2], bh = b[3];
            inst.boxes.push_back({double((cx - bw / 2) * float(w)), double((cy - bh / 2) * float(h)),
                                  double((cx + bw / 2) * float(w)), double((cy + bh / 2) * float(h))});
            std::vector<float> prob(size_t(w) * h);
            resize_linear(masks + size_t(q) * mh * mw, mw, mh, prob.data(), w, h);
            for (float &p : prob) p = sigmoid(p);
            inst.probs.push_back(std::move(prob));
            inst.scores.push_back(score);
        }
        return inst;
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
        const char *in_names[] = {"pixel_values"};
        const char *out_names[] = {"alphas"};
        auto out = vitmatte->Run(Ort::RunOptions{nullptr}, in_names, in.data(), 1, out_names, 1);
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

PipelineQ::PipelineQ(const std::string &models_dir, const std::string &precision, int num_threads)
    : impl_(std::make_unique<Impl>()) {
    impl_->env.DisableTelemetryEvents();
    Ort::SessionOptions opts;
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (num_threads > 0) opts.SetIntraOpNumThreads(num_threads);
    auto load = [&](const std::string &file) {
        std::string path = models_dir + "/" + file;
        try {
            return std::make_unique<Ort::Session>(impl_->env, to_ort_path(path).c_str(), opts);
        } catch (const Ort::Exception &e) {
            throw std::runtime_error("cannot load " + path + ": " + e.what());
        }
    };
    impl_->image_encoder = load("sam3_image_encoder." + precision + ".onnx");
    impl_->text_encoder = load("sam3_text_encoder." + precision + ".onnx");
    impl_->decoder = load("sam3_decoder." + precision + ".onnx");
    impl_->vitmatte = load("vitmatte.fp16.onnx");
    Ort::AllocatorWithDefaultOptions alloc;
    for (size_t i = 0; i < impl_->decoder->GetInputCount(); i++)
        impl_->decoder_inputs.push_back(impl_->decoder->GetInputNameAllocated(i, alloc).get());
    impl_->tokenizer = std::make_unique<ClipTokenizer>(models_dir + "/clip_bpe_merges.txt");
}

PipelineQ::~PipelineQ() = default;

ImageU8 PipelineQ::run(const ImageU8 &rgb, const std::vector<Subject> &subjects, TimingsQ *timings) {
    if (rgb.channels != 3 || rgb.w <= 0 || rgb.h <= 0) throw std::runtime_error("expected a non-empty RGB image");
    if (subjects.empty()) throw NothingSegmented("no subjects given");
    try {
        double t0 = now();
        ImageU8 work;
        Impl::Feats feats;
        if (impl_->prepared && impl_->prepared_id == fingerprint(rgb)) {
            work = std::move(impl_->prepared_work);
            feats = std::move(impl_->prepared_feats);
        } else {
            work = resize_long_side(rgb);
            feats = impl_->encode_image(work);
        }
        impl_->prepared = false;
        const int w = work.w, h = work.h;
        const double sx = double(w) / rgb.w, sy = double(h) / rgb.h;
        Mask mask(w, h);
        for (const Subject &s : subjects) {
            std::array<double, 4> box{s.box[0] * sx, s.box[1] * sy, s.box[2] * sx, s.box[3] * sy};
            Instances inst = impl_->find(feats, s.label, box, w, h);
            for (int i : drop_duplicates(inst, select_instances(inst, box))) {
                Mask m = complete_instance(inst.probs[i], w, h, box);
                for (size_t k = 0; k < m.data.size(); k++) mask.data[k] |= m.data[k];
            }
        }
        double t1 = now();
        if (!mask.any()) throw NothingSegmented("SAM3 did not segment anything at the given boxes");
        ImageU8 out = matte_and_compose(rgb, work, mask);
        double t2 = now();
        if (timings) *timings = {t1 - t0, t2 - t1, t2 - t0};
        return out;
    } catch (const Ort::Exception &e) {
        throw std::runtime_error(std::string("onnxruntime: ") + e.what());
    }
}

void PipelineQ::prepare(const ImageU8 &rgb) {
    if (rgb.channels != 3 || rgb.w <= 0 || rgb.h <= 0) throw std::runtime_error("expected a non-empty RGB image");
    try {
        impl_->prepared = false;
        impl_->prepared_work = resize_long_side(rgb);
        impl_->prepared_feats = impl_->encode_image(impl_->prepared_work);
        impl_->prepared_id = fingerprint(rgb);
        impl_->prepared = true;
    } catch (const Ort::Exception &e) {
        throw std::runtime_error(std::string("onnxruntime: ") + e.what());
    }
}

ImageU8 PipelineQ::run_plain(const ImageU8 &rgb, TimingsQ *timings) {
    if (rgb.channels != 3 || rgb.w <= 0 || rgb.h <= 0) throw std::runtime_error("expected a non-empty RGB image");
    try {
        double t0 = now();
        ImageU8 work = resize_long_side(rgb);
        std::array<double, 3> bg{};
        if (!plain_background(work, bg)) throw NotPlain("the background is not a plain colour");
        Mask mask = key_mask(work, bg);
        if (!mask.any()) throw NotPlain("nothing differs from the background colour");
        double t1 = now();
        ImageU8 out = matte_and_compose(rgb, work, mask);
        double t2 = now();
        if (timings) *timings = {t1 - t0, t2 - t1, t2 - t0};
        return out;
    } catch (const Ort::Exception &e) {
        throw std::runtime_error(std::string("onnxruntime: ") + e.what());
    }
}

PipelineQ::Feats &PipelineQ::feats_for(const ImageU8 &rgb, ImageU8 &work) {
    uint64_t id = fingerprint(rgb);
    if (!impl_->prepared || impl_->prepared_id != id) {
        impl_->prepared = false;
        impl_->prepared_work = resize_long_side(rgb);
        impl_->prepared_feats = impl_->encode_image(impl_->prepared_work);
        impl_->prepared_id = id;
        impl_->prepared = true;
    }
    work = impl_->prepared_work;
    return impl_->prepared_feats;
}

std::vector<Candidate> PipelineQ::candidates(const ImageU8 &rgb, const std::vector<Subject> &subjects, bool auto_key,
                                             bool *used_key) {
    if (rgb.channels != 3 || rgb.w <= 0 || rgb.h <= 0) throw std::runtime_error("expected a non-empty RGB image");
    try {
        std::vector<Candidate> out;
        if (used_key) *used_key = false;
        if (auto_key) {
            ImageU8 work = resize_long_side(rgb);
            std::array<double, 3> bg{};
            if (plain_background(work, bg))
                for (Mask &m : key_candidates(work, bg)) out.push_back({std::move(m), -1});
            if (!out.empty()) {
                if (used_key) *used_key = true;
                return out;
            }
        }
        ImageU8 work;
        Feats &feats = feats_for(rgb, work);
        const int w = work.w, h = work.h;
        const double sx = double(w) / rgb.w, sy = double(h) / rgb.h;
        for (int k = 0; k < int(subjects.size()); k++) {
            const Subject &s = subjects[k];
            std::array<double, 4> box{s.box[0] * sx, s.box[1] * sy, s.box[2] * sx, s.box[3] * sy};
            Instances inst = impl_->find(feats, s.label, box, w, h);
            for (int i : drop_duplicates(inst, select_instances(inst, box))) {
                Mask m = complete_instance(inst.probs[i], w, h, box);
                if (m.any()) out.push_back({std::move(m), k});
            }
        }
        return out;
    } catch (const Ort::Exception &e) {
        throw std::runtime_error(std::string("onnxruntime: ") + e.what());
    }
}

void PipelineQ::remove(const ImageU8 &rgb, Mask &mask, const std::vector<Subject> &regions) {
    if (rgb.channels != 3 || rgb.w <= 0 || rgb.h <= 0) throw std::runtime_error("expected a non-empty RGB image");
    try {
        const ImageU8 plain_work = resize_long_side(rgb);
        if (mask.w != plain_work.w || mask.h != plain_work.h) throw std::runtime_error("mask is not at work size");
        const int w = mask.w, h = mask.h;
        const double sx = double(w) / rgb.w, sy = double(h) / rgb.h;
        for (const Subject &r : regions) {
            std::array<double, 4> box{r.box[0] * sx, r.box[1] * sy, r.box[2] * sx, r.box[3] * sy};
            if ((box[2] - box[0]) * (box[3] - box[1]) > kRemoveMaxArea * w * h || box_cover(mask, box) < kRemoveMinCover)
                continue;
            ImageU8 work;
            Feats &feats = feats_for(rgb, work);
            Instances inst = impl_->find(feats, r.label, box, w, h);
            for (int i : select_instances(inst, box)) {
                Mask seed = complete_instance(inst.probs[i], w, h, box);
                for (size_t k = 0; k < seed.data.size(); k++) seed.data[k] &= mask.data[k];
                Mask cut = grow_like(seed, mask, work, box);
                for (size_t k = 0; k < cut.data.size(); k++)
                    if (cut.data[k]) mask.data[k] = 0;
            }
        }
    } catch (const Ort::Exception &e) {
        throw std::runtime_error(std::string("onnxruntime: ") + e.what());
    }
}

ImageU8 PipelineQ::matte_mask(const ImageU8 &rgb, const Mask &mask) {
    if (rgb.channels != 3 || rgb.w <= 0 || rgb.h <= 0) throw std::runtime_error("expected a non-empty RGB image");
    if (!mask.any()) throw NothingSegmented("empty mask");
    try {
        impl_->prepared = false;
        impl_->prepared_feats = Feats();
        impl_->prepared_work = ImageU8();
        ImageU8 work = resize_long_side(rgb);
        if (mask.w != work.w || mask.h != work.h) throw std::runtime_error("mask is not at work size");
        return matte_and_compose(rgb, work, mask);
    } catch (const Ort::Exception &e) {
        throw std::runtime_error(std::string("onnxruntime: ") + e.what());
    }
}

ImageU8 PipelineQ::matte_and_compose(const ImageU8 &rgb, const ImageU8 &work, const Mask &mask) {
    std::vector<uint8_t> trimap = make_trimap(mask);
    // ViTMatte only on the trimap's non-background box plus a margin (matte_cropped in pipeline_vlm.py).
    int x0 = work.w, y0 = work.h, x1 = -1, y1 = -1;
    for (int y = 0; y < work.h; y++)
        for (int x = 0; x < work.w; x++)
            if (trimap[size_t(y) * work.w + x]) x0 = std::min(x0, x), y0 = std::min(y0, y), x1 = std::max(x1, x), y1 = std::max(y1, y);
    x0 = std::max(0, x0 - kMatteMargin), y0 = std::max(0, y0 - kMatteMargin);
    x1 = std::min(work.w, x1 + 1 + kMatteMargin), y1 = std::min(work.h, y1 + 1 + kMatteMargin);
    const int cw = x1 - x0, ch = y1 - y0;
    ImageU8 crop(cw, ch, 3);
    std::vector<uint8_t> crop_trimap(size_t(cw) * ch);
    for (int y = 0; y < ch; y++) {
        std::copy(work.row(y + y0) + size_t(x0) * 3, work.row(y + y0) + size_t(x1) * 3, crop.row(y));
        std::copy(&trimap[size_t(y + y0) * work.w + x0], &trimap[size_t(y + y0) * work.w + x1], &crop_trimap[size_t(y) * cw]);
    }
    std::vector<float> crop_alpha = impl_->matte(crop, crop_trimap);
    std::vector<float> alpha(size_t(work.w) * work.h, 0.f);
    for (int y = 0; y < ch; y++)
        std::copy(&crop_alpha[size_t(y) * cw], &crop_alpha[size_t(y) * cw] + cw, &alpha[size_t(y + y0) * work.w + x0]);
    std::vector<float> full(size_t(rgb.w) * rgb.h);
    resize_linear(alpha.data(), work.w, work.h, full.data(), rgb.w, rgb.h);
    ImageU8 out(rgb.w, rgb.h, 4);
    for (size_t i = 0; i < full.size(); i++) {
        float a = std::min(std::max(full[i], 0.f), 1.f);
        out.data[i * 4] = rgb.data[i * 3];
        out.data[i * 4 + 1] = rgb.data[i * 3 + 1];
        out.data[i * 4 + 2] = rgb.data[i * 3 + 2];
        out.data[i * 4 + 3] = uint8_t(a * 255.f);
    }
    return out;
}

}  // namespace sm
