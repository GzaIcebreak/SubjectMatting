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
// Models

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

    struct Feats {
        std::vector<float> fpn0, fpn1, fpn2, pos2;
    };

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
        ImageU8 work = resize_long_side(rgb);
        const int w = work.w, h = work.h;
        const double sx = double(w) / rgb.w, sy = double(h) / rgb.h;
        Impl::Feats feats = impl_->encode_image(work);
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
        double t2 = now();
        if (timings) *timings = {t1 - t0, t2 - t1, t2 - t0};
        return out;
    } catch (const Ort::Exception &e) {
        throw std::runtime_error(std::string("onnxruntime: ") + e.what());
    }
}

}  // namespace sm
