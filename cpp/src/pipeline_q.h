// Pipeline Q: the subjects (label + box) come from a vision-language model (Qwen) chosen by the caller;
// EfficientSAM3 segments each one, instances are picked and cleaned like pipeline A, then trimap + ViTMatte.
// A port of matting/pipeline_vlm_onnx.py (+ pipeline_vlm.py); shares image ops and mask helpers with pipeline A.
#pragma once

#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "image_ops.h"
#include "pipeline.h"

namespace sm {

struct Subject {
    std::string label;          // short English noun phrase
    std::array<double, 4> box;  // x1, y1, x2, y2 in pixels of the input image
};

// SAM3 instances for one prompt, at work-image size.
struct Instances {
    std::vector<std::vector<float>> probs;  // per-pixel mask probability
    std::vector<std::array<double, 4>> boxes;
    std::vector<float> scores;
    int w = 0, h = 0;
    size_t size() const { return scores.size(); }
};

// Selection and cleanup (pure functions, mirrored from matting/pipeline_vlm.py and unit tested there).
double inside_fraction(const std::vector<float> &prob, int w, int h, const std::array<double, 4> &box, double margin = 0.05);
std::vector<int> select_instances(const Instances &inst, const std::array<double, 4> &box, double min_inside = 0.5,
                                  double fallback_iou = 0.3);
std::vector<int> drop_duplicates(const Instances &inst, const std::vector<int> &idx, double share = 0.5);
Mask complete_instance(const std::vector<float> &prob, int w, int h, const std::array<double, 4> &box);

// Plain-background fast path (matting/plain_background.py): the background colour if the border ring is uniform.
bool plain_background(const ImageU8 &work, std::array<double, 3> &bg);
// Pixels that differ from the background colour, specks removed.
Mask key_mask(const ImageU8 &work, const std::array<double, 3> &bg);

struct TimingsQ {
    double sam3 = 0, matting = 0, total = 0;  // seconds
};

// The background is not a plain colour (run_plain); use run() with subjects instead.
struct NotPlain : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// No instance of any subject could be segmented.
struct NothingSegmented : std::runtime_error {
    using std::runtime_error::runtime_error;
};

class PipelineQ {
public:
    // Loads <dir>/{sam3_image_encoder,sam3_text_encoder,sam3_decoder}.<precision>.onnx, vitmatte.fp16.onnx and
    // clip_bpe_merges.txt. Throws std::runtime_error.
    PipelineQ(const std::string &models_dir, const std::string &precision, int num_threads);
    ~PipelineQ();
    PipelineQ(const PipelineQ &) = delete;
    PipelineQ &operator=(const PipelineQ &) = delete;

    // rgb: 3-channel image; subjects in its pixel coordinates -> RGBA. Throws NothingSegmented or runtime_error.
    ImageU8 run(const ImageU8 &rgb, const std::vector<Subject> &subjects, TimingsQ *timings = nullptr);

    // Auto mode on a solid background: everything that isn't background, no SAM3. Throws NotPlain otherwise.
    // timings->sam3 holds the keying time.
    ImageU8 run_plain(const ImageU8 &rgb, TimingsQ *timings = nullptr);

private:
    ImageU8 matte_and_compose(const ImageU8 &rgb, const ImageU8 &work, const Mask &mask);

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace sm
