// Pipeline A on ONNX Runtime: BiRefNet_lite saliency -> point prompts -> SAM2.1-tiny object completion -> trimap
// -> ViTMatte. A line-by-line port of matting/pipeline_onnx.py (+ the helpers it shares with pipeline_a.py).
#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "image_ops.h"

namespace sm {

struct Timings {
    double saliency = 0, sam2 = 0, matting = 0, total = 0;
};

struct Prompts {
    std::array<double, 4> box{};             // x1, y1, x2, y2 of the salient region
    std::vector<std::array<float, 2>> points;  // positive prompts, one SAM object each
    Mask sal_mask;
};

// Pure functions, exposed for tests (they mirror the Python functions of the same names).
std::vector<Mask> largest_components(const Mask &m, double min_ratio = 0.1);
std::vector<std::array<float, 2>> sample_points(const Mask &m, int max_points = 25, double erode_frac = 0.015,
                                                double grid_frac = 0.1);
Prompts prompts_from_saliency(const std::vector<uint8_t> &sal_u8, int w, int h, int max_points = 25);
std::vector<uint8_t> make_trimap(const Mask &mask);
Mask fuse_masks(const Mask &sam, const Mask &sal, const std::array<double, 4> &box);
std::array<double, 4> pad_box(const std::array<double, 4> &box, long pad, int w, int h);
ImageU8 resize_long_side(const ImageU8 &img, int long_side = 1024);

class Pipeline {
public:
    // Loads <dir>/{birefnet_lite,sam2_encoder,sam2_decoder,vitmatte}.<precision>.onnx. Throws std::runtime_error.
    Pipeline(const std::string &models_dir, const std::string &precision, int num_threads);
    ~Pipeline();
    Pipeline(const Pipeline &) = delete;
    Pipeline &operator=(const Pipeline &) = delete;

    // rgb: 3-channel image -> RGBA with the computed alpha. Throws std::runtime_error.
    ImageU8 run(const ImageU8 &rgb, Timings *timings = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace sm
