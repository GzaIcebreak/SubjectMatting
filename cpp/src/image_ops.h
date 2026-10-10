// Image operations the pipeline needs, written to match what the Python reference uses (PIL resize, OpenCV
// morphology / connected components / distance transform / bilinear resize) closely enough that both pick the
// same SAM masks. No third-party dependencies.
#pragma once

#include <cstdint>
#include <vector>

namespace sm {

// 8-bit interleaved image, rows packed (stride == w * channels).
struct ImageU8 {
    int w = 0, h = 0, channels = 0;
    std::vector<uint8_t> data;

    ImageU8() = default;
    ImageU8(int w_, int h_, int c_) : w(w_), h(h_), channels(c_), data(size_t(w_) * h_ * c_) {}
    uint8_t *row(int y) { return data.data() + size_t(y) * w * channels; }
    const uint8_t *row(int y) const { return data.data() + size_t(y) * w * channels; }
};

// Binary mask: one byte per pixel, 0 or 1.
struct Mask {
    int w = 0, h = 0;
    std::vector<uint8_t> data;

    Mask() = default;
    Mask(int w_, int h_) : w(w_), h(h_), data(size_t(w_) * h_, 0) {}
    uint8_t &at(int x, int y) { return data[size_t(y) * w + x]; }
    uint8_t at(int x, int y) const { return data[size_t(y) * w + x]; }
    bool any() const;
    size_t count() const;
};

enum class Filter { Bilinear, Bicubic };

// PIL Image.resize(..., BILINEAR|BICUBIC) for 8-bit images: same coefficients, fixed-point math and
// horizontal-then-vertical passes, so results match Pillow bit for bit.
ImageU8 pil_resize(const ImageU8 &src, int out_w, int out_h, Filter filter);

// cv2.resize(src, (out_w, out_h), interpolation=INTER_LINEAR) for single-channel float images
// (half-pixel centers, edge clamping; equals torch interpolate bilinear with align_corners=False).
void resize_linear(const float *src, int sw, int sh, float *dst, int dw, int dh);

// cv2.erode / cv2.dilate with getStructuringElement(MORPH_ELLIPSE, (2r+1, 2r+1)) and the default border
// (pixels outside the image never erode or dilate anything).
Mask erode(const Mask &m, int r);
Mask dilate(const Mask &m, int r);

// cv2.connectedComponents(m, connectivity=8): labels in raster order of each component's first pixel, 0 is
// background. Returns the number of labels including background; areas[label] is filled when non-null.
int connected_components(const Mask &m, std::vector<int32_t> &labels, std::vector<int64_t> *areas = nullptr);

// cv2.distanceTransform(m, DIST_L2, mask_size) with mask_size 3 or 5 (OpenCV's chamfer approximations).
std::vector<float> distance_transform(const Mask &m, int mask_size);

// Python's round(): halves go to the even neighbour.
long round_half_even(double v);

}  // namespace sm
