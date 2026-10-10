#include "image_ops.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <numeric>

namespace sm {

bool Mask::any() const { return std::any_of(data.begin(), data.end(), [](uint8_t v) { return v != 0; }); }

size_t Mask::count() const { return size_t(std::count_if(data.begin(), data.end(), [](uint8_t v) { return v != 0; })); }

long round_half_even(double v) { return std::lrint(v); }  // default FE_TONEAREST rounds halves to even

// ---------------------------------------------------------------------------------------------------------------
// PIL resampling (Pillow src/libImaging/Resample.c)

namespace {

constexpr int kPrecisionBits = 32 - 8 - 2;

double bilinear_filter(double x) {
    x = std::fabs(x);
    return x < 1.0 ? 1.0 - x : 0.0;
}

double bicubic_filter(double x) {
    constexpr double a = -0.5;
    x = std::fabs(x);
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
    if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
    return 0.0;
}

struct Coeffs {
    int ksize = 0;
    std::vector<int> bounds;  // xmin, count per output pixel
    std::vector<int32_t> kk;  // fixed-point weights, ksize per output pixel
};

Coeffs precompute(int in_size, int out_size, Filter filter) {
    double (*fn)(double) = filter == Filter::Bilinear ? bilinear_filter : bicubic_filter;
    const double support0 = filter == Filter::Bilinear ? 1.0 : 2.0;
    double scale = double(in_size) / out_size;
    double filterscale = scale < 1.0 ? 1.0 : scale;
    double support = support0 * filterscale;
    Coeffs c;
    c.ksize = int(std::ceil(support)) * 2 + 1;
    c.bounds.resize(size_t(out_size) * 2);
    c.kk.assign(size_t(out_size) * c.ksize, 0);
    std::vector<double> k(c.ksize);
    for (int xx = 0; xx < out_size; xx++) {
        double center = (xx + 0.5) * scale;
        double ww = 0.0, ss = 1.0 / filterscale;
        int xmin = int(center - support + 0.5);
        if (xmin < 0) xmin = 0;
        int xmax = int(center + support + 0.5);
        if (xmax > in_size) xmax = in_size;
        xmax -= xmin;
        for (int x = 0; x < xmax; x++) {
            double w = fn((x + xmin - center + 0.5) * ss);
            k[x] = w;
            ww += w;
        }
        for (int x = 0; x < xmax; x++) {
            if (ww != 0.0) k[x] /= ww;
        }
        for (int x = 0; x < xmax; x++) {
            double v = k[x] * (1 << kPrecisionBits);
            c.kk[size_t(xx) * c.ksize + x] = v < 0 ? int32_t(-0.5 + v) : int32_t(0.5 + v);
        }
        c.bounds[xx * 2] = xmin;
        c.bounds[xx * 2 + 1] = xmax;
    }
    return c;
}

inline uint8_t clip8(int64_t v) {
    v >>= kPrecisionBits;
    return uint8_t(v < 0 ? 0 : v > 255 ? 255 : v);
}

}  // namespace

ImageU8 pil_resize(const ImageU8 &src, int out_w, int out_h, Filter filter) {
    if (out_w == src.w && out_h == src.h) return src;
    const int ch = src.channels;
    Coeffs horiz = precompute(src.w, out_w, filter);
    Coeffs vert = precompute(src.h, out_h, filter);
    const bool need_h = out_w != src.w, need_v = out_h != src.h;

    // Horizontal pass over only the rows the vertical pass reads.
    int ybox_first = vert.bounds[0];
    int ybox_last = vert.bounds[out_h * 2 - 2] + vert.bounds[out_h * 2 - 1];
    ImageU8 tmp;
    const ImageU8 *in = &src;
    int row_offset = 0;
    if (need_h) {
        tmp = ImageU8(out_w, ybox_last - ybox_first, ch);
        for (int yy = 0; yy < tmp.h; yy++) {
            const uint8_t *s = src.row(yy + ybox_first);
            uint8_t *d = tmp.row(yy);
            for (int xx = 0; xx < out_w; xx++) {
                int xmin = horiz.bounds[xx * 2], xmax = horiz.bounds[xx * 2 + 1];
                const int32_t *k = &horiz.kk[size_t(xx) * horiz.ksize];
                for (int c = 0; c < ch; c++) {
                    int64_t ss = int64_t(1) << (kPrecisionBits - 1);
                    for (int x = 0; x < xmax; x++) ss += int64_t(s[(x + xmin) * ch + c]) * k[x];
                    d[xx * ch + c] = clip8(ss);
                }
            }
        }
        in = &tmp;
        row_offset = ybox_first;
    }
    if (!need_v) return need_h ? tmp : src;

    ImageU8 out(out_w, out_h, ch);
    for (int yy = 0; yy < out_h; yy++) {
        int ymin = vert.bounds[yy * 2] - row_offset, ymax = vert.bounds[yy * 2 + 1];
        const int32_t *k = &vert.kk[size_t(yy) * vert.ksize];
        uint8_t *d = out.row(yy);
        for (int xx = 0; xx < out_w * ch; xx++) {
            int64_t ss = int64_t(1) << (kPrecisionBits - 1);
            for (int y = 0; y < ymax; y++) ss += int64_t(in->row(y + ymin)[xx]) * k[y];
            d[xx] = clip8(ss);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------------------------------------------
// OpenCV INTER_LINEAR for float images

void resize_linear(const float *src, int sw, int sh, float *dst, int dw, int dh) {
    if (sw == dw && sh == dh) {
        std::copy(src, src + size_t(sw) * sh, dst);
        return;
    }
    auto taps = [](int dsize, int ssize, std::vector<int> &idx, std::vector<float> &frac) {
        double scale = double(ssize) / dsize;
        idx.resize(dsize);
        frac.resize(dsize);
        for (int d = 0; d < dsize; d++) {
            float f = float((d + 0.5) * scale - 0.5);
            int s = int(std::floor(f));
            f -= s;
            if (s < 0) f = 0, s = 0;
            if (s >= ssize - 1) f = 0, s = ssize - 1;
            idx[d] = s;
            frac[d] = f;
        }
    };
    std::vector<int> xi, yi;
    std::vector<float> xf, yf;
    taps(dw, sw, xi, xf);
    taps(dh, sh, yi, yf);
    std::vector<float> row0(dw), row1(dw);
    auto hresize = [&](int sy, float *out) {
        const float *s = src + size_t(sy) * sw;
        for (int x = 0; x < dw; x++) {
            int i = xi[x];
            float f = xf[x];
            out[x] = f == 0 ? s[i] : s[i] * (1.f - f) + s[i + 1] * f;
        }
    };
    int cached0 = -1, cached1 = -1;
    for (int y = 0; y < dh; y++) {
        int s0 = yi[y], s1 = std::min(s0 + 1, sh - 1);
        if (cached0 != s0) {
            if (cached1 == s0) {
                std::swap(row0, row1);
                std::swap(cached0, cached1);
            } else {
                hresize(s0, row0.data());
                cached0 = s0;
            }
        }
        if (cached1 != s1) {
            hresize(s1, row1.data());
            cached1 = s1;
        }
        float f = yf[y];
        float *d = dst + size_t(y) * dw;
        for (int x = 0; x < dw; x++) d[x] = row0[x] * (1.f - f) + row1[x] * f;
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Morphology with OpenCV's elliptical structuring element

namespace {

// Half-width of the kernel row at each dy in [-r, r] (cv::getStructuringElement MORPH_ELLIPSE).
std::vector<int> ellipse_half_widths(int r) {
    std::vector<int> hw(2 * r + 1, -1);
    double inv_r2 = r ? 1.0 / (double(r) * r) : 0;
    for (int i = 0; i <= 2 * r; i++) {
        int dy = i - r;
        if (std::abs(dy) <= r) {
            int dx = int(std::lrint(r * std::sqrt((double(r) * r - double(dy) * dy) * inv_r2)));
            hw[i] = std::min(dx, r);
        }
    }
    return hw;
}

// erode: output 1 iff every in-bounds kernel pixel is 1; dilate: iff any is 1.
Mask morph(const Mask &m, int r, bool is_erode) {
    const int w = m.w, h = m.h;
    // Per-row prefix counts of the pixels that decide the result (zeros for erode, ones for dilate).
    std::vector<int32_t> pre(size_t(w + 1) * h);
    for (int y = 0; y < h; y++) {
        int32_t *p = &pre[size_t(y) * (w + 1)];
        p[0] = 0;
        for (int x = 0; x < w; x++) p[x + 1] = p[x] + ((m.at(x, y) != 0) != is_erode ? 1 : 0);
    }
    std::vector<int> hw = ellipse_half_widths(r);
    Mask out(w, h);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            bool hit = false;  // found a deciding pixel
            for (int i = 0; i <= 2 * r && !hit; i++) {
                int yy = y + i - r;
                if (hw[i] < 0 || yy < 0 || yy >= h) continue;
                int x0 = std::max(x - hw[i], 0), x1 = std::min(x + hw[i], w - 1);
                const int32_t *p = &pre[size_t(yy) * (w + 1)];
                hit = p[x1 + 1] - p[x0] > 0;
            }
            out.at(x, y) = is_erode ? !hit : hit;
        }
    }
    return out;
}

}  // namespace

Mask erode(const Mask &m, int r) { return morph(m, r, true); }
Mask dilate(const Mask &m, int r) { return morph(m, r, false); }

// ---------------------------------------------------------------------------------------------------------------
// Connected components (8-connectivity)

namespace {

int32_t find_root(std::vector<int32_t> &parent, int32_t i) {
    while (parent[i] != i) {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    return i;
}

}  // namespace

int connected_components(const Mask &m, std::vector<int32_t> &labels, std::vector<int64_t> *areas) {
    const int w = m.w, h = m.h;
    labels.assign(size_t(w) * h, 0);
    std::vector<int32_t> parent{0};
    auto unite = [&](int32_t a, int32_t b) {
        a = find_root(parent, a);
        b = find_root(parent, b);
        if (a < b) parent[b] = a;
        else if (b < a) parent[a] = b;
    };
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            if (!m.at(x, y)) continue;
            int32_t best = 0;
            const int nx[4] = {x - 1, x - 1, x, x + 1}, ny[4] = {y, y - 1, y - 1, y - 1};
            for (int k = 0; k < 4; k++) {
                if (nx[k] < 0 || nx[k] >= w || ny[k] < 0) continue;
                int32_t l = labels[size_t(ny[k]) * w + nx[k]];
                if (!l) continue;
                if (!best) best = l;
                else unite(best, l);
            }
            if (!best) {
                best = int32_t(parent.size());
                parent.push_back(best);
            }
            labels[size_t(y) * w + x] = best;
        }
    }
    // OpenCV numbers components in the order its 2x2-block scan first meets them: by row pair, then column pair.
    const int32_t n_prov = int32_t(parent.size());
    std::vector<int64_t> first_block(n_prov, INT64_MAX);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int32_t l = labels[size_t(y) * w + x];
            if (!l) continue;
            int32_t root = find_root(parent, l);
            int64_t key = int64_t(y / 2) * ((w + 1) / 2) + x / 2;
            first_block[root] = std::min(first_block[root], key);
        }
    }
    std::vector<int32_t> roots;
    for (int32_t i = 1; i < n_prov; i++) {
        if (find_root(parent, i) == i) roots.push_back(i);
    }
    std::stable_sort(roots.begin(), roots.end(), [&](int32_t a, int32_t b) { return first_block[a] < first_block[b]; });
    std::vector<int32_t> final_label(n_prov, 0);
    for (size_t i = 0; i < roots.size(); i++) final_label[roots[i]] = int32_t(i + 1);
    const int n = int(roots.size()) + 1;
    if (areas) areas->assign(n, 0);
    for (auto &l : labels) {
        if (!l) {
            if (areas) (*areas)[0]++;
            continue;
        }
        l = final_label[find_root(parent, l)];
        if (areas) (*areas)[l]++;
    }
    return n;
}

// ---------------------------------------------------------------------------------------------------------------
// Distance transform (OpenCV distransform.cpp, DIST_L2 with 3x3 / 5x5 chamfer masks)

std::vector<float> distance_transform(const Mask &m, int mask_size) {
    constexpr int kShift = 16;
    constexpr uint32_t kInit = INT_MAX;
    const float scale = 1.f / (1 << kShift);
    auto fix = [](double v) { return uint32_t(std::lrint(v * (1 << kShift))); };
    const int w = m.w, h = m.h;
    const int border = mask_size == 5 ? 2 : 1;
    const int tw = w + 2 * border;
    std::vector<uint32_t> tmp(size_t(tw) * (h + 2 * border), kInit);
    auto T = [&](int x, int y) -> uint32_t & { return tmp[size_t(y + border) * tw + x + border]; };
    std::vector<float> dist(size_t(w) * h);

    if (mask_size == 3) {
        const uint32_t HV = fix(0.955f), DIAG = fix(1.3693f);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                if (!m.at(x, y)) {
                    T(x, y) = 0;
                    continue;
                }
                uint32_t t0 = T(x - 1, y - 1) + DIAG;
                t0 = std::min(t0, T(x, y - 1) + HV);
                t0 = std::min(t0, T(x + 1, y - 1) + DIAG);
                t0 = std::min(t0, T(x - 1, y) + HV);
                T(x, y) = t0;
            }
        }
        for (int y = h - 1; y >= 0; y--) {
            for (int x = w - 1; x >= 0; x--) {
                uint32_t t0 = T(x, y);
                if (t0 > HV) {
                    t0 = std::min(t0, T(x + 1, y + 1) + DIAG);
                    t0 = std::min(t0, T(x, y + 1) + HV);
                    t0 = std::min(t0, T(x - 1, y + 1) + DIAG);
                    t0 = std::min(t0, T(x + 1, y) + HV);
                    T(x, y) = t0;
                }
                t0 = std::min(t0, kInit);
                dist[size_t(y) * w + x] = float(t0 * scale);
            }
        }
    } else {
        const uint32_t HV = fix(1.0f), DIAG = fix(1.4f), LONG = fix(2.1969f);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                if (!m.at(x, y)) {
                    T(x, y) = 0;
                    continue;
                }
                uint32_t t0 = T(x - 1, y - 2) + LONG;
                t0 = std::min(t0, T(x + 1, y - 2) + LONG);
                t0 = std::min(t0, T(x - 2, y - 1) + LONG);
                t0 = std::min(t0, T(x - 1, y - 1) + DIAG);
                t0 = std::min(t0, T(x, y - 1) + HV);
                t0 = std::min(t0, T(x + 1, y - 1) + DIAG);
                t0 = std::min(t0, T(x + 2, y - 1) + LONG);
                t0 = std::min(t0, T(x - 1, y) + HV);
                T(x, y) = t0;
            }
        }
        for (int y = h - 1; y >= 0; y--) {
            for (int x = w - 1; x >= 0; x--) {
                uint32_t t0 = T(x, y);
                if (t0 > HV) {
                    t0 = std::min(t0, T(x + 1, y + 2) + LONG);
                    t0 = std::min(t0, T(x - 1, y + 2) + LONG);
                    t0 = std::min(t0, T(x + 2, y + 1) + LONG);
                    t0 = std::min(t0, T(x + 1, y + 1) + DIAG);
                    t0 = std::min(t0, T(x, y + 1) + HV);
                    t0 = std::min(t0, T(x - 1, y + 1) + DIAG);
                    t0 = std::min(t0, T(x - 2, y + 1) + LONG);
                    t0 = std::min(t0, T(x + 1, y) + HV);
                    T(x, y) = t0;
                }
                t0 = std::min(t0, kInit);
                dist[size_t(y) * w + x] = float(t0 * scale);
            }
        }
    }
    return dist;
}

}  // namespace sm
