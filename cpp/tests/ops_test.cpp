// Checks image_ops against PIL / OpenCV outputs written by gen_ops_cases.py. Usage: sm_ops_test <cases dir>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "image_ops.h"

using namespace sm;

static std::string dir;

template <class T>
static std::vector<T> load(const std::string &name, size_t n) {
    std::vector<T> v(n);
    std::ifstream f(dir + "/" + name, std::ios::binary);
    f.read(reinterpret_cast<char *>(v.data()), std::streamsize(n * sizeof(T)));
    if (size_t(f.gcount()) != n * sizeof(T)) throw std::runtime_error("short file " + name);
    return v;
}

static Mask load_mask(const std::string &name, int w, int h) {
    Mask m(w, h);
    m.data = load<uint8_t>(name, size_t(w) * h);
    return m;
}

// Reports mismatches; returns true when they are within tolerance.
template <class A, class B>
static bool compare(const std::string &what, const std::vector<A> &got, const std::vector<B> &exp, double tol) {
    size_t bad = 0;
    double worst = 0;
    for (size_t i = 0; i < got.size(); i++) {
        double d = std::fabs(double(got[i]) - double(exp[i]));
        worst = std::max(worst, d);
        if (d > tol) bad++;
    }
    bool ok = bad == 0;
    std::printf("%s %-40s max|diff| %-10g mismatches %zu/%zu\n", ok ? "PASS" : "FAIL", what.c_str(), worst, bad, got.size());
    return ok;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: sm_ops_test <cases dir>\n");
        return 2;
    }
    dir = argv[1];
    std::ifstream manifest(dir + "/manifest.txt");
    std::string line;
    int failed = 0, total = 0;
    while (std::getline(manifest, line)) {
        if (line.empty()) continue;
        std::istringstream s(line);
        std::string op;
        s >> op;
        bool ok = false;
        if (op == "pil_resize") {
            int sw, sh, c, dw, dh;
            std::string f, in, ex;
            s >> sw >> sh >> c >> dw >> dh >> f >> in >> ex;
            ImageU8 src(sw, sh, c);
            src.data = load<uint8_t>(in, src.data.size());
            ImageU8 got = pil_resize(src, dw, dh, f == "bilinear" ? Filter::Bilinear : Filter::Bicubic);
            ok = compare(line.substr(0, line.find(" r")), got.data, load<uint8_t>(ex, size_t(dw) * dh * c), 0);
        } else if (op == "resize_linear") {
            int sw, sh, dw, dh;
            std::string in, ex;
            s >> sw >> sh >> dw >> dh >> in >> ex;
            std::vector<float> src = load<float>(in, size_t(sw) * sh), got(size_t(dw) * dh);
            resize_linear(src.data(), sw, sh, got.data(), dw, dh);
            ok = compare(op + " " + std::to_string(sw) + "x" + std::to_string(sh) + "->" + std::to_string(dw) + "x" +
                             std::to_string(dh),
                         got, load<float>(ex, got.size()), 1e-4);
        } else if (op == "erode" || op == "dilate") {
            int w, h, r;
            std::string in, ex;
            s >> w >> h >> r >> in >> ex;
            Mask m = load_mask(in, w, h);
            Mask got = op == "erode" ? erode(m, r) : dilate(m, r);
            ok = compare(op + " r=" + std::to_string(r) + " " + in, got.data, load<uint8_t>(ex, got.data.size()), 0);
        } else if (op == "components") {
            int w, h;
            std::string in, ex;
            s >> w >> h >> in >> ex;
            std::vector<int32_t> labels;
            connected_components(load_mask(in, w, h), labels);
            ok = compare("components " + in, labels, load<int32_t>(ex, labels.size()), 0);
        } else if (op == "distance") {
            int w, h, ms;
            std::string in, ex;
            s >> w >> h >> ms >> in >> ex;
            std::vector<float> got = distance_transform(load_mask(in, w, h), ms);
            ok = compare("distance " + std::to_string(ms) + "x" + std::to_string(ms) + " " + in, got,
                         load<float>(ex, got.size()), 1e-4);
        } else {
            std::printf("SKIP unknown op %s\n", op.c_str());
            continue;
        }
        total++;
        failed += !ok;
    }
    std::printf("%d/%d passed\n", total - failed, total);
    return failed ? 1 : 0;
}
