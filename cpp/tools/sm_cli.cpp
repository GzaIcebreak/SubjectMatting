// Command-line front end to the C API: sm_cli [--models DIR] [--precision fp16] [--threads N] in.jpg out.png [...]
// Pairs of input/output paths; the models are loaded once. Reads jpg/png/bmp/tga (stb_image).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "subject_matting.h"

static int usage() {
    std::fprintf(stderr,
                 "usage: sm_cli [--models DIR] [--precision fp16|fp32|int8] [--threads N] in.jpg out.png [in2 out2 ...]\n");
    return 2;
}

int main(int argc, char **argv) {
    std::string models = "onnx_models", precision = "fp16";
    int threads = 0;
    std::vector<std::string> paths;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--models" && i + 1 < argc) models = argv[++i];
        else if (a == "--precision" && i + 1 < argc) precision = argv[++i];
        else if (a == "--threads" && i + 1 < argc) threads = std::atoi(argv[++i]);
        else if (a == "--version") {
            std::printf("%s\n", sm_version());
            return 0;
        }
        else if (a.rfind("--", 0) == 0) return usage();
        else paths.push_back(a);
    }
    if (paths.empty() || paths.size() % 2) return usage();

    char err[512];
    sm_context *ctx = sm_create(models.c_str(), precision.c_str(), threads, err, sizeof err);
    if (!ctx) {
        std::fprintf(stderr, "sm_create: %s\n", err);
        return 1;
    }
    int failed = 0;
    for (size_t i = 0; i < paths.size(); i += 2) {
        int w, h, n;
        // Like matting.common.load_rgb: images with alpha are composited onto white.
        unsigned char *px = stbi_load(paths[i].c_str(), &w, &h, &n, 4);
        if (!px) {
            std::fprintf(stderr, "%s: %s\n", paths[i].c_str(), stbi_failure_reason());
            failed++;
            continue;
        }
        std::vector<uint8_t> rgb(size_t(w) * h * 3), rgba(size_t(w) * h * 4);
        for (size_t p = 0; p < size_t(w) * h; p++) {
            int a = px[p * 4 + 3];
            for (int c = 0; c < 3; c++) {
                int v = px[p * 4 + c];
                rgb[p * 3 + c] = uint8_t(n == 4 || n == 2 ? (v * a + 255 * (255 - a) + 127) / 255 : v);
            }
        }
        stbi_image_free(px);
        sm_timings t;
        if (sm_matte(ctx, rgb.data(), w, h, w * 3, rgba.data(), &t, err, sizeof err) != 0) {
            std::fprintf(stderr, "%s: %s\n", paths[i].c_str(), err);
            failed++;
            continue;
        }
        if (!stbi_write_png(paths[i + 1].c_str(), w, h, 4, rgba.data(), w * 4)) {
            std::fprintf(stderr, "%s: cannot write\n", paths[i + 1].c_str());
            failed++;
            continue;
        }
        std::printf("%s -> %s  %.2fs (saliency %.2f, sam2 %.2f, matting %.2f)\n", paths[i].c_str(), paths[i + 1].c_str(),
                    t.total, t.saliency, t.sam2, t.matting);
    }
    sm_destroy(ctx);
    return failed ? 1 : 0;
}
