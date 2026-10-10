#include <algorithm>
#include <cstring>
#include <exception>
#include <string>

#include "pipeline.h"
#include "subject_matting.h"

struct sm_context {
    sm::Pipeline pipeline;
    sm_context(const char *dir, const char *precision, int threads) : pipeline(dir, precision, threads) {}
};

namespace {

void set_error(char *err, size_t err_len, const std::string &msg) {
    if (!err || err_len == 0) return;
    size_t n = std::min(msg.size(), err_len - 1);
    std::memcpy(err, msg.data(), n);
    err[n] = '\0';
}

}  // namespace

extern "C" {

const char *sm_version(void) { return SM_VERSION; }

sm_context *sm_create(const char *models_dir, const char *precision, int num_threads, char *err, size_t err_len) {
    if (!models_dir) {
        set_error(err, err_len, "models_dir is NULL");
        return nullptr;
    }
    try {
        return new sm_context(models_dir, precision && *precision ? precision : "fp16", num_threads);
    } catch (const std::exception &e) {
        set_error(err, err_len, e.what());
    } catch (...) {
        set_error(err, err_len, "unknown error");
    }
    return nullptr;
}

int sm_matte(sm_context *ctx, const uint8_t *rgb, int width, int height, int stride, uint8_t *out_rgba,
             sm_timings *timings, char *err, size_t err_len) {
    if (!ctx || !rgb || !out_rgba || width <= 0 || height <= 0 || stride < width * 3) {
        set_error(err, err_len, "invalid arguments");
        return 1;
    }
    try {
        sm::ImageU8 img(width, height, 3);
        for (int y = 0; y < height; y++) std::memcpy(img.row(y), rgb + size_t(y) * stride, size_t(width) * 3);
        sm::Timings t;
        sm::ImageU8 out = ctx->pipeline.run(img, &t);
        std::memcpy(out_rgba, out.data.data(), out.data.size());
        if (timings) *timings = {t.saliency, t.sam2, t.matting, t.total};
        return 0;
    } catch (const std::exception &e) {
        set_error(err, err_len, e.what());
    } catch (...) {
        set_error(err, err_len, "unknown error");
    }
    return 2;
}

void sm_destroy(sm_context *ctx) { delete ctx; }

}  // extern "C"
