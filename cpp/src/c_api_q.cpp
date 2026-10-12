#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>

#include "pipeline_q.h"
#include "subject_matting_q.h"

struct smq_context {
    sm::PipelineQ pipeline;
    smq_context(const char *dir, const char *precision, int threads) : pipeline(dir, precision, threads) {}
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

const char *smq_version(void) { return SM_VERSION; }

smq_context *smq_create(const char *models_dir, const char *precision, int num_threads, char *err, size_t err_len) {
    if (!models_dir) {
        set_error(err, err_len, "models_dir is NULL");
        return nullptr;
    }
    // Official ONNX Runtime builds start Microsoft's telemetry unless this is set before the first OrtEnv.
#ifdef _WIN32
    if (!std::getenv("ORT_DISABLE_TELEMETRY")) _putenv_s("ORT_DISABLE_TELEMETRY", "1");
#else
    setenv("ORT_DISABLE_TELEMETRY", "1", 0);
#endif
    try {
        return new smq_context(models_dir, precision && *precision ? precision : "fp16", num_threads);
    } catch (const std::exception &e) {
        set_error(err, err_len, e.what());
    } catch (...) {
        set_error(err, err_len, "unknown error");
    }
    return nullptr;
}

int smq_matte(smq_context *ctx, const uint8_t *rgb, int width, int height, int stride, const smq_subject *subjects,
              int num_subjects, uint8_t *out_rgba, smq_timings *timings, char *err, size_t err_len) {
    if (!ctx || !rgb || !out_rgba || width <= 0 || height <= 0 || stride < width * 3 || num_subjects < 0 ||
        (num_subjects > 0 && !subjects)) {
        set_error(err, err_len, "invalid arguments");
        return SMQ_INVALID_ARGUMENTS;
    }
    try {
        sm::ImageU8 img(width, height, 3);
        for (int y = 0; y < height; y++) std::memcpy(img.row(y), rgb + size_t(y) * stride, size_t(width) * 3);
        std::vector<sm::Subject> subs;
        for (int i = 0; i < num_subjects; i++) {
            const smq_subject &s = subjects[i];
            subs.push_back({s.label ? s.label : "", {s.x1, s.y1, s.x2, s.y2}});
        }
        sm::TimingsQ t;
        sm::ImageU8 out = ctx->pipeline.run(img, subs, &t);
        std::memcpy(out_rgba, out.data.data(), out.data.size());
        if (timings) *timings = {t.sam3, t.matting, t.total};
        return SMQ_OK;
    } catch (const sm::NothingSegmented &e) {
        set_error(err, err_len, e.what());
        return SMQ_NOTHING_SEGMENTED;
    } catch (const std::exception &e) {
        set_error(err, err_len, e.what());
    } catch (...) {
        set_error(err, err_len, "unknown error");
    }
    return SMQ_ERROR;
}

void smq_destroy(smq_context *ctx) { delete ctx; }

}  // extern "C"
