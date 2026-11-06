#include <algorithm>
#include <chrono>
#include <new>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <string>

#include "pipeline_q.h"
#include "subject_matting_q.h"

struct smq_context {
    sm::PipelineQ pipeline;
    smq_context(const char *dir, const char *precision, int threads) : pipeline(dir, precision, threads) {}
};

namespace {

sm::ImageU8 copy_rgb(const uint8_t *rgb, int width, int height, int stride) {
    sm::ImageU8 img(width, height, 3);
    for (int y = 0; y < height; y++) std::memcpy(img.row(y), rgb + size_t(y) * stride, size_t(width) * 3);
    return img;
}

std::vector<sm::Subject> copy_subjects(const smq_subject *subjects, int n) {
    std::vector<sm::Subject> out;
    for (int i = 0; i < n; i++) out.push_back({subjects[i].label ? subjects[i].label : "", {subjects[i].x1, subjects[i].y1, subjects[i].x2, subjects[i].y2}});
    return out;
}

bool valid_image(const uint8_t *rgb, int width, int height, int stride) {
    return rgb && width > 0 && height > 0 && stride >= width * 3;
}

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

int smq_detail(smq_context *ctx, const uint8_t *rgb, int width, int height, int stride,
               uint8_t *out_direct_rgba, uint8_t *out_refined_rgba, smq_timings *timings,
               char *err, size_t err_len) {
    // Check before width*3 (signed overflow) or any pointer arithmetic/allocation.
    const uint64_t limit = uint64_t(std::numeric_limits<ptrdiff_t>::max());
    if (!ctx || !rgb || !out_direct_rgba || !out_refined_rgba || width <= 0 || height <= 0 ||
        width > std::numeric_limits<int>::max() / 3 || stride < int64_t(width) * 3 ||
        uint64_t(width) * uint64_t(height) > limit / 4 ||
        uint64_t(height - 1) * uint64_t(stride) + uint64_t(width) * 3 > limit) {
        set_error(err, err_len, "invalid arguments");
        return SMQ_INVALID_ARGUMENTS;
    }
    try {
        sm::TimingsQ t;
        auto out = ctx->pipeline.detail(copy_rgb(rgb, width, height, stride), &t);
        std::memcpy(out_direct_rgba, out[0].data.data(), out[0].data.size());
        std::memcpy(out_refined_rgba, out[1].data.data(), out[1].data.size());
        if (timings) *timings = {t.sam3, t.matting, t.total};
        return SMQ_OK;
    } catch (const std::exception &e) {
        set_error(err, err_len, e.what());
    } catch (...) {
        set_error(err, err_len, "unknown error");
    }
    return SMQ_ERROR;
}

int smq_matte_plain(smq_context *ctx, const uint8_t *rgb, int width, int height, int stride, uint8_t *out_rgba,
                    smq_timings *timings, char *err, size_t err_len) {
    if (!ctx || !rgb || !out_rgba || width <= 0 || height <= 0 || stride < width * 3) {
        set_error(err, err_len, "invalid arguments");
        return SMQ_INVALID_ARGUMENTS;
    }
    try {
        sm::ImageU8 img(width, height, 3);
        for (int y = 0; y < height; y++) std::memcpy(img.row(y), rgb + size_t(y) * stride, size_t(width) * 3);
        sm::TimingsQ t;
        sm::ImageU8 out = ctx->pipeline.run_plain(img, &t);
        std::memcpy(out_rgba, out.data.data(), out.data.size());
        if (timings) *timings = {t.sam3, t.matting, t.total};
        return SMQ_OK;
    } catch (const sm::NotPlain &e) {
        set_error(err, err_len, e.what());
        return SMQ_NOT_PLAIN;
    } catch (const std::exception &e) {
        set_error(err, err_len, e.what());
    } catch (...) {
        set_error(err, err_len, "unknown error");
    }
    return SMQ_ERROR;
}

int smq_prepare(smq_context *ctx, const uint8_t *rgb, int width, int height, int stride, char *err, size_t err_len) {
    if (!ctx || !rgb || width <= 0 || height <= 0 || stride < width * 3) {
        set_error(err, err_len, "invalid arguments");
        return SMQ_INVALID_ARGUMENTS;
    }
    try {
        sm::ImageU8 img(width, height, 3);
        for (int y = 0; y < height; y++) std::memcpy(img.row(y), rgb + size_t(y) * stride, size_t(width) * 3);
        ctx->pipeline.prepare(img);
        return SMQ_OK;
    } catch (const std::exception &e) {
        set_error(err, err_len, e.what());
    } catch (...) {
        set_error(err, err_len, "unknown error");
    }
    return SMQ_ERROR;
}

int smq_is_plain(const uint8_t *rgb, int width, int height, int stride) {
    if (!rgb || width <= 0 || height <= 0 || stride < width * 3) return 0;
    try {
        sm::ImageU8 img(width, height, 3);
        for (int y = 0; y < height; y++) std::memcpy(img.row(y), rgb + size_t(y) * stride, size_t(width) * 3);
        return sm::is_plain(img) ? 1 : 0;
    } catch (...) {
        return 0;
    }
}

void smq_work_size(int width, int height, int *work_width, int *work_height) {
    int ww = width, wh = height;
    if (width > 0 && height > 0) {
        double scale = 1024.0 / std::max(width, height);
        if (scale < 1.0) ww = int(sm::round_half_even(width * scale)), wh = int(sm::round_half_even(height * scale));
    }
    if (work_width) *work_width = ww;
    if (work_height) *work_height = wh;
}

int smq_candidates(smq_context *ctx, const uint8_t *rgb, int width, int height, int stride, const smq_subject *subjects,
                   int num_subjects, int auto_key, smq_masks **out, char *err, size_t err_len) {
    if (!ctx || !valid_image(rgb, width, height, stride) || !out || num_subjects < 0 || (num_subjects > 0 && !subjects)) {
        set_error(err, err_len, "invalid arguments");
        return SMQ_INVALID_ARGUMENTS;
    }
    *out = nullptr;
    try {
        bool used_key = false;
        std::vector<sm::Candidate> cands =
            ctx->pipeline.candidates(copy_rgb(rgb, width, height, stride), copy_subjects(subjects, num_subjects), auto_key != 0, &used_key);
        smq_masks *m = static_cast<smq_masks *>(std::calloc(1, sizeof(smq_masks)));
        if (!m) throw std::bad_alloc();
        smq_work_size(width, height, &m->width, &m->height);
        m->count = int(cands.size());
        m->used_key = used_key;
        const size_t plane = size_t(m->width) * m->height;
        m->data = static_cast<uint8_t *>(std::malloc(std::max<size_t>(1, plane * cands.size())));
        m->owner = static_cast<int *>(std::malloc(sizeof(int) * std::max<size_t>(1, cands.size())));
        if (!m->data || !m->owner) {
            smq_masks_free(m);
            throw std::bad_alloc();
        }
        for (size_t i = 0; i < cands.size(); i++) {
            std::memcpy(m->data + i * plane, cands[i].mask.data.data(), plane);
            m->owner[i] = cands[i].owner;
        }
        *out = m;
        return SMQ_OK;
    } catch (const std::exception &e) {
        set_error(err, err_len, e.what());
    } catch (...) {
        set_error(err, err_len, "unknown error");
    }
    return SMQ_ERROR;
}

void smq_masks_free(smq_masks *masks) {
    if (!masks) return;
    std::free(masks->data);
    std::free(masks->owner);
    std::free(masks);
}

int smq_remove(smq_context *ctx, const uint8_t *rgb, int width, int height, int stride, uint8_t *mask,
               const smq_subject *regions, int num_regions, char *err, size_t err_len) {
    if (!ctx || !valid_image(rgb, width, height, stride) || !mask || num_regions < 0 || (num_regions > 0 && !regions)) {
        set_error(err, err_len, "invalid arguments");
        return SMQ_INVALID_ARGUMENTS;
    }
    try {
        int ww, wh;
        smq_work_size(width, height, &ww, &wh);
        sm::Mask m(ww, wh);
        for (size_t i = 0; i < m.data.size(); i++) m.data[i] = mask[i] != 0;
        ctx->pipeline.remove(copy_rgb(rgb, width, height, stride), m, copy_subjects(regions, num_regions));
        std::memcpy(mask, m.data.data(), m.data.size());
        return SMQ_OK;
    } catch (const std::exception &e) {
        set_error(err, err_len, e.what());
    } catch (...) {
        set_error(err, err_len, "unknown error");
    }
    return SMQ_ERROR;
}

int smq_matte_mask(smq_context *ctx, const uint8_t *rgb, int width, int height, int stride, const uint8_t *mask,
                   uint8_t *out_rgba, smq_timings *timings, char *err, size_t err_len) {
    if (!ctx || !valid_image(rgb, width, height, stride) || !mask || !out_rgba) {
        set_error(err, err_len, "invalid arguments");
        return SMQ_INVALID_ARGUMENTS;
    }
    try {
        auto t0 = std::chrono::steady_clock::now();
        int ww, wh;
        smq_work_size(width, height, &ww, &wh);
        sm::Mask m(ww, wh);
        for (size_t i = 0; i < m.data.size(); i++) m.data[i] = mask[i] != 0;
        sm::ImageU8 out = ctx->pipeline.matte_mask(copy_rgb(rgb, width, height, stride), m);
        std::memcpy(out_rgba, out.data.data(), out.data.size());
        double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (timings) *timings = {0.0, secs, secs};
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
