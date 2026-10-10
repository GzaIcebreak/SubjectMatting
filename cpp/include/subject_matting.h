/*
 * subject_matting: automatic subject matting (BiRefNet_lite -> SAM2.1-tiny -> ViTMatte) on ONNX Runtime.
 *
 * Plain C interface so it can be called from Swift, Kotlin/JNI, C#, Python ctypes, ... The models are the
 * .onnx files written by scripts/export_onnx.py.
 *
 *   char err[256];
 *   sm_context *ctx = sm_create("/path/to/onnx_models", "fp16", 0, err, sizeof err);
 *   if (!ctx) { puts(err); return; }
 *   uint8_t *rgba = malloc((size_t)w * h * 4);
 *   if (sm_matte(ctx, rgb, w, h, w * 3, rgba, NULL, err, sizeof err) != 0) puts(err);
 *   sm_destroy(ctx);
 *
 * A context is not thread safe: use one per thread, or serialize calls. Creating one loads ~200MB of models.
 */
#ifndef SUBJECT_MATTING_H
#define SUBJECT_MATTING_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#if defined(SM_BUILDING_DLL)
#define SM_API __declspec(dllexport)
#elif defined(SM_USING_DLL)
#define SM_API __declspec(dllimport)
#else
#define SM_API
#endif
#else
#define SM_API __attribute__((visibility("default")))
#endif

typedef struct sm_context sm_context;

/* Seconds spent in each stage of the last sm_matte() call. */
typedef struct sm_timings {
    double saliency; /* BiRefNet_lite + prompt points */
    double sam2;     /* SAM2 encoder + decoder + mask selection */
    double matting;  /* trimap + ViTMatte */
    double total;
} sm_timings;

/* Library version, e.g. "0.2.0". */
SM_API const char *sm_version(void);

/*
 * Loads <models_dir>/{birefnet_lite,sam2_encoder,sam2_decoder,vitmatte}.<precision>.onnx.
 * precision: "fp16" (recommended), "fp32" or "int8". num_threads: 0 = ONNX Runtime default (physical cores).
 * Returns NULL on failure and writes a message to err (if err_len > 0).
 */
SM_API sm_context *sm_create(const char *models_dir, const char *precision, int num_threads, char *err, size_t err_len);

/*
 * rgb: width*height 8-bit RGB pixels, rows `stride` bytes apart (stride >= width*3). Apply EXIF orientation
 * before calling. out_rgba: width*height*4 bytes, receives the input colors with the computed alpha.
 * timings may be NULL. Returns 0 on success, non-zero on failure (message in err).
 */
SM_API int sm_matte(sm_context *ctx, const uint8_t *rgb, int width, int height, int stride, uint8_t *out_rgba,
                    sm_timings *timings, char *err, size_t err_len);

SM_API void sm_destroy(sm_context *ctx);

#ifdef __cplusplus
}
#endif

#endif /* SUBJECT_MATTING_H */
