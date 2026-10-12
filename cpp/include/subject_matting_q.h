/*
 * subject_matting_q: matting for subjects named by the caller (pipeline Q). The caller asks a vision-language
 * model (Qwen) what to cut out and passes each subject as an English noun phrase plus a box; EfficientSAM3
 * segments them and ViTMatte produces the alpha. No network access happens in this library.
 *
 *   smq_subject subjects[] = {{"cat", 340, 20, 640, 375}};
 *   smq_context *ctx = smq_create("/path/to/models", "fp16", 0, err, sizeof err);
 *   int rc = smq_matte(ctx, rgb, w, h, w * 3, subjects, 1, out_rgba, NULL, err, sizeof err);
 *   smq_destroy(ctx);
 *
 * Models: <dir>/sam3_{image_encoder,text_encoder,decoder}.<precision>.onnx, vitmatte.fp16.onnx and
 * clip_bpe_merges.txt (scripts/export_sam3_onnx.py). A context is not thread safe.
 */
#ifndef SUBJECT_MATTING_Q_H
#define SUBJECT_MATTING_Q_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#if defined(SM_BUILDING_DLL)
#define SMQ_API __declspec(dllexport)
#else
#define SMQ_API
#endif
#else
#define SMQ_API __attribute__((visibility("default")))
#endif

typedef struct smq_context smq_context;

typedef struct smq_subject {
    const char *label;     /* UTF-8, short English noun phrase, e.g. "cat", "shopping bag" */
    float x1, y1, x2, y2;  /* box in pixels of the input image */
} smq_subject;

typedef struct smq_timings {
    double sam3;     /* image encoder + one text/decoder pass per subject + instance selection */
    double matting;  /* trimap + ViTMatte */
    double total;
} smq_timings;

/* Return codes of smq_matte. */
#define SMQ_OK 0
#define SMQ_INVALID_ARGUMENTS 1
#define SMQ_ERROR 2
#define SMQ_NOTHING_SEGMENTED 3 /* SAM3 found nothing at the given boxes */

SMQ_API const char *smq_version(void);

/* precision: "fp16" (recommended) or "fp32". num_threads: 0 = ONNX Runtime default. NULL on failure (see err). */
SMQ_API smq_context *smq_create(const char *models_dir, const char *precision, int num_threads, char *err, size_t err_len);

/* rgb: width*height 8-bit RGB, rows `stride` bytes apart, EXIF orientation applied. out_rgba: width*height*4. */
SMQ_API int smq_matte(smq_context *ctx, const uint8_t *rgb, int width, int height, int stride, const smq_subject *subjects,
                      int num_subjects, uint8_t *out_rgba, smq_timings *timings, char *err, size_t err_len);

SMQ_API void smq_destroy(smq_context *ctx);

#ifdef __cplusplus
}
#endif

#endif /* SUBJECT_MATTING_Q_H */
