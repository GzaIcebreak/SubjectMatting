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
#define SMQ_NOT_PLAIN 4         /* smq_matte_plain: the background is not a solid colour */

SMQ_API const char *smq_version(void);

/* precision: "fp16" (recommended) or "fp32". num_threads: 0 = ONNX Runtime default. NULL on failure (see err). */
SMQ_API smq_context *smq_create(const char *models_dir, const char *precision, int num_threads, char *err, size_t err_len);

/* rgb: width*height 8-bit RGB, rows `stride` bytes apart, EXIF orientation applied. out_rgba: width*height*4. */
SMQ_API int smq_matte(smq_context *ctx, const uint8_t *rgb, int width, int height, int stride, const smq_subject *subjects,
                      int num_subjects, uint8_t *out_rgba, smq_timings *timings, char *err, size_t err_len);

/*
 * Auto mode for solid backgrounds (product shots, renders): if the image border is one uniform colour, the subject is
 * everything that differs from it - no subjects needed, no SAM3 run. Returns SMQ_NOT_PLAIN otherwise; then ask the
 * vision-language model and call smq_matte. timings->sam3 holds the keying time.
 */
SMQ_API int smq_matte_plain(smq_context *ctx, const uint8_t *rgb, int width, int height, int stride, uint8_t *out_rgba,
                            smq_timings *timings, char *err, size_t err_len);

/*
 * Optional: runs SAM3's image encoder now (e.g. while waiting for the vision-language model), so the next smq_matte
 * on the same pixels starts from the encoded image. Returns SMQ_OK or SMQ_ERROR.
 */
SMQ_API int smq_prepare(smq_context *ctx, const uint8_t *rgb, int width, int height, int stride, char *err, size_t err_len);

/* 1 if smq_matte_plain would handle this image (solid background), else 0. Needs no context; thread safe. */
SMQ_API int smq_is_plain(const uint8_t *rgb, int width, int height, int stride);

/*
 * Step-by-step API: the caller decides which candidates make up the subject (e.g. by asking a vision-language
 * model) and which regions inside it are see-through. Masks are at "work size" (smq_work_size), one byte per pixel
 * (0 or 1), row-major.
 *
 *   smq_masks *c = NULL;
 *   smq_candidates(ctx, rgb, w, h, stride, subjects, n, 1, &c, err, sizeof err);   // pick some of c->count
 *   ... mask = union of the kept c->data + i * c->width * c->height ...
 *   smq_remove(ctx, rgb, w, h, stride, mask, regions, m, err, sizeof err);
 *   smq_matte_mask(ctx, rgb, w, h, stride, mask, out_rgba, NULL, err, sizeof err);
 *   smq_masks_free(c);
 */
typedef struct smq_masks {
    int count, width, height;
    uint8_t *data;  /* count * width * height bytes */
    int *owner;     /* per mask: index of the subject it was segmented for, -1 for a colour-key piece */
    int used_key;   /* 1 if the masks came from the colour key (solid background) */
} smq_masks;

/* The size masks are at for an input of width x height (long side scaled down to 1024). */
SMQ_API void smq_work_size(int width, int height, int *work_width, int *work_height);

/*
 * Candidate pieces of the subject. auto_key = 1: on a solid background, the colour key's connected pieces (holes
 * filled; subjects are not used); otherwise, and with auto_key = 0, SAM3's instances for each subject. *out is
 * allocated by the library (free it with smq_masks_free); it may hold 0 masks.
 */
SMQ_API int smq_candidates(smq_context *ctx, const uint8_t *rgb, int width, int height, int stride,
                           const smq_subject *subjects, int num_subjects, int auto_key, smq_masks **out, char *err,
                           size_t err_len);
SMQ_API void smq_masks_free(smq_masks *masks);

/* Cuts see-through regions (label + box in input pixels, e.g. "checkerboard") out of a work-size mask, in place. */
SMQ_API int smq_remove(smq_context *ctx, const uint8_t *rgb, int width, int height, int stride, uint8_t *mask,
                       const smq_subject *regions, int num_regions, char *err, size_t err_len);

/* Trimap + ViTMatte for a work-size mask. SMQ_NOTHING_SEGMENTED if the mask is empty. */
SMQ_API int smq_matte_mask(smq_context *ctx, const uint8_t *rgb, int width, int height, int stride, const uint8_t *mask,
                           uint8_t *out_rgba, smq_timings *timings, char *err, size_t err_len);

SMQ_API void smq_destroy(smq_context *ctx);

#ifdef __cplusplus
}
#endif

#endif /* SUBJECT_MATTING_Q_H */
