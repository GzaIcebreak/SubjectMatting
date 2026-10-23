// smq_cli [--models DIR] [--precision fp16] in.png out.png "label:x1,y1,x2,y2" [...more subjects]
// smq_cli [--models DIR] --plain in.png out.png   (solid-background fast path; exit 4 when not plain)
// smq_cli [--models DIR] --steps in.png out.png "label:box" ... [--remove "label:box"]...
//     step-by-step API, auto mode, every candidate kept (cpp/tests/parity_q.py --steps)
// smq_cli [--models DIR] --tokenize "text"   (prints the CLIP token ids; used by the tokenizer test)
// Subjects in pixels of the input image. Exit code 3: nothing segmented.
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "clip_tokenizer.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "subject_matting_q.h"

static int usage() {
    std::fprintf(stderr, "usage: smq_cli [--models DIR] [--precision fp16|fp32] in.png out.png \"label:x1,y1,x2,y2\" ...\n"
                         "       smq_cli [--models DIR] --tokenize \"text\"\n");
    return 2;
}

int main(int argc, char **argv) {
    std::string models = "onnx_models_q", precision = "fp16", tokenize;
    bool do_tokenize = false, plain = false, steps = false;
    std::vector<std::string> args, remove_specs;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--models" && i + 1 < argc) models = argv[++i];
        else if (a == "--precision" && i + 1 < argc) precision = argv[++i];
        else if (a == "--tokenize" && i + 1 < argc) do_tokenize = true, tokenize = argv[++i];
        else if (a == "--plain") plain = true;
        else if (a == "--steps") steps = true;
        else if (a == "--remove" && i + 1 < argc) remove_specs.push_back(argv[++i]);
        else if (a.rfind("--", 0) == 0) return usage();
        else args.push_back(a);
    }
    if (do_tokenize) {
        sm::ClipTokenizer tok(models + "/clip_bpe_merges.txt");
        for (int64_t id : tok.tokenize(tokenize, 16)) std::printf("%lld ", (long long)id);
        std::printf("\n");
        return 0;
    }
    if (args.size() < (plain ? 2u : 3u)) return usage();

    std::vector<std::string> labels, remove_labels;
    std::vector<smq_subject> subjects, regions;
    auto parse = [](const std::string &spec, std::vector<std::string> &names, std::vector<smq_subject> &out) {
        size_t colon = spec.rfind(':');
        float b[4];
        if (colon == std::string::npos || std::sscanf(spec.c_str() + colon + 1, "%f,%f,%f,%f", b, b + 1, b + 2, b + 3) != 4)
            return false;
        names.push_back(spec.substr(0, colon));
        out.push_back({nullptr, b[0], b[1], b[2], b[3]});
        return true;
    };
    for (size_t i = 2; i < args.size(); i++)
        if (!parse(args[i], labels, subjects)) return usage();
    for (const std::string &r : remove_specs)
        if (!parse(r, remove_labels, regions)) return usage();
    for (size_t i = 0; i < subjects.size(); i++) subjects[i].label = labels[i].c_str();
    for (size_t i = 0; i < regions.size(); i++) regions[i].label = remove_labels[i].c_str();

    int w, h, n;
    unsigned char *px = stbi_load(args[0].c_str(), &w, &h, &n, 3);
    if (!px) {
        std::fprintf(stderr, "%s: %s\n", args[0].c_str(), stbi_failure_reason());
        return 1;
    }
    char err[512];
    smq_context *ctx = smq_create(models.c_str(), precision.c_str(), 0, err, sizeof err);
    if (!ctx) {
        std::fprintf(stderr, "smq_create: %s\n", err);
        return 1;
    }
    std::vector<uint8_t> rgba(size_t(w) * h * 4);
    smq_timings t;
    int rc;
    if (steps) {
        smq_masks *c = nullptr;
        rc = smq_candidates(ctx, px, w, h, w * 3, subjects.data(), int(subjects.size()), 1, &c, err, sizeof err);
        if (rc == SMQ_OK) {
            const size_t plane = size_t(c->width) * c->height;
            std::vector<uint8_t> mask(plane, 0);
            for (int i = 0; i < c->count; i++)
                for (size_t k = 0; k < plane; k++) mask[k] |= c->data[i * plane + k];
            std::printf("%d candidates (%s)\n", c->count, c->used_key ? "key" : "sam3");
            smq_masks_free(c);
            rc = smq_remove(ctx, px, w, h, w * 3, mask.data(), regions.data(), int(regions.size()), err, sizeof err);
            if (rc == SMQ_OK) rc = smq_matte_mask(ctx, px, w, h, w * 3, mask.data(), rgba.data(), &t, err, sizeof err);
        }
    } else {
        rc = plain ? smq_matte_plain(ctx, px, w, h, w * 3, rgba.data(), &t, err, sizeof err)
                   : smq_matte(ctx, px, w, h, w * 3, subjects.data(), int(subjects.size()), rgba.data(), &t, err, sizeof err);
    }
    stbi_image_free(px);
    smq_destroy(ctx);
    if (rc != SMQ_OK) {
        std::fprintf(stderr, "%s: %s\n", args[0].c_str(), err);
        return rc;
    }
    if (!stbi_write_png(args[1].c_str(), w, h, 4, rgba.data(), w * 4)) return 1;
    std::printf("%s -> %s  %.2fs (sam3 %.2f, matting %.2f)\n", args[0].c_str(), args[1].c_str(), t.total, t.sam3, t.matting);
    return 0;
}
