// CLIP byte-level BPE tokenizer (port of SimpleTokenizer in EfficientSAM3's tokenizer_ve.py), for SAM3's text
// encoder. Labels come from Qwen as short English phrases, so text cleanup is reduced to what matters for them:
// collapse whitespace and lowercase (ftfy / HTML unescaping are not ported).
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace sm {

class ClipTokenizer {
public:
    // merges_path: clip_bpe_merges.txt written by scripts/export_sam3_onnx.py (one "a b" pair per line, in rank order).
    explicit ClipTokenizer(const std::string &merges_path);

    // <start_of_text> + BPE ids + <end_of_text>, truncated to context_length (keeping the end token), zero padded.
    std::vector<int64_t> tokenize(const std::string &text, int context_length = 16) const;

    // BPE ids of the text alone.
    std::vector<int64_t> encode(const std::string &text) const;

    int64_t sot() const { return sot_; }
    int64_t eot() const { return eot_; }

private:
    std::vector<std::string> bpe(const std::string &token) const;

    std::vector<std::string> byte_encoder_;  // byte -> its unicode stand-in, UTF-8 encoded
    std::unordered_map<std::string, int64_t> encoder_;
    std::unordered_map<std::string, int> ranks_;  // "first second" -> merge rank
    int64_t sot_ = 0, eot_ = 0;
};

}  // namespace sm
