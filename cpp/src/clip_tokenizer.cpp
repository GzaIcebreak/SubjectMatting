#include "clip_tokenizer.h"

#include <climits>
#include <fstream>
#include <stdexcept>

namespace sm {

namespace {

std::string utf8(uint32_t cp) {
    std::string s;
    if (cp < 0x80) {
        s += char(cp);
    } else if (cp < 0x800) {
        s += char(0xC0 | (cp >> 6));
        s += char(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        s += char(0xE0 | (cp >> 12));
        s += char(0x80 | ((cp >> 6) & 0x3F));
        s += char(0x80 | (cp & 0x3F));
    } else {
        s += char(0xF0 | (cp >> 18));
        s += char(0x80 | ((cp >> 12) & 0x3F));
        s += char(0x80 | ((cp >> 6) & 0x3F));
        s += char(0x80 | (cp & 0x3F));
    }
    return s;
}

// GPT-2 / CLIP bytes_to_unicode: printable bytes map to themselves, the rest to 256+n.
std::vector<uint32_t> bytes_to_unicode() {
    std::vector<int> bs;
    for (int b = '!'; b <= '~'; b++) bs.push_back(b);
    for (int b = 0xA1; b <= 0xAC; b++) bs.push_back(b);
    for (int b = 0xAE; b <= 0xFF; b++) bs.push_back(b);
    std::vector<uint32_t> cs(bs.begin(), bs.end());
    std::vector<bool> present(256, false);
    for (int b : bs) present[b] = true;
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (!present[b]) {
            bs.push_back(b);
            cs.push_back(256 + n++);
        }
    }
    // Return in the order of bs (the vocabulary order), as pairs packed: index i -> (byte bs[i], codepoint cs[i]).
    std::vector<uint32_t> packed(512);
    for (size_t i = 0; i < 256; i++) {
        packed[i] = uint32_t(bs[i]);
        packed[256 + i] = cs[i];
    }
    return packed;
}

// Decodes one UTF-8 character starting at s[i]; returns its byte length.
size_t utf8_len(unsigned char c) { return c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1; }

enum class Cls { Space, Letter, Number, Other };

Cls classify(const std::string &s, size_t i) {
    unsigned char c = s[i];
    if (c < 0x80) {
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v') return Cls::Space;
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) return Cls::Letter;
        if (c >= '0' && c <= '9') return Cls::Number;
        return Cls::Other;
    }
    return Cls::Letter;  // non-ASCII: treated as letters (labels are English; good enough for accents)
}

}  // namespace

ClipTokenizer::ClipTokenizer(const std::string &merges_path) {
    std::vector<uint32_t> packed = bytes_to_unicode();
    byte_encoder_.assign(256, "");
    std::vector<std::string> vocab;
    for (size_t i = 0; i < 256; i++) {
        byte_encoder_[packed[i]] = utf8(packed[256 + i]);
        vocab.push_back(utf8(packed[256 + i]));
    }
    for (size_t i = 0; i < 256; i++) vocab.push_back(vocab[i] + "</w>");

    std::ifstream f(merges_path);
    if (!f) throw std::runtime_error("cannot open " + merges_path);
    std::string line;
    int rank = 0;
    while (std::getline(f, line)) {
        size_t sp = line.find(' ');
        if (line.empty() || sp == std::string::npos) continue;
        ranks_[line] = rank++;
        vocab.push_back(line.substr(0, sp) + line.substr(sp + 1));
    }
    if (rank != 48894) throw std::runtime_error("unexpected number of BPE merges in " + merges_path);
    vocab.push_back("<start_of_text>");
    vocab.push_back("<end_of_text>");
    for (size_t i = 0; i < vocab.size(); i++) encoder_.emplace(vocab[i], int64_t(i));
    sot_ = encoder_.at("<start_of_text>");
    eot_ = encoder_.at("<end_of_text>");
}

std::vector<std::string> ClipTokenizer::bpe(const std::string &token) const {
    // Split into characters (UTF-8), the last one with the end-of-word marker.
    std::vector<std::string> word;
    for (size_t i = 0; i < token.size();) {
        size_t n = utf8_len(static_cast<unsigned char>(token[i]));
        word.push_back(token.substr(i, n));
        i += n;
    }
    if (word.empty()) return word;
    word.back() += "</w>";
    while (word.size() > 1) {
        int best = INT_MAX;
        size_t at = 0;
        for (size_t i = 0; i + 1 < word.size(); i++) {
            auto it = ranks_.find(word[i] + " " + word[i + 1]);
            if (it != ranks_.end() && it->second < best) best = it->second, at = i;
        }
        if (best == INT_MAX) break;
        // Merge every occurrence of the best pair, left to right.
        const std::string first = word[at], second = word[at + 1];
        std::vector<std::string> merged;
        for (size_t i = 0; i < word.size();) {
            if (i + 1 < word.size() && word[i] == first && word[i + 1] == second) {
                merged.push_back(first + second);
                i += 2;
            } else {
                merged.push_back(word[i]);
                i += 1;
            }
        }
        word.swap(merged);
    }
    return word;
}

std::vector<int64_t> ClipTokenizer::encode(const std::string &text) const {
    // Clean: lowercase ASCII, collapse whitespace, trim.
    std::string clean;
    bool space = false;
    for (unsigned char c : text) {
        bool ws = c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
        if (ws) {
            space = !clean.empty();
            continue;
        }
        if (space) clean += ' ';
        space = false;
        clean += char(c >= 'A' && c <= 'Z' ? c + 32 : c);
    }

    // Pre-tokenize like the upstream regex: 's|'t|'re|'ve|'m|'ll|'d | letters+ | one digit | other+ (no spaces).
    std::vector<std::string> pieces;
    static const char *contractions[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
    for (size_t i = 0; i < clean.size();) {
        Cls c = classify(clean, i);
        if (c == Cls::Space) {
            i++;
            continue;
        }
        bool matched = false;
        for (const char *con : contractions) {
            size_t n = std::char_traits<char>::length(con);
            if (clean.compare(i, n, con) == 0) {
                pieces.push_back(clean.substr(i, n));
                i += n;
                matched = true;
                break;
            }
        }
        if (matched) continue;
        size_t j = i;
        if (c == Cls::Number) {
            j = i + 1;
        } else {
            while (j < clean.size() && classify(clean, j) == c) j += utf8_len(static_cast<unsigned char>(clean[j]));
        }
        pieces.push_back(clean.substr(i, j - i));
        i = j;
    }

    std::vector<int64_t> ids;
    for (const std::string &piece : pieces) {
        std::string mapped;
        for (unsigned char b : piece) mapped += byte_encoder_[b];
        for (const std::string &t : bpe(mapped)) {
            auto it = encoder_.find(t);
            if (it != encoder_.end()) ids.push_back(it->second);
        }
    }
    return ids;
}

std::vector<int64_t> ClipTokenizer::tokenize(const std::string &text, int context_length) const {
    std::vector<int64_t> ids{sot_};
    std::vector<int64_t> body = encode(text);
    ids.insert(ids.end(), body.begin(), body.end());
    ids.push_back(eot_);
    if (int(ids.size()) > context_length) {
        ids.resize(context_length);
        ids.back() = eot_;
    }
    ids.resize(context_length, 0);
    return ids;
}

}  // namespace sm
