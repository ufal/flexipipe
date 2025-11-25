#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <string>
#include <vector>

class Tokenizer {
public:
    // UD-style tokenization (rule-based)
    static std::vector<std::string> tokenize_ud_style(const std::string& text);
    
    // Simple whitespace tokenization
    static std::vector<std::string> tokenize_whitespace(const std::string& text);
};

class SentenceSegmenter {
public:
    // Segment text into sentences
    static std::vector<std::string> segment(const std::string& text);
};

#endif // TOKENIZER_H

