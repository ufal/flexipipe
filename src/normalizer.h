#ifndef NORMALIZER_H
#define NORMALIZER_H

#include <string>
#include "vocab_loader.h"

class Normalizer {
public:
    // Normalize word using vocabulary
    // Returns normalized form if found, empty string otherwise
    static std::string normalize(const std::string& word, const Vocab& vocab, bool conservative = true);
    
private:
    // Check if word exists as a reg value in vocab (already normalized)
    static bool word_exists_as_reg(const std::string& word, const Vocab& vocab);
    
    // Get reg from analysis
    static std::string get_reg_from_analysis(const VocabAnalysis& analysis);
};

#endif // NORMALIZER_H

