#ifndef CONTRACTIONS_H
#define CONTRACTIONS_H

#include <string>
#include <vector>
#include "vocab_loader.h"

class ContractionSplitter {
public:
    // Split contraction into component words
    // Returns vector of split words if contraction detected, empty vector otherwise
    static std::vector<std::string> split(const std::string& form, const Vocab& vocab,
                                         const std::string& upos = "", const std::string& xpos = "");
    
private:
    // Check if base word exists in vocab (with morphological variations)
    static bool base_exists_in_vocab(const std::string& base, const Vocab& vocab);
};

#endif // CONTRACTIONS_H

