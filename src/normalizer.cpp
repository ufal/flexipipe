#include "normalizer.h"
#include <algorithm>
#include <cctype>

std::string Normalizer::normalize(const std::string& word, const Vocab& vocab, bool conservative) {
    if (word.empty()) {
        return "";
    }
    
    std::string word_lower = word;
    std::transform(word_lower.begin(), word_lower.end(), word_lower.begin(), ::tolower);
    
    // Step 0: Early check - if word appears as a 'reg' value, it's already normalized
    if (word_exists_as_reg(word, vocab)) {
        return "";  // Word is already a normalized form
    }
    
    // Step 1: Check for explicit normalization mapping in vocabulary
    // Try exact case first
    const std::vector<VocabAnalysis>* analyses = vocab.get(word);
    if (analyses && !analyses->empty()) {
        for (const auto& analysis : *analyses) {
            std::string reg = get_reg_from_analysis(analysis);
            if (!reg.empty() && reg != "_" && reg != word) {
                return reg;
            }
        }
    }
    
    // Try lowercase
    analyses = vocab.get(word_lower);
    if (analyses && !analyses->empty()) {
        for (const auto& analysis : *analyses) {
            std::string reg = get_reg_from_analysis(analysis);
            if (!reg.empty() && reg != "_" && reg != word_lower) {
                return reg;
            }
        }
    }
    
    // Conservative mode: only use explicit mappings
    if (conservative) {
        return "";
    }
    
    // Non-conservative mode could add pattern-based normalization here
    // For now, we only support explicit mappings
    
    return "";
}

bool Normalizer::word_exists_as_reg(const std::string& word, const Vocab& vocab) {
    std::string word_lower = word;
    std::transform(word_lower.begin(), word_lower.end(), word_lower.begin(), ::tolower);
    
    // Check all vocab entries for reg values matching word
    for (const auto& entry : vocab.entries) {
        for (const auto& analysis : entry.second) {
            std::string reg = get_reg_from_analysis(analysis);
            if (!reg.empty() && reg != "_") {
                std::string reg_lower = reg;
                std::transform(reg_lower.begin(), reg_lower.end(), reg_lower.begin(), ::tolower);
                if (reg_lower == word_lower) {
                    return true;
                }
            }
        }
    }
    
    return false;
}

std::string Normalizer::get_reg_from_analysis(const VocabAnalysis& analysis) {
    if (!analysis.reg.empty() && analysis.reg != "_") {
        return analysis.reg;
    }
    return "";
}

