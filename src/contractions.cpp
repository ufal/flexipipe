#include "contractions.h"
#include <algorithm>
#include <cctype>

std::vector<std::string> ContractionSplitter::split(const std::string& form, const Vocab& vocab,
                                                    const std::string& upos, const std::string& xpos) {
    std::vector<std::string> result;
    
    if (form.empty()) {
        return result;
    }
    
    std::string form_lower = form;
    std::transform(form_lower.begin(), form_lower.end(), form_lower.begin(), ::tolower);
    
    // Step 1: Check for explicit parts in vocabulary (most reliable)
    const std::vector<VocabAnalysis>* analyses = vocab.get(form);
    if (!analyses) {
        analyses = vocab.get(form_lower);
    }
    
    if (analyses && !analyses->empty()) {
        // First, check if any analysis has parts - prefer those
        const VocabAnalysis* best_analysis = nullptr;
        int max_count = 0;
        
        for (const auto& analysis : *analyses) {
            if (!analysis.parts.empty() && analysis.count > max_count) {
                best_analysis = &analysis;
                max_count = analysis.count;
            }
        }
        
        if (best_analysis && !best_analysis->parts.empty()) {
            // Found explicit parts - return them
            return best_analysis->parts;
        }
    }
    
    // Step 2: Pattern-based splitting could be added here
    // For now, we only support explicit parts from vocabulary
    
    return result;
}

bool ContractionSplitter::base_exists_in_vocab(const std::string& base, const Vocab& vocab) {
    if (base.empty()) {
        return false;
    }
    
    // Try exact match
    if (vocab.get(base)) {
        return true;
    }
    
    // Try lowercase
    std::string base_lower = base;
    std::transform(base_lower.begin(), base_lower.end(), base_lower.begin(), ::tolower);
    if (vocab.get(base_lower)) {
        return true;
    }
    
    // Try with dropped final 'r' (common in Portuguese/Spanish before clitics)
    if (base.length() > 2 && base.back() == 'r') {
        std::string base_no_r = base.substr(0, base.length() - 1);
        if (vocab.get(base_no_r)) {
            return true;
        }
        base_no_r = base_lower.substr(0, base_lower.length() - 1);
        if (vocab.get(base_no_r)) {
            return true;
        }
    }
    
    return false;
}

