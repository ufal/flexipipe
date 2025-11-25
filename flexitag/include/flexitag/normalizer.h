#pragma once

#include "lexicon.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace flexitag {

class Normalizer {
public:
    // Initialize normalizer with lexicon (extracts patterns from lexicon)
    // Only enabled if lexicon has normalization data (entries with reg fields)
    explicit Normalizer(const Lexicon* lexicon);
    
    // Set debug flag for verbose output
    void set_debug(bool debug) { debug_ = debug; }
    
    // Check if normalization is enabled (lexicon has normalization data)
    bool is_enabled() const { return enabled_; }
    
    // Normalize word using enhanced strategies
    // Returns normalized form if found, empty string otherwise
    // conservative: if true, only use explicit mappings and morphological variations
    //              if false, also use pattern-based substitutions (aggressive mode)
    std::string normalize(const std::string& word, bool conservative = true) const;
    
    // Get inflection suffixes derived from vocab
    const std::vector<std::string>& inflection_suffixes() const { return inflection_suffixes_; }
    
    // Get frequent substitution patterns (for aggressive mode)
    // Returns map of "from_to" -> count (e.g., "y_i" -> 5)
    const std::unordered_map<std::string, int>& substitution_patterns() const {
        return substitution_patterns_;
    }
    
    // Check if normalization is capitalization-only (e.g., "de" -> "De")
    bool is_capitalization_only(const std::string& form, const std::string& reg) const;
    
    // Check if normalization is for punctuation (e.g., punctuation -> punctuation)
    bool is_punctuation_normalization(const std::string& form, const std::string& reg) const;
    
    // Get reg from lexicon entry (public for use by tagger)
    // Returns the most frequent reg value that is more frequent than non-normalized form
    // Skips capitalization-only and punctuation normalizations
    std::string get_reg_from_entry(const LexiconItem* item) const;

private:
    const Lexicon* lexicon_;
    bool enabled_;
    bool debug_ = false;
    std::vector<std::string> inflection_suffixes_;
    std::unordered_map<std::string, int> substitution_patterns_;  // "from_to" -> count (e.g., "y_i" -> 5)
    int min_pattern_count_ = 10;  // Minimum count for a pattern to be considered frequent (increased from 3)
    
    // Derive inflection suffixes from vocab form->reg mappings
    void derive_inflection_suffixes();
    
    // Extract substitution patterns from vocab form->reg mappings
    void extract_substitution_patterns();
    
    // Check if word exists as a reg value in lexicon (already normalized)
    bool word_exists_as_reg(const std::string& word) const;
    
    // Apply morphological variation normalization
    // If "mysterio" -> "misterio" exists, also normalize "mysterios" -> "misterios"
    std::string apply_morphological_variation(const std::string& word) const;
    
    // Apply pattern-based substitution (aggressive mode)
    std::string apply_pattern_substitution(const std::string& word) const;
};

} // namespace flexitag

