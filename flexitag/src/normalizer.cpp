#include "flexitag/normalizer.h"
#include "flexitag/unicode_utils.h"

#include <algorithm>
#include <cctype>
#include <utility>
#include <iostream>

namespace {
// Using ICU-based Unicode utilities instead of manual UTF-8 handling
using flexitag::unicode::char_count;
using flexitag::unicode::char_at;
using flexitag::unicode::char_byte_range;
using flexitag::unicode::replace_char;
using flexitag::unicode::char_exists_in_string;
using flexitag::unicode::split_pattern;
using flexitag::unicode::to_lower;
using flexitag::unicode::to_upper;
using flexitag::unicode::to_unicode_string;

// Helper function to check if normalization is capitalization-only
bool is_capitalization_only(const std::string& form, const std::string& reg) {
    if (form.empty() || reg.empty()) {
        return false;
    }
    // Use Unicode-aware lowercase comparison
    std::string form_lower = to_lower(form);
    std::string reg_lower = to_lower(reg);
    return form_lower == reg_lower;
}

// Helper function to check if normalization is for punctuation
bool is_punctuation_normalization(const std::string& form, const std::string& reg) {
    if (form.empty() || reg.empty()) {
        return false;
    }
    
    // Use ICU to check if characters are punctuation
    icu::UnicodeString form_ustr = to_unicode_string(form);
    icu::UnicodeString reg_ustr = to_unicode_string(reg);
    
    // If both are single characters, check if they're punctuation
    if (form_ustr.length() == 1 && reg_ustr.length() == 1) {
        UChar32 form_char = form_ustr.char32At(0);
        UChar32 reg_char = reg_ustr.char32At(0);
        
        // Check if both are punctuation characters
        UCharCategory form_cat = static_cast<UCharCategory>(u_getIntPropertyValue(form_char, UCHAR_GENERAL_CATEGORY));
        UCharCategory reg_cat = static_cast<UCharCategory>(u_getIntPropertyValue(reg_char, UCHAR_GENERAL_CATEGORY));
        
        bool form_is_punct = (form_cat == U_OTHER_PUNCTUATION || 
                              form_cat == U_INITIAL_PUNCTUATION || 
                              form_cat == U_FINAL_PUNCTUATION);
        bool reg_is_punct = (reg_cat == U_OTHER_PUNCTUATION || 
                            reg_cat == U_INITIAL_PUNCTUATION || 
                            reg_cat == U_FINAL_PUNCTUATION);
        
        // If both are punctuation, it's a punctuation normalization
        if (form_is_punct && reg_is_punct) {
            return true;
        }
    }
    
    // Also check if form is a single punctuation character (normalizing punctuation to something else)
    if (form_ustr.length() == 1) {
        UChar32 form_char = form_ustr.char32At(0);
        UCharCategory form_cat = static_cast<UCharCategory>(u_getIntPropertyValue(form_char, UCHAR_GENERAL_CATEGORY));
        bool form_is_punct = (form_cat == U_OTHER_PUNCTUATION || 
                              form_cat == U_INITIAL_PUNCTUATION || 
                              form_cat == U_FINAL_PUNCTUATION);
        if (form_is_punct) {
            return true;
        }
    }
    
    return false;
}

// Helper function to get word frequency from lexicon
int get_word_frequency(const flexitag::Lexicon* lexicon, const std::string& word) {
    if (!lexicon) {
        return 0;
    }
    const flexitag::LexiconItem* item = lexicon->find(word);
    if (!item) {
        item = lexicon->find_lower(word);
    }
    if (!item || item->tokens.empty()) {
        return 0;
    }
    // Sum up counts from all entries
    int total_count = 0;
    for (const auto& token : item->tokens) {
        for (const auto& entry : token.entries) {
            total_count += entry.count;
        }
    }
    return total_count;
}
}

namespace flexitag {

Normalizer::Normalizer(const Lexicon* lexicon) : lexicon_(lexicon), enabled_(false) {
    if (!lexicon_) {
        return;
    }
    
    // Check if lexicon has any normalization data (entries with reg fields)
    enabled_ = lexicon_->has_normalizations();
    
    if (enabled_) {
        derive_inflection_suffixes();
        extract_substitution_patterns();
    }
}

void Normalizer::derive_inflection_suffixes() {
    inflection_suffixes_.clear();
    
    if (!lexicon_) {
        return;
    }
    
    // Get all normalization mappings from lexicon
    auto mappings = lexicon_->get_normalization_mappings();
    
    // Derive inflection suffixes from form->reg mappings
    // Look for entries where both form and reg end with the same suffix
    // and the stems differ (e.g., "mysterio"->"misterio" while preserving plural 's')
    std::unordered_map<std::string, int> suffix_counts;
    const int max_suffix_len = 4;
    const int min_count = 5;  // Increased from 3 to require more evidence
    
    for (const auto& [form, reg] : mappings) {
        // CRITICAL: Use Unicode-aware lowercase conversion
        std::string form_lower = to_lower(form);
        std::string reg_lower = to_lower(reg);
        
        if (form_lower == reg_lower) {
            continue;
        }
        
        int max_k = std::min({max_suffix_len, static_cast<int>(form_lower.length()), static_cast<int>(reg_lower.length())});
        for (int k = 1; k <= max_k; ++k) {
            if (form_lower.length() < k || reg_lower.length() < k) {
                continue;
            }
            std::string suffix = form_lower.substr(form_lower.length() - k);
            // Check if reg_lower ends with suffix (C++17 compatible, not using C++20 ends_with)
            if (reg_lower.length() >= k && reg_lower.substr(reg_lower.length() - k) == suffix) {
                std::string form_stem = form_lower.substr(0, form_lower.length() - k);
                std::string reg_stem = reg_lower.substr(0, reg_lower.length() - k);
                if (form_stem != reg_stem) {
                    suffix_counts[suffix]++;
                }
            }
        }
    }
    
    // Keep suffixes that appear at least min_count times
    for (const auto& [suffix, count] : suffix_counts) {
        if (count >= min_count) {
            inflection_suffixes_.push_back(suffix);
        }
    }
    
    // Sort by frequency (most frequent first), then by length (longest first)
    std::sort(inflection_suffixes_.begin(), inflection_suffixes_.end(),
              [&suffix_counts](const std::string& a, const std::string& b) {
                  int count_a = suffix_counts.at(a);
                  int count_b = suffix_counts.at(b);
                  if (count_a != count_b) {
                      return count_a > count_b;
                  }
                  if (a.length() != b.length()) {
                      return a.length() > b.length();
                  }
                  return a < b;
              });
}

void Normalizer::extract_substitution_patterns() {
    substitution_patterns_.clear();
    
    if (!lexicon_) {
        return;
    }
    
    // Get all normalization mappings from lexicon
    auto mappings = lexicon_->get_normalization_mappings();
    
    // Extract character substitution patterns
    // Count how often each character substitution is used
    for (const auto& [form, reg] : mappings) {
        // CRITICAL: Use Unicode-aware lowercase conversion, not std::transform with ::tolower
        // std::transform with ::tolower operates byte-by-byte and corrupts multi-byte UTF-8 characters
        std::string form_lower = to_lower(form);
        std::string reg_lower = to_lower(reg);
        
        // Compare Unicode characters (code points) using ICU
        size_t form_chars = char_count(form_lower);
        size_t reg_chars = char_count(reg_lower);
        if (form_chars == reg_chars) {
            // Same character count: character substitution
            for (size_t char_idx = 0; char_idx < form_chars; ++char_idx) {
                std::string form_char = char_at(form_lower, char_idx);
                std::string reg_char = char_at(reg_lower, char_idx);
                if (form_char != reg_char && !form_char.empty() && !reg_char.empty()) {
                    // Store as "variant_standard" string for hashing
                    // CRITICAL: For normalization, we want to go from variant (form) to standard (reg)
                    // Pattern "form_char_reg_char" means "replace form_char (variant) with reg_char (standard)"
                    // This is correct: if we see form_char (variant), replace with reg_char (standard)
                    // Example: "coraçón" -> "corazón" gives pattern "ç_z" (replace ç with z)
                    std::string pattern = form_char + "_" + reg_char;
                    substitution_patterns_[pattern]++;
                }
            }
        }
    }
}

std::string Normalizer::normalize(const std::string& word, bool conservative) const {
    if (word.empty() || !lexicon_ || !enabled_) {
        return "";
    }
    
    // CRITICAL: Use Unicode-aware lowercase conversion
    std::string word_lower = to_lower(word);
    
    // Step 0: Early check - if word appears as a 'reg' value, it's already normalized
    if (word_exists_as_reg(word)) {
        return "";  // Word is already normalized
    }
    
    // Step 1: Check for explicit normalization mapping in lexicon
    // Try exact match first, then lowercase
    const LexiconItem* item = lexicon_->find(word);
    if (!item) {
        item = lexicon_->find_lower(word);
    }
    
    if (item) {
        std::string reg = get_reg_from_entry(item);
        // RESTRICTION 1: Only use explicit reg if it exists and is different from word
        // Also verify that the reg value exists in the vocab (as key or as another reg value)
        if (!reg.empty() && reg != "_" && reg != "--" && reg != word) {
            // Skip TEITOK-specific "--" normalization (suppression code)
            if (reg == "--") {
                return "";
            }
            
            // Skip capitalization-only normalizations (position-dependent, not real normalization)
            if (is_capitalization_only(word, reg)) {
                if (debug_) {
                    std::cerr << "[normalizer] normalize: skipping capitalization-only normalization '" << word << "' -> '" << reg << "'\n";
                }
                return "";
            }
            
            // Skip punctuation normalizations
            if (is_punctuation_normalization(word, reg)) {
                if (debug_) {
                    std::cerr << "[normalizer] normalize: skipping punctuation normalization '" << word << "' -> '" << reg << "'\n";
                }
                return "";
            }
            
            // RESTRICTION 2: Verify the normalized form exists in the vocab
            bool reg_exists = lexicon_->find(reg) || lexicon_->find_lower(reg);
            if (!reg_exists) {
                // Check if reg exists as a reg value in the lexicon
                auto mappings = lexicon_->get_normalization_mappings();
                for (const auto& [form, reg_val] : mappings) {
                    if (to_lower(reg_val) == to_lower(reg)) {
                        reg_exists = true;
                        break;
                    }
                }
            }
            
            if (reg_exists) {
                // RESTRICTION 3: For frequent words, only normalize if normalized form is much more frequent
                // This prevents normalizing common words like "de", "el" that are sometimes normalized
                int word_freq = get_word_frequency(lexicon_, word);
                int reg_freq = get_word_frequency(lexicon_, reg);
                
                // If word is frequent (>= 100), require normalized form to be at least 5x more frequent
                // and non-normalized form to be relatively rare (< 10% of normalized form)
                if (word_freq >= 100) {
                    if (reg_freq < word_freq * 5 || word_freq > reg_freq * 0.1) {
                        if (debug_) {
                            std::cerr << "[normalizer] normalize: word='" << word << "' (freq=" << word_freq 
                                      << ") -> '" << reg << "' (freq=" << reg_freq 
                                      << ") rejected: normalized form not frequent enough\n";
                        }
                        return "";
                    }
                }
                
                // CRITICAL: If the reg value itself exists as a form that has a normalization,
                // we should use that form's normalization instead (e.g., "corazon" -> "corazón")
                const LexiconItem* reg_item = lexicon_->find(reg);
                if (!reg_item) {
                    reg_item = lexicon_->find_lower(reg);
                }
                if (reg_item) {
                    std::string reg_reg = get_reg_from_entry(reg_item);
                    if (!reg_reg.empty() && reg_reg != "_" && reg_reg != "--" && reg_reg != reg) {
                        // Skip capitalization-only, punctuation, and "--" normalizations
                        if (!is_capitalization_only(reg, reg_reg) && 
                            !is_punctuation_normalization(reg, reg_reg) && 
                            reg_reg != "--") {
                            if (debug_) {
                                std::cerr << "[normalizer] normalize: reg='" << reg << "' itself has normalization '" << reg_reg << "', using that instead\n";
                            }
                            return reg_reg;
                        }
                    }
                }
                
                return reg;
            } else if (debug_) {
                std::cerr << "[normalizer] normalize: word='" << word << "' has reg='" << reg << "' but reg doesn't exist in vocab, ignoring\n";
            }
        }
    }
    
    // Also check if word matches any reg value in the lexicon (reverse lookup)
    // This handles cases where the form isn't a key but matches a reg value
    // For example, if vocab has "corazon" with reg="corazón", and we're looking for "corazón"
    // We can't easily do a full reverse lookup, but we can check if word exists as-is
    // This is a performance optimization - we'll rely on pattern substitution for most cases
    
    // Step 2: Check morphological variations of known mappings
    if (conservative) {
        std::string normalized = apply_morphological_variation(word);
        if (!normalized.empty()) {
            return normalized;
        }
    }
    
    // Step 3: Pattern-based substitution (aggressive mode)
    if (!conservative) {
        std::string normalized = apply_pattern_substitution(word);
        if (!normalized.empty()) {
            return normalized;
        }
    }
    
    return "";
}

bool Normalizer::word_exists_as_reg(const std::string& word) const {
    if (!lexicon_) {
        return false;
    }
    
    // CRITICAL: Use Unicode-aware lowercase conversion
    std::string word_lower = to_lower(word);
    
    // Check if word exists as a reg value (normalized form) anywhere in the lexicon
    // This means the word is already normalized and shouldn't be normalized further
    // We iterate through all lexicon items to check if any have this word as their reg value
    auto mappings = lexicon_->get_normalization_mappings();
    
    // Check if word appears as a reg value (but NOT as a form that has normalization)
    // Forms that have normalizations should NOT be considered already normalized
    bool is_reg_value = false;
    bool is_form_with_reg = false;
    
    for (const auto& [form, reg] : mappings) {
        // CRITICAL: Use Unicode-aware lowercase conversion
        std::string form_lower = to_lower(form);
        std::string reg_lower = to_lower(reg);
        
        if (reg_lower == word_lower) {
            is_reg_value = true;
        }
        if (form_lower == word_lower) {
            is_form_with_reg = true;
        }
    }
    
    // Word is normalized if it's a reg value but NOT a form that has normalization
    // (forms with normalization need to be normalized themselves)
    return is_reg_value && !is_form_with_reg;
}

bool Normalizer::is_capitalization_only(const std::string& form, const std::string& reg) const {
    return ::is_capitalization_only(form, reg);
}

bool Normalizer::is_punctuation_normalization(const std::string& form, const std::string& reg) const {
    return ::is_punctuation_normalization(form, reg);
}

std::string Normalizer::get_reg_from_entry(const LexiconItem* item) const {
    if (!item || item->tokens.empty()) {
        return "";
    }
    
    // Collect all reg values with their frequencies, skipping capitalization-only and "--"
    std::unordered_map<std::string, int> reg_counts;
    int non_normalized_count = 0;
    
    for (const auto& token : item->tokens) {
        for (const auto& entry : token.entries) {
            if (entry.reg.empty() || entry.reg == "_" || entry.reg == "--") {
                // Count non-normalized entries (no reg or reg is empty/underscore/--)
                non_normalized_count += entry.count;
                    } else {
                        // Skip capitalization-only normalizations
                        std::string form_lower = to_lower(item->form);
                        std::string reg_lower = to_lower(entry.reg);
                        if (form_lower == reg_lower) {
                            // Capitalization-only, count as non-normalized (it's the same word)
                            non_normalized_count += entry.count;
                            continue;
                        }
                        // Skip punctuation normalizations
                        if (is_punctuation_normalization(item->form, entry.reg)) {
                            // Punctuation normalization, count as non-normalized
                            non_normalized_count += entry.count;
                            continue;
                        }
                        // Count this reg value
                        reg_counts[entry.reg] += entry.count;
                    }
        }
    }
    
    // Find the most frequent reg value that is more frequent than non-normalized form
    std::string best_reg = "";
    int best_count = 0;
    
    // Calculate total count for all reg values
    int total_reg_count = 0;
    for (const auto& [reg, count] : reg_counts) {
        total_reg_count += count;
    }
    int total_count = non_normalized_count + total_reg_count;
    
    if (debug_) {
        std::cerr << "[normalizer] get_reg_from_entry: form='" << item->form << "' non_normalized=" << non_normalized_count 
                  << " total_reg=" << total_reg_count << " total=" << total_count << "\n";
        for (const auto& [reg, count] : reg_counts) {
            std::cerr << "[normalizer]   reg='" << reg << "' count=" << count << "\n";
        }
    }
    
    for (const auto& [reg, count] : reg_counts) {
        // Only use reg if it's more frequent than non-normalized form
        // or if non-normalized form is rare (< 20% of total)
        // CRITICAL: For a reg to be used, it must be the dominant form
        if (count > non_normalized_count || (total_count > 0 && non_normalized_count < total_count * 0.2)) {
            if (count > best_count) {
                best_reg = reg;
                best_count = count;
            }
        } else if (debug_) {
            std::cerr << "[normalizer]   reg='" << reg << "' (count=" << count 
                      << ") rejected: not more frequent than non-normalized (count=" << non_normalized_count << ")\n";
        }
    }
    
    if (debug_ && !best_reg.empty()) {
        std::cerr << "[normalizer] get_reg_from_entry: selected reg='" << best_reg << "' (count=" << best_count << ")\n";
    } else if (debug_ && best_reg.empty() && !reg_counts.empty()) {
        std::cerr << "[normalizer] get_reg_from_entry: no reg selected (all rejected)\n";
    }
    
    return best_reg;
}

std::string Normalizer::apply_morphological_variation(const std::string& word) const {
    if (!lexicon_) {
        return "";
    }
    
    // CRITICAL: Use Unicode-aware lowercase conversion
    std::string word_lower = to_lower(word);
    
    // Use inflection suffixes derived from vocab
    if (inflection_suffixes_.empty()) {
        return "";
    }
    
    // Try removing suffixes to find base form (longest first, since they're already sorted)
    for (const auto& suffix : inflection_suffixes_) {
        // Check if word_lower ends with suffix (C++17 compatible)
        if (word_lower.length() > suffix.length() + 2 && 
            word_lower.length() >= suffix.length() &&
            word_lower.substr(word_lower.length() - suffix.length()) == suffix) {
            std::string base_form = word_lower.substr(0, word_lower.length() - suffix.length());
            
            // Check if base form has a normalization mapping
            const LexiconItem* base_item = lexicon_->find(base_form);
            if (!base_item) {
                base_item = lexicon_->find_lower(base_form);
            }
            
            if (base_item) {
                std::string reg = get_reg_from_entry(base_item);
                if (!reg.empty() && reg != "_" && reg != base_form) {
                    // Apply same suffix to normalized form
                    std::string normalized = reg + suffix;
                    
                    // Verify the normalized form exists in lexicon
                    if (lexicon_->find(normalized) || lexicon_->find_lower(normalized)) {
                        return normalized;
                    }
                }
            }
            
            // Also try with original case
            if (word.length() > suffix.length()) {
                std::string base_form_orig = word.substr(0, word.length() - suffix.length());
                const LexiconItem* base_item_orig = lexicon_->find(base_form_orig);
                if (!base_item_orig) {
                    base_item_orig = lexicon_->find_lower(base_form_orig);
                }
                
                if (base_item_orig) {
                    std::string reg = get_reg_from_entry(base_item_orig);
                    if (!reg.empty() && reg != "_" && reg != base_form_orig) {
                        std::string normalized = reg + suffix;
                        if (lexicon_->find(normalized) || lexicon_->find_lower(normalized)) {
                            return normalized;
                        }
                    }
                }
            }
        }
    }
    
    return "";
}

std::string Normalizer::apply_pattern_substitution(const std::string& word) const {
    if (!lexicon_ || substitution_patterns_.empty()) {
        return "";
    }
    
    // CRITICAL: Use Unicode-aware lowercase conversion
    std::string word_lower = to_lower(word);
    
    // RESTRICTION 1: If the word exists in the vocab (as a non-normalized form), don't normalize it by patterns
    // Words that are already in the vocab should only be normalized if they have explicit reg values
    // Pattern substitution should only apply to OOV (Out-Of-Vocabulary) words
    const LexiconItem* word_item = lexicon_->find(word);
    if (!word_item) {
        word_item = lexicon_->find_lower(word);
    }
    if (word_item) {
        // Word exists in vocab - check if it has an explicit reg value
        // If it does, that should be used (handled by explicit mapping check in normalize())
        // If it doesn't, don't normalize it by patterns
        std::string reg = get_reg_from_entry(word_item);
        if (reg.empty() || reg == "_" || reg == word) {
            // Word is in vocab but has no normalization - don't normalize by patterns
            if (debug_) {
                std::cerr << "[normalizer] apply_pattern_substitution: word='" << word_lower << "' exists in vocab without normalization, skipping pattern substitution\n";
            }
            return "";
        }
    }
    
    // Try applying frequent substitution patterns
    // Sort patterns by frequency (most frequent first)
    std::vector<std::pair<std::string, int>> sorted_patterns;
    for (const auto& [pattern, count] : substitution_patterns_) {
        if (count >= min_pattern_count_) {
            sorted_patterns.push_back({pattern, count});
        }
    }
    std::sort(sorted_patterns.begin(), sorted_patterns.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Apply all applicable patterns to build the normalized form
    // We need to apply multiple patterns (e.g., ç->c and ó->o for coraçón->corazon)
    std::string normalized_candidate = word_lower;
    bool any_applied = false;
    
    if (debug_) {
        std::cerr << "[normalizer] apply_pattern_substitution: word='" << word_lower << "'\n";
    }
    
    for (const auto& [pattern_str, count] : sorted_patterns) {
        // Parse "from_to" pattern using Unicode-aware splitting
        // CRITICAL: Cannot use substr() because it's byte-based and can split multi-byte UTF-8 characters
        auto [from_char, to_char] = split_pattern(pattern_str);
        if (from_char.empty() || to_char.empty()) {
            if (debug_) {
                std::cerr << "[normalizer]   skipping invalid pattern: '" << pattern_str << "'\n";
            }
            continue;
        }
        
        if (debug_) {
            std::cerr << "[normalizer]   checking pattern: '" << pattern_str << "' (from='" << from_char << "' to='" << to_char << "' count=" << count << ")\n";
        }
        
        // Apply this pattern: replace from_char (variant) with to_char (standard)
        // CRITICAL: Only apply patterns that normalize variant->standard, not standard->variant
        // We can detect this by checking if from_char is more "complex" than to_char
        // For now, we'll use a simple heuristic: if to_char is ASCII and from_char is not, it's likely normalization
        // Also, if from_char has more bytes than to_char, it's likely a variant
        // Skip patterns that go in the wrong direction (e.g., 'o'->'ó' should not be applied)
        bool is_normalization = false;
        if (to_char.size() == 1 && from_char.size() > 1) {
            // to_char is single-byte ASCII, from_char is multi-byte -> likely normalization
            is_normalization = true;
        } else if (from_char.size() > to_char.size()) {
            // from_char is longer (more complex) -> likely normalization
            is_normalization = true;
        } else if (from_char.size() == to_char.size() && from_char.size() > 1) {
            // Both are multi-byte, check if from_char has diacritics and to_char doesn't
            // This is a heuristic: accented characters typically normalize to unaccented
            // We'll apply the pattern if it exists in the word
            is_normalization = true;
        }
        
        // Only apply if it's a normalization pattern (variant->standard)
        if (is_normalization && char_exists_in_string(normalized_candidate, from_char)) {
            std::string before_replace = normalized_candidate;  // Save state before replacement
            // Try applying this pattern to see if it leads to a valid normalized form
            std::string test_candidate = replace_char(normalized_candidate, from_char, to_char);
            if (test_candidate != normalized_candidate) {
                // Check if the result exists in the lexicon (as key, case-insensitive, or as reg value)
                // This ensures we only apply patterns that lead to valid normalized forms
                // CRITICAL: If the normalized form exists as a form that itself has a normalization,
                // we should use that form's normalization instead (e.g., "corazon" -> "corazón")
                const LexiconItem* test_item = lexicon_->find(test_candidate);
                if (!test_item) {
                    test_item = lexicon_->find_lower(test_candidate);
                }
                
                bool is_valid = (test_item != nullptr);
                std::string final_normalized = test_candidate;
                
                if (is_valid) {
                    // Check if test_candidate itself has a normalization - if so, use that instead
                    std::string test_reg = get_reg_from_entry(test_item);
                    if (!test_reg.empty() && test_reg != "_" && test_reg != "--" && test_reg != test_candidate) {
                        // Skip capitalization-only, punctuation, and "--" normalizations
                        if (!is_capitalization_only(test_candidate, test_reg) && 
                            !is_punctuation_normalization(test_candidate, test_reg) && 
                            test_reg != "--") {
                            final_normalized = test_reg;
                            if (debug_) {
                                std::cerr << "[normalizer]     normalized form '" << test_candidate << "' itself has normalization '" << test_reg << "', using that instead\n";
                            }
                        }
                    }
                } else {
                    // Check if test_candidate exists as a reg value (normalized form) in the lexicon
                    auto mappings = lexicon_->get_normalization_mappings();
                    for (const auto& [form, reg] : mappings) {
                        std::string reg_lower = to_lower(reg);
                        if (reg_lower == test_candidate) {
                            is_valid = true;
                            final_normalized = test_candidate;
                            break;
                        }
                    }
                }
                
                if (is_valid) {
                    // Only apply the pattern if it leads to a valid form in the lexicon
                    normalized_candidate = final_normalized;
                    any_applied = true;
                    if (debug_) {
                        std::cerr << "[normalizer]     applied: '" << before_replace << "' -> '" << normalized_candidate << "' (valid=" << is_valid << ")\n";
                    }
                } else if (debug_) {
                    std::cerr << "[normalizer]     skipped (doesn't lead to valid form): '" << from_char << "' -> '" << to_char << "' (test='" << test_candidate << "')\n";
                }
            }
        } else if (debug_ && char_exists_in_string(normalized_candidate, from_char)) {
            std::cerr << "[normalizer]     skipped (wrong direction): '" << from_char << "' -> '" << to_char << "'\n";
        }
    }
    
    // If we applied any patterns, verify the result
    if (any_applied && normalized_candidate != word_lower) {
        // RESTRICTION 2: The normalized form MUST exist in the vocab (as key or as reg value)
        // This prevents normalizing to invalid forms like "pzÉñzzzzÉ"
        // CRITICAL: If the normalized form exists as a form that itself has a normalization,
        // we should use that form's normalization instead (e.g., "corazon" -> "corazón")
        bool verified = false;
        std::string final_normalized = normalized_candidate;
        
        // Check if normalized form exists as a key (exact or case-insensitive)
        const LexiconItem* norm_item = lexicon_->find(normalized_candidate);
        if (!norm_item) {
            norm_item = lexicon_->find_lower(normalized_candidate);
        }
        
        if (norm_item) {
            verified = true;
            // Check if normalized_candidate itself has a normalization - if so, use that instead
            std::string norm_reg = get_reg_from_entry(norm_item);
            if (!norm_reg.empty() && norm_reg != "_" && norm_reg != "--" && norm_reg != normalized_candidate) {
                // Skip capitalization-only, punctuation, and "--" normalizations
                if (!is_capitalization_only(normalized_candidate, norm_reg) && 
                    !is_punctuation_normalization(normalized_candidate, norm_reg) && 
                    norm_reg != "--") {
                    final_normalized = norm_reg;
                    if (debug_) {
                        std::cerr << "[normalizer] apply_pattern_substitution: normalized form '" << normalized_candidate << "' itself has normalization '" << norm_reg << "', using that instead\n";
                    }
                }
            }
        } else {
            // Also try checking if the normalized form (with different case) exists
            std::string normalized_upper = to_upper(normalized_candidate);
            norm_item = lexicon_->find(normalized_upper);
            if (norm_item) {
                verified = true;
                std::string norm_reg = get_reg_from_entry(norm_item);
                if (!norm_reg.empty() && norm_reg != "_" && norm_reg != "--" && norm_reg != normalized_upper) {
                    if (!is_capitalization_only(normalized_upper, norm_reg) && 
                        !is_punctuation_normalization(normalized_upper, norm_reg) && 
                        norm_reg != "--") {
                        final_normalized = norm_reg;
                    }
                }
            }
        }
        
        // Also check if normalized form exists as a reg value (normalized form) in the lexicon
        if (!verified) {
            auto mappings = lexicon_->get_normalization_mappings();
            for (const auto& [form, reg] : mappings) {
                std::string reg_lower = to_lower(reg);
                if (reg_lower == normalized_candidate) {
                    verified = true;
                    final_normalized = normalized_candidate;
                    break;
                }
            }
        }
        
        // CRITICAL: Only return normalized form if it's verified to exist in the vocab
        // Do NOT use pattern frequency as a fallback - this leads to invalid normalizations
        if (verified) {
            if (debug_) {
                std::cerr << "[normalizer] apply_pattern_substitution: verified normalized form '" << final_normalized << "' exists in vocab\n";
            }
            return final_normalized;
        } else {
            if (debug_) {
                std::cerr << "[normalizer] apply_pattern_substitution: normalized form '" << normalized_candidate << "' does NOT exist in vocab, rejecting\n";
            }
        }
    }
    
    return "";
}

} // namespace flexitag

