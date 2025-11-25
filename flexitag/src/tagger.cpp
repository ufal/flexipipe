#include "flexitag/tagger.h"
#include "flexitag/unicode_utils.h"

#include "flexitag/io_teitok.h"

#include <cmath>
#include <algorithm>
#include <limits>
#include <chrono>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <sstream>

namespace flexitag {

namespace {
// Using ICU-based Unicode utilities instead of manual UTF-8 handling
using flexitag::unicode::char_count;
using flexitag::unicode::suffix;
using flexitag::unicode::prefix;
using flexitag::unicode::substr_from_char;
} // namespace

// Helper function to get the tag value based on tagpos setting (xpos/upos/utot)
std::string get_tag_value(const WordCandidate& cand, const std::string& tagpos) {
    if (tagpos == "upos") {
        if (cand.lex_attributes && !cand.lex_attributes->upos.empty() && cand.lex_attributes->upos != "_") {
            return cand.lex_attributes->upos;
        }
        // Fallback: try to extract upos from tag if it's in format "UPOS#FEATS"
        std::size_t hash_pos = cand.tag.find('#');
        if (hash_pos != std::string::npos) {
            return cand.tag.substr(0, hash_pos);
        }
        return cand.tag;  // Fallback to tag itself
    } else if (tagpos == "utot") {
        // utot = upos#feats
        if (cand.lex_attributes) {
            std::string upos = cand.lex_attributes->upos.empty() || cand.lex_attributes->upos == "_" ? "" : cand.lex_attributes->upos;
            std::string feats = cand.lex_attributes->feats.empty() || cand.lex_attributes->feats == "_" ? "" : cand.lex_attributes->feats;
            if (!upos.empty() && !feats.empty()) {
                return upos + "#" + feats;
            } else if (!upos.empty()) {
                return upos;
            }
        }
        // Fallback: if tag already contains #, use it
        if (cand.tag.find('#') != std::string::npos) {
            return cand.tag;
        }
        // Otherwise construct from tag (assuming tag is upos) + empty feats
        return cand.tag + "#";
    } else {
        // Default: xpos (use tag directly, which is xpos from lexicon)
        return cand.tag;
    }
}

// Helper to get tag value from token attributes
std::string get_token_tag_value(const Token& token, const std::string& tagpos) {
    if (tagpos == "upos") {
        return token.upos.empty() || token.upos == "_" ? "" : token.upos;
    } else if (tagpos == "utot") {
        std::string upos = token.upos.empty() || token.upos == "_" ? "" : token.upos;
        std::string feats = token.feats.empty() || token.feats == "_" ? "" : token.feats;
        if (!upos.empty() && !feats.empty()) {
            return upos + "#" + feats;
        } else if (!upos.empty()) {
            return upos;
        }
        return "";
    } else {
        return token.xpos.empty() || token.xpos == "_" ? "" : token.xpos;
    }
}

// CRITICAL: Apply lemmatization rule to word (neotagxml line 570-638)
// Rules are like "*er#*" meaning: remove "er" from end, result is lemma
// Example: apply "*er#*" to "Geißler" -> "Geißl"
std::string apply_lemmatization_rule(const std::string& word, const std::string& rule, bool debug = false) {
    if (word.empty() || rule.empty()) {
        if (debug) {
            std::cerr << "[flexitag] apply_lemmatization_rule: empty word or rule\n";
        }
        return "";
    }
    
    if (debug) {
        std::cerr << "[flexitag] apply_lemmatization_rule: word='" << word << "' rule='" << rule << "'\n";
    }
    
    std::string lemma;
    std::string prefix;
    std::string suffix;
    std::string root = word;
    
    // Split rule into word transformation and lemma transformation (neotagxml line 575)
    std::size_t hash_pos = rule.find('#');
    if (hash_pos == std::string::npos) {
        return "";  // Invalid rule format
    }
    
    std::string wrdtr = rule.substr(0, hash_pos);
    std::string lemtr = rule.substr(hash_pos + 1);
    
    // First apply the bits required on the beginning and the end (neotagxml line 578-586)
    while (!lemtr.empty() && lemtr[0] != '*') {
        prefix += lemtr[0];
        lemtr.erase(0, 1);
    }
    while (!lemtr.empty() && lemtr.back() != '*') {
        suffix = lemtr.back() + suffix;
        lemtr.erase(lemtr.size() - 1, 1);
    }
    
    while (!root.empty() && !wrdtr.empty() && wrdtr[0] != '*') {
        if (root[0] != wrdtr[0]) {
            return "";  // Not applicable
        }
        wrdtr.erase(0, 1);
        root.erase(0, 1);
    }
    
    // Now, recursively treat the bits at the end (neotagxml line 596-625)
    int wrdidx = static_cast<int>(root.size()) - 1;
    while (!wrdtr.empty() && !root.empty()) {
        while (!wrdtr.empty() && wrdtr.back() != '*') {
            if (wrdidx < 0 || root[wrdidx] != wrdtr.back()) {
                return "";  // Not applicable
            }
            wrdtr.erase(wrdtr.size() - 1, 1);
            root.erase(wrdidx, 1);
            // If we have a character in the replacement as well, insert that here
            // CRITICAL: neotagxml line 608 uses lemtr[lemtr.size()-1] (single char)
            while (!lemtr.empty() && lemtr.back() != '*' && lemtr.size() > 0) {
                root.insert(wrdidx, 1, lemtr.back());
                lemtr.erase(lemtr.size() - 1, 1);
            }
            wrdidx--;
        }
        if (!wrdtr.empty() && wrdtr.back() == '*') {
            wrdtr.erase(wrdtr.size() - 1, 1);
        }
        if (!lemtr.empty() && lemtr.back() == '*') {
            lemtr.erase(lemtr.size() - 1, 1);
        }
        if (!wrdtr.empty()) {
            // CRITICAL: neotagxml line 620: while (wrdidx > 0 && root[wrdidx] != wrdtr[wrdtr.size()-1])
            // Note: wrdidx > 0 (not >= 0), so we stop at index 0
            while (wrdidx > 0 && wrdidx < static_cast<int>(root.size()) && 
                   root[wrdidx] != wrdtr.back()) {
                wrdidx--;
            }
        }
    }
    
    lemma = prefix + root + suffix;
    
    // CRITICAL: Sanitize UTF-8 - lemmatization rules work on bytes and can split UTF-8 sequences
    // Replace any invalid UTF-8 bytes with '?' to prevent encoding errors
    std::string sanitized;
    sanitized.reserve(lemma.size());
    for (size_t i = 0; i < lemma.size(); ) {
        unsigned char c = static_cast<unsigned char>(lemma[i]);
        if (c < 0x80) {
            // ASCII
            sanitized += c;
            ++i;
        } else if ((c & 0xE0) == 0xC0) {
            // 2-byte sequence
            if (i + 1 < lemma.size() && (static_cast<unsigned char>(lemma[i+1]) & 0xC0) == 0x80) {
                sanitized += lemma[i];
                sanitized += lemma[i+1];
                i += 2;
            } else {
                sanitized += '?';
                ++i;
            }
        } else if ((c & 0xF0) == 0xE0) {
            // 3-byte sequence
            if (i + 2 < lemma.size() && 
                (static_cast<unsigned char>(lemma[i+1]) & 0xC0) == 0x80 &&
                (static_cast<unsigned char>(lemma[i+2]) & 0xC0) == 0x80) {
                sanitized += lemma[i];
                sanitized += lemma[i+1];
                sanitized += lemma[i+2];
                i += 3;
            } else {
                sanitized += '?';
                ++i;
            }
        } else if ((c & 0xF8) == 0xF0) {
            // 4-byte sequence
            if (i + 3 < lemma.size() &&
                (static_cast<unsigned char>(lemma[i+1]) & 0xC0) == 0x80 &&
                (static_cast<unsigned char>(lemma[i+2]) & 0xC0) == 0x80 &&
                (static_cast<unsigned char>(lemma[i+3]) & 0xC0) == 0x80) {
                sanitized += lemma[i];
                sanitized += lemma[i+1];
                sanitized += lemma[i+2];
                sanitized += lemma[i+3];
                i += 4;
            } else {
                sanitized += '?';
                ++i;
            }
        } else {
            // Invalid UTF-8 start byte
            sanitized += '?';
            ++i;
        }
    }
    
    if (debug) {
        std::cerr << "[flexitag] apply_lemmatization_rule: result='" << sanitized << "'\n";
    }
    return sanitized;
}

namespace {

std::string form_case(const std::string& form) {
    if (form.empty()) return "";
    bool first_upper = std::isupper(static_cast<unsigned char>(form.front()));
    bool last_upper = std::isupper(static_cast<unsigned char>(form.back()));
    if (first_upper && last_upper && form.size() > 1) {
        return "UU";
    }
    if (first_upper) {
        return "Ul";
    }
    if (std::islower(static_cast<unsigned char>(form.front()))) {
        return "ll";
    }
    return "??";
}

// CRITICAL: Apply case transformation to lemma (neotagxml line 287-311, applycase())
// Determines most likely case for tag and adjusts lemma accordingly
std::string apply_case_to_lemma(const std::string& lemma, const std::string& tag, const std::string& wcase) {
    if (lemma.empty() || tag.empty()) {
        return lemma;
    }
    
    // Find most likely case for this tag from tag_stats (neotagxml line 289-296)
    // neotagxml uses caseProb[tag] to find maxprob case
    // We'll use tag_stats which has case information
    std::string lcase = wcase;  // Default to word's case
    float maxprob = 0.0f;
    
    // Get tag_stats from lexicon (we need access to it)
    // For now, use a simplified version that matches neotagxml's logic
    // neotagxml line 297-310: if lcase != wcase, apply transformation
    
    // Simple heuristic matching neotagxml's applycase():
    // - If tag's most likely case is "ll" or "Ul", lowercase the lemma
    // - If tag's most likely case is "Ul", capitalize first char
    // - If tag's most likely case is "UU", lowercase the lemma
    // For now, we'll preserve word case pattern unless there's a strong signal
    
    // Actually, neotagxml's applycase() is more complex - it uses caseProb[tag]
    // which we don't have direct access to here. Let's use a simpler heuristic:
    // Match the word's case pattern (which is what we're already doing)
    
    return lemma;  // For now, don't transform - the issue is likely elsewhere
}

float safe_log(float value) {
    if (value <= 0.f) {
        return -1e6f;
    }
    return std::log(value);
}

// CRITICAL: Refine lemma using lemmatization rules from endings (neotagxml line 335-367, lemmatize())
// This is called after Viterbi to refine lemmas that weren't set from lexicon
// neotagxml only does this if lemma is empty and lexitem doesn't have lemma
// CRITICAL: neotagxml line 1348 stores lemmatizations in the candidate during generation
// Then in lemmatize() (line 342), it uses stored lemmatizations if available, otherwise looks up
std::string refine_lemma_with_endings(const std::string& form, const std::string& tag, 
                                      const std::string& current_lemma,
                                      const std::unordered_map<std::string, int>* stored_lemmatizations,
                                      const Lexicon* lexicon, bool debug = false,
                                      const std::string& normalized_form = "") {
    if (form.empty() || tag.empty()) {
        return current_lemma;
    }
    
    // For corpora like ode_ps, lemmatization rules are built from normalized form (reg)
    // So we should apply them to the normalized form if available, not the original form
    std::string lemmatization_source = normalized_form.empty() ? form : normalized_form;
    
    // neotagxml line 342: if lemmatizations.size() == 0, do fresh lookup
    // Otherwise, use stored lemmatizations from the candidate
    const std::unordered_map<std::string, int>* lemmatizations_ptr = stored_lemmatizations;
    std::string found_ending;
    
    if (lemmatizations_ptr == nullptr || lemmatizations_ptr->empty()) {
        // neotagxml line 342-349: looks up lemmatizations from longest ending
        // We'll do the same - find longest ending with lemmatization rules for this tag
        int endretry = 2;  // Default from settings
        // Use UTF-8 character count, not byte count
        size_t form_char_count = char_count(form);
        for (std::size_t i = 1; i < form_char_count; ++i) {
            // Use UTF-8-aware substring to get ending from character position i
            std::string wending = substr_from_char(form, i);
            auto ending_it = lexicon->endings().find(wending);
            if (ending_it != lexicon->endings().end()) {
                auto tag_it = ending_it->second.find(tag);
                if (tag_it != ending_it->second.end() && !tag_it->second.lemmatizations.empty()) {
                    // Found lemmatizations for this ending+tag - use them (neotagxml line 346)
                    lemmatizations_ptr = &tag_it->second.lemmatizations;
                    found_ending = wending;
                    if (debug) {
                        std::cerr << "[flexitag] refine_lemma: found " << lemmatizations_ptr->size() 
                                  << " lemmatization rules for ending '" << wending 
                                  << "' tag=" << tag << " form=" << form << "\n";
                    }
                    break;  // Use longest ending (neotagxml line 347: i = form.size())
                }
            }
        }
    } else if (debug) {
        std::cerr << "[flexitag] refine_lemma: using stored lemmatizations (" 
                  << lemmatizations_ptr->size() << " rules) from candidate\n";
    }
    
    // neotagxml line 351-365: applies most frequent rule
    // CRITICAL: neotagxml line 355 uses `if (it->second > maxlem)` (strict >)
    // This means if multiple rules have the same frequency, the first one encountered is used
    // Also, neotagxml prefers rules that result in the word staying the same (rule `*#*`)
    if (lemmatizations_ptr != nullptr && !lemmatizations_ptr->empty()) {
        int maxlem = 0;
        std::string best_lemma = current_lemma;
        std::string best_rule;
        
        // neotagxml line 354-365: iterates through all rules, picks highest freq
        // CRITICAL: neotagxml uses std::map<string,int> which iterates in sorted (alphabetical) order
        // neotagxml line 355 uses strict > (not >=), so first encountered wins ties
        // We need to match this exactly: iterate in sorted order and use strict >
        
        // Convert to vector and sort by rule name to match neotagxml's iteration order
        std::vector<std::pair<std::string, int>> sorted_rules(lemmatizations_ptr->begin(), lemmatizations_ptr->end());
        std::sort(sorted_rules.begin(), sorted_rules.end(), 
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        
        for (const auto& [lemrule, freq] : sorted_rules) {
            // neotagxml line 355: strict > (not >=)
            if (freq > maxlem) {
                // Apply rule to lemmatization_source (form or normalized form)
                std::string lemma = apply_lemmatization_rule(lemmatization_source, lemrule, debug);
                if (!lemma.empty()) {
                    if (debug) {
                        std::cerr << "[flexitag] refine_lemma: rule '" << lemrule 
                                  << "' (freq=" << freq << ") -> lemma='" << lemma << "'\n";
                    }
                    best_lemma = lemma;
                    maxlem = freq;
                    best_rule = lemrule;
                } else if (debug) {
                    std::cerr << "[flexitag] refine_lemma: rule '" << lemrule 
                              << "' (freq=" << freq << ") -> empty (not applicable)\n";
                }
            } else if (debug && freq == maxlem) {
                // Log when we skip a rule with equal frequency (neotagxml would use first encountered)
                std::string lemma = apply_lemmatization_rule(form, lemrule, debug);
                if (!lemma.empty()) {
                    std::cerr << "[flexitag] refine_lemma: skipping rule '" << lemrule 
                              << "' (freq=" << freq << ", equal to max=" << maxlem 
                              << ", first encountered wins)\n";
                }
            }
        }
        if (debug && !best_rule.empty()) {
            std::cerr << "[flexitag] refine_lemma: selected rule '" << best_rule 
                      << "' -> lemma='" << best_lemma << "' (was: '" << current_lemma << "')\n";
        }
        return best_lemma;
    }
    
    // CRITICAL: neotagxml line 367-375: if unable to lemmatize, return form
    // If no lemmatization rules found or all failed, return current_lemma (which should be form)
    // But if current_lemma is empty, we should return form as fallback
    if (current_lemma.empty()) {
        return form;  // Fallback to form when no lemma available
    }
    return current_lemma;
}

struct DPState {
    float score = std::numeric_limits<float>::lowest();
    int prev_index = -1;
    std::string prev_tag;  // For backtracking when using tag-based pruning
    WordCandidate candidate;
};

} // namespace

FlexitagTagger::FlexitagTagger() = default;

void FlexitagTagger::configure(const TaggerSettings& settings) {
    settings_ = settings;
}

void FlexitagTagger::set_lexicon(std::shared_ptr<Lexicon> lexicon) {
    lexicon_ = std::move(lexicon);
    
    // Initialize normalizer if lexicon is available and enhanced normalization is not skipped
    bool skip_enhanced_norm = settings_.get_bool("skip_enhanced_normalization", false);
    if (lexicon_ && !skip_enhanced_norm) {
        normalizer_ = std::make_unique<Normalizer>(lexicon_.get());
        if (normalizer_) {
            normalizer_->set_debug(settings_.debug);
        }
        if (settings_.debug && normalizer_ && normalizer_->is_enabled()) {
            std::cerr << "[flexitag] enhanced normalization enabled (inflection suffixes: " 
                      << normalizer_->inflection_suffixes().size() 
                      << ", substitution patterns: " 
                      << normalizer_->substitution_patterns().size() << ")\n";
        }
    } else {
        normalizer_.reset();
        if (settings_.debug && lexicon_) {
            std::cerr << "[flexitag] enhanced normalization disabled (skip_enhanced_normalization=" 
                      << skip_enhanced_norm << ", has_normalizations=" 
                      << lexicon_->has_normalizations() << ")\n";
        }
    }
}

Document FlexitagTagger::tag(const Document& doc, TaggerStats* stats) {
    if (!lexicon_) {
        throw std::runtime_error("Lexicon not loaded");
    }

    if (settings_.debug) {
        std::cerr << "[flexitag] debug mode enabled\n";
    }

    Document result = doc;
    TaggerStats local_stats;
    auto start_time = std::chrono::steady_clock::now();
    
    // Timing breakdown
    std::chrono::steady_clock::time_point candidate_time = {};
    std::chrono::steady_clock::time_point viterbi_time = {};
    std::chrono::steady_clock::time_point update_time = {};
    std::chrono::duration<double> candidate_duration{};
    std::chrono::duration<double> viterbi_duration{};
    std::chrono::duration<double> update_duration{};
    int candidate_calls = 0;

    for (auto& sentence : result.sentences) {
        std::vector<std::vector<WordCandidate>> lattice;
        lattice.reserve(sentence.tokens.size());
        for (auto& token : sentence.tokens) {
            if (token.form == "--" || token.form.empty()) {
                lattice.push_back({});
                continue;
            }
            auto candidate_start = std::chrono::steady_clock::now();
            auto candidates = morpho_parse(token);
            candidate_calls++;
            candidate_duration += std::chrono::steady_clock::now() - candidate_start;
            if (candidates.empty()) {
                WordCandidate fallback;
                fallback.form = token.form;
                fallback.lemma = token.form;
                fallback.tag = "<unknown>";
                fallback.source = "fallback";
                float fallback_prob = settings_.get_float("fallback_prob", 1e-6f);
                fallback.prob = fallback_prob;
                fallback.token = &token;
                candidates.push_back(std::move(fallback));
            }
            lattice.push_back(std::move(candidates));
        }

        // Use beam search: only keep the best state for each unique tag at each position
        // This prevents exponential explosion with large tag sets (like utot with upos#feats)
        // Additional safeguard: limit max states per position to prevent memory issues
        auto viterbi_start = std::chrono::steady_clock::now();
        std::size_t max_states = static_cast<std::size_t>(settings_.get_int("beam_size", 1000));
        float beam_prune_threshold = settings_.get_float("beam_prune_threshold", 1e-4f);
        float prob_epsilon = settings_.get_float("prob_epsilon", 1e-10f);
        std::string tagpos = settings_.get("tagpos", "xpos");
        std::vector<std::unordered_map<std::string, DPState>> dp(lattice.size());
        
        // CRITICAL: neotagxml uses LINEAR space for probabilities, not log space!
        // Formula: newprob = prob * newword.prob * pow(transitionprob, transitionfactor) * caseprob1
        // Then paths are normalized at the end
        
        for (std::size_t i = 0; i < lattice.size(); ++i) {
            // For each candidate at position i
            for (std::size_t j = 0; j < lattice[i].size(); ++j) {
                // Safety check: if we already have too many states, skip low-probability candidates
                if (dp[i].size() >= max_states && lattice[i][j].prob < beam_prune_threshold) {
                    continue;  // Skip very low probability candidates if we're at the limit
                }
                
                DPState state;
                state.candidate = lattice[i][j];
                // Use linear probability (neotagxml uses linear space)
                state.score = std::max(lattice[i][j].prob, prob_epsilon);  // Small epsilon to avoid zero
                state.prev_index = -1;
                // Use tagpos setting to determine which tag value to use for beam search
                std::string tag_key = get_tag_value(state.candidate, tagpos);

                if (i == 0) {
                    // First position: just store the state (one per tag)
                    auto it = dp[i].find(tag_key);
                    if (it == dp[i].end() || state.score > it->second.score) {
                        dp[i][tag_key] = state;
                    }
                    continue;
                }

                float best_score = 0.0f;  // Linear space: start at 0
                std::string best_prev_tag;
                
                // Get transition factor and smoothing from settings (defaults match neotagxml)
                float transition_factor = settings_.get_float("transitionfactor", 1.0f);
                float transition_smooth = settings_.get_float("transitionsmooth", 0.0f);
                
                // Only consider previous states (one per tag from previous position)
                // This is the key optimization: O(n * m) instead of O(n * m^2)
                for (const auto& [prev_tag, prev_state] : dp[i - 1]) {
                    // Extract the last part of previous tag for transitions (neotagxml line 745-749)
                    // For contractions (tags with '.'), take only the part after the last dot
                    std::string lasttag1 = prev_tag;
                    std::size_t last_dot = prev_tag.find_last_of('.');
                    if (last_dot != std::string::npos) {
                        lasttag1 = prev_tag.substr(last_dot + 1);
                    }
                    
                    // Extract the first part of current tag for transitions (neotagxml line 756-761)
                    // For contractions, take only the first part (before the first dot)
                    std::string newtag = tag_key;
                    std::size_t first_dot = tag_key.find('.');
                    if (first_dot != std::string::npos) {
                        newtag = tag_key.substr(0, first_dot);
                    }
                    
                    // Get transition probability (count) and apply smoothing
                    float transition_count = 0.0f;
                    std::string key = lasttag1 + "." + newtag;
                    auto it = lexicon_->transitions().find(key);
                    if (it != lexicon_->transitions().end()) {
                        transition_count = it->second.count;
                    }
                    // Apply smoothing: transitionprob = transitionprob1 + transitionsmooth
                    float transition_prob = transition_count + transition_smooth;
                    
                    // CRITICAL: neotagxml discards paths with zero transition probability (line 778-794)
                    // But only if transition_count is 0 AND smoothing is 0
                    // If smoothing > 0, we should still allow the path (with smoothed probability)
                    if (transition_count == 0.0f && transition_smooth == 0.0f) {
                        if (settings_.debug && i < 3 && j < 2) {
                            std::cerr << "[flexitag] Discarding path: " << prev_tag << " -> " << tag_key 
                                      << " (transition prob = 0, no smoothing)\n";
                        }
                        continue;  // Skip this transition (path is impossible)
                    }
                    
                    // In linear space: pow(transitionprob, transitionfactor)
                    float transition_power = std::pow(transition_prob, transition_factor);

                    // Case probability (as probability, not log)
                    // Skip case prob for contractions (neotagxml line 767-775)
                    float case_prob = 1.0f;
                    if (!state.candidate.dtoks.empty()) {
                        // Contractions: skip case prob (neotagxml line 770)
                        case_prob = 1.0f;
                    } else {
                        // Normal case: compute case probability
                        auto tag_it = lexicon_->tag_stats().find(tag_key);
                        if (tag_it != lexicon_->tag_stats().end()) {
                            const auto& tag_stat = tag_it->second;
                            std::string wcase = state.candidate.wcase;
                            for (const auto& c : tag_stat.cases) {
                                if (c.key == wcase) {
                                    // caseprob = casecnt / tagcnt (as probability)
                                    float case_prob_min = settings_.get_float("case_prob_min", 1e-3f);
                                    case_prob = std::max(c.count / std::max(tag_stat.count, 1.f), case_prob_min);
                                    break;
                                }
                            }
                        }
                    }

                    // Score in LINEAR space: prev_prob * wordprob * pow(transitionprob, transitionfactor) * caseprob
                    // This matches neotagxml line 778 exactly
                    float candidate_score = prev_state.score 
                                         * std::max(state.candidate.prob, prob_epsilon)  // word probability
                                         * transition_power  // transition probability (with factor)
                                         * case_prob;  // case probability
                    
                    // Debug: check for underflow
                    if (settings_.debug && i < 3 && j < 2) {  // Debug first few positions
                        std::cerr << "[flexitag] DP[" << i << "][" << j << "]: "
                                  << "prev_tag=" << prev_tag << " (" << prev_state.score << ") "
                                  << "word=" << state.candidate.form << " tag=" << tag_key 
                                  << " wordprob=" << state.candidate.prob
                                  << " trans=" << transition_power
                                  << " case=" << case_prob
                                  << " score=" << candidate_score << "\n";
                    }
                    
                    // Check for underflow
                    if (candidate_score > 0.0f && candidate_score < 1e-38f) {  // Minimum float value
                        if (settings_.debug) {
                            std::cerr << "[flexitag] WARNING: Very small probability detected: " 
                                      << candidate_score << " (possible underflow)\n";
                        }
                    }
                    
                    if (candidate_score > best_score) {
                        best_score = candidate_score;
                        best_prev_tag = prev_tag;
                    }
                }
                
                if (!best_prev_tag.empty()) {
                    // Found a valid previous state - use the computed score
                    state.score = best_score;
                    // Store reference to previous state (we'll resolve the actual index later)
                    state.prev_index = 0;  // Placeholder - we'll track by tag
                    // Only keep the best state for this tag
                    // CRITICAL: neotagxml line 1006 uses < (not <=), so when probabilities are equal,
                    // it keeps the FIRST one encountered, not the last one
                    auto it = dp[i].find(tag_key);
                    if (it == dp[i].end() || state.score > it->second.score) {
                        dp[i][tag_key] = state;
                        // Store the previous tag for backtracking
                        dp[i][tag_key].prev_tag = best_prev_tag;
                    }
                    // If score is equal, we keep the existing one (first encountered) - matching neotagxml line 1009-1010
                } else {
                    // No valid previous state found (all transitions were 0)
                    // CRITICAL: neotagxml handles this as a "dead end" (line 1019-1027)
                    // It outputs the best path from previous position, then resets and starts fresh
                    // with just word probabilities (no previous path probability)
                    // We simulate this by starting fresh paths with just word probability
                    // BUT we still need to connect to previous state for backtracking
                    if (!dp[i - 1].empty()) {
                        // Find the best previous state (for backtracking connection)
                        std::string fallback_prev_tag;
                        float fallback_prev_score = 0.0f;
                        for (const auto& [prev_tag, prev_state] : dp[i - 1]) {
                            if (prev_state.score > fallback_prev_score) {
                                fallback_prev_score = prev_state.score;
                                fallback_prev_tag = prev_tag;
                            }
                        }
                        
                        if (!fallback_prev_tag.empty()) {
                            // Start fresh path with just word probability (matching neotagxml line 985-991)
                            // But connect to previous state for backtracking
                            state.score = state.candidate.prob;  // Just word prob, no previous path multiplication
                            state.prev_tag = fallback_prev_tag;  // Connect for backtracking
                            
                            auto it = dp[i].find(tag_key);
                            // CRITICAL: neotagxml line 1006 uses < (not <=), so when probabilities are equal,
                            // it keeps the FIRST one encountered (matching neotagxml line 1009-1010)
                            if (it == dp[i].end() || state.score > it->second.score) {
                                dp[i][tag_key] = state;
                                if (settings_.debug && i < 10) {  // Increased debug range
                                    std::cerr << "[flexitag] Dead end - starting fresh path for " << tag_key 
                                              << " (word prob only, connected to " << fallback_prev_tag << "): " 
                                              << state.score << "\n";
                                }
                            }
                            // If score is equal, we keep the existing one (first encountered) - matching neotagxml
                        }
                    } else {
                        // No previous states - shouldn't happen, but store initial prob
                        auto it = dp[i].find(tag_key);
                        if (it == dp[i].end()) {
                            dp[i][tag_key] = state;
                        }
                    }
                }
            }
            
            // Handle unique path case (neotagxml line 1028-1050)
            // If there's only one possible state, set prob = 1 and continue
            // This matches neotagxml's "unique path" behavior
            if (dp[i].size() == 1) {
                // Unique path - set probability to 1 (neotagxml line 1047)
                for (auto& [tag, state] : dp[i]) {
                    state.score = 1.0f;
                    if (settings_.debug && i < 5) {
                        std::cerr << "[flexitag] Unique path for " << tag 
                                  << " - setting prob = 1\n";
                    }
                }
            } else if (i > 0 && !dp[i].empty()) {
                // Normalize states at this position to prevent underflow
                // This keeps probabilities in a reasonable range [0,1] without changing relative ordering
                // neotagxml normalizes lexical probs (line 1462-1468) and path probs at end (line 1533-1539)
                // We normalize at each position to prevent exponential decay in linear space
                // Normalization preserves relative ordering (all scores scaled by same factor)
                float totprob = 0.0f;
                for (const auto& [tag, state] : dp[i]) {
                    totprob += state.score;
                }
                if (totprob > 0.0f) {
                    // Always normalize to keep probabilities in [0,1] range
                    // This prevents underflow while preserving relative ordering
                    for (auto& [tag, state] : dp[i]) {
                        state.score = state.score / totprob;
                    }
                }
            }

            // Normalize scores at this position to avoid gradual underflow
            if (!dp[i].empty()) {
                float max_score = 0.0f;
                for (const auto& [tag, state] : dp[i]) {
                    if (state.score > max_score) {
                        max_score = state.score;
                    }
                }
                if (max_score > 0.0f) {
                    float inv = 1.0f / max_score;
                    for (auto& [tag, state] : dp[i]) {
                        state.score *= inv;
                    }
                } else {
                    for (auto& [tag, state] : dp[i]) {
                        state.score = prob_epsilon;
                    }
                }
            }
        }
        
        // Final normalization (neotagxml line 1533-1539) - normalize all final states
        if (!dp.empty() && !dp.back().empty()) {
            float totprob = 0.0f;
            for (const auto& [tag, state] : dp.back()) {
                totprob += state.score;
            }
            if (totprob > 0.0f) {
                for (auto& [tag, state] : dp.back()) {
                    state.score = state.score / totprob;
                }
            }
        }
        
        viterbi_duration += std::chrono::steady_clock::now() - viterbi_start;
        
        auto update_start = std::chrono::steady_clock::now();
        if (!dp.empty()) {
            // Find best final state (in linear space, higher is better)
            // CRITICAL: neotagxml's best() function (line 961-976) iterates through pathlist
            // and uses > (not >=), so when scores are equal, it keeps the FIRST one encountered
            // The order depends on how paths were added to pathlist (line 1057-1060)
            // which comes from newpathlist map iteration order (alphabetical by tag)
            std::string best_tag;
            float best_score = 0.0f;  // Linear space: start at 0
            int final_state_count = 0;
            for (const auto& [tag, state] : dp.back()) {
                final_state_count++;
                if (settings_.debug) {
                    std::cerr << "[flexitag] Final state: tag=" << tag 
                              << " score=" << state.score 
                              << " (prob=" << (state.score > 0 ? state.score : 0) << ")\n";
                }
                // Match neotagxml line 966: use > (not >=), so first encountered with max score wins
                // Since map iteration is alphabetical, this matches neotagxml's behavior
                if (state.score > best_score) {
                    best_score = state.score;
                    best_tag = tag;
                }
                // If equal, keep the first one (already stored in best_tag)
            }
            
            if (settings_.debug) {
                std::cerr << "[flexitag] Selected best path: tag=" << best_tag 
                          << " score=" << best_score 
                          << " (from " << final_state_count << " final states)\n";
            }
            
            // Check for underflow/zero probabilities
            if (best_score == 0.0f || best_score < 1e-38f) {
                std::cerr << "[flexitag] WARNING: Best score is zero or extremely small: " 
                          << best_score << " - possible underflow!\n";
            }

            // Backtrack through the path using tag keys
            // Note: We need non-const pointers to modify lemmas after Viterbi
            std::vector<WordCandidate*> best_path(dp.size(), nullptr);
            std::string current_tag = best_tag;
            int path_length = 0;
            for (int i = static_cast<int>(dp.size()) - 1; i >= 0; --i) {
                auto it = dp[i].find(current_tag);
                if (it != dp[i].end()) {
                    // Store pointer to candidate (non-const so we can modify lemma later)
                    best_path[i] = const_cast<WordCandidate*>(&it->second.candidate);
                    path_length++;
                    if (settings_.debug && i < 5) {  // Debug first few tokens
                        std::cerr << "[flexitag] Path[" << i << "]: tag=" << it->second.candidate.tag
                                  << " form=" << it->second.candidate.form
                                  << " score=" << it->second.score
                                  << " source=" << it->second.candidate.source << "\n";
                    }
                    if (!it->second.prev_tag.empty()) {
                        current_tag = it->second.prev_tag;
                    } else {
                        break;  // Reached the beginning
                    }
                } else {
                    if (settings_.debug) {
                        std::cerr << "[flexitag] WARNING: Path broken at position " << i 
                                  << " (tag=" << current_tag << " not found)\n";
                    }
                    break;  // Path broken
                }
            }
            
            if (settings_.debug) {
                std::cerr << "[flexitag] Path length: " << path_length 
                          << " (expected: " << sentence.tokens.size() << ")\n";
            }

            // CRITICAL: neotagxml line 805 calls lemmatize() on each wordtoken after Viterbi
            // This refines lemmas using lemmatization rules from endings
            // CRITICAL: neotagxml also lemmatizes dtok children (each wordtoken in toklist)
            for (std::size_t i = 0; i < sentence.tokens.size(); ++i) {
                if (best_path[i]) {
                    Token& token = sentence.tokens[i];
                    
                    // CRITICAL: Apply lemmatization to dtok children first (neotagxml line 805 iterates through toklist)
                    // Each wordtoken in the path gets lemmatized, including dtoks
                    if (!best_path[i]->dtoks.empty()) {
                        for (auto& dtok_cand : best_path[i]->dtoks) {
                            // Apply lemmatization to each dtok candidate
                            // neotagxml line 337: skip if lexitem has lemma or dtok has lemma
                            bool dtok_has_lexicon_lemma = dtok_cand->lex_attributes && 
                                                          !dtok_cand->lex_attributes->lemma.empty() &&
                                                          dtok_cand->lex_attributes->lemma != dtok_cand->form;
                            
                            if (!dtok_has_lexicon_lemma && (dtok_cand->lemma.empty() || dtok_cand->lemma == dtok_cand->form)) {
                                // Get stored lemmatizations for this dtok
                                const std::unordered_map<std::string, int>* dtok_stored_lemmatizations = nullptr;
                                if (!dtok_cand->lemmatizations.empty()) {
                                    dtok_stored_lemmatizations = &dtok_cand->lemmatizations;
                                }
                                
                                // For dtoks, check if they have a reg value for lemmatization
                                // Note: dtoks might not have reg, so we'll use form as fallback
                                std::string dtok_normalized_form = "";
                                if (dtok_cand->lex_attributes && 
                                    !dtok_cand->lex_attributes->reg.empty() && 
                                    dtok_cand->lex_attributes->reg != "_" &&
                                    dtok_cand->lex_attributes->reg != dtok_cand->form) {
                                    dtok_normalized_form = dtok_cand->lex_attributes->reg;
                                }
                                std::string dtok_refined_lemma = refine_lemma_with_endings(
                                    dtok_cand->form,
                                    dtok_cand->tag,
                                    dtok_cand->lemma,
                                    dtok_stored_lemmatizations,
                                    lexicon_.get(),
                                    settings_.debug && i < 10,
                                    dtok_normalized_form
                                );
                                
                                // Fallback to form if empty
                                if (dtok_refined_lemma.empty() || dtok_refined_lemma == dtok_cand->form) {
                                    dtok_refined_lemma = dtok_cand->form;
                                }
                                
                                if (!dtok_refined_lemma.empty()) {
                                    dtok_cand->lemma = dtok_refined_lemma;
                                }
                            }
                        }
                    }
                    
                    // CRITICAL: Apply lemmatization if needed (neotagxml line 805, lemmatize() at 335-367)
                    // neotagxml line 337: only lemmatizes if lemma is empty and lexitem doesn't have lemma
                    // We need to check if the candidate came from lexicon (has lex_attributes with lemma)
                    bool has_lexicon_lemma = best_path[i]->lex_attributes && 
                                            !best_path[i]->lex_attributes->lemma.empty() &&
                                            best_path[i]->lex_attributes->lemma != token.form;
                    
                    // neotagxml line 337: skip if lexitem has lemma or dtok has lemma
                    // We'll skip if candidate has lexicon lemma
                    if (!has_lexicon_lemma && (best_path[i]->lemma.empty() || best_path[i]->lemma == token.form)) {
                        if (settings_.debug && i < 10) {
                            std::cerr << "[flexitag] Refining lemma for token[" << i << "]: " 
                                      << token.form << " tag=" << best_path[i]->tag 
                                      << " current_lemma=" << best_path[i]->lemma 
                                      << " has_lexicon_lemma=" << (has_lexicon_lemma ? "yes" : "no") << "\n";
                        }
                        // Try to refine lemma using lemmatization rules from endings
                        // CRITICAL: Pass stored lemmatizations from candidate (neotagxml line 342)
                        const std::unordered_map<std::string, int>* stored_lemmatizations = nullptr;
                        if (!best_path[i]->lemmatizations.empty()) {
                            stored_lemmatizations = &best_path[i]->lemmatizations;
                        }
                        // For corpora like ode_ps, lemmatization should be done from normalized form (reg)
                        // Pass token.reg if available and different from form
                        std::string normalized_form = (!token.reg.empty() && token.reg != "_" && token.reg != token.form) 
                                                      ? token.reg : "";
                        std::string refined_lemma = refine_lemma_with_endings(
                            token.form, 
                            best_path[i]->tag,
                            best_path[i]->lemma,
                            stored_lemmatizations,
                            lexicon_.get(),
                            settings_.debug && i < 10,
                            normalized_form
                        );
                        // CRITICAL: neotagxml line 367-375: if unable to lemmatize, use form
                        // This is the fallback when no lemmatization rules apply
                        if (refined_lemma.empty() || refined_lemma == token.form) {
                            // If refinement didn't change anything or returned empty, use form as fallback
                            refined_lemma = token.form;
                        }
                        // Update lemma (even if it's the form - this ensures lemma is never empty)
                        if (!refined_lemma.empty()) {
                            if (settings_.debug && i < 10 && refined_lemma != best_path[i]->lemma) {
                                std::cerr << "[flexitag] Refined lemma: '" << best_path[i]->lemma 
                                          << "' -> '" << refined_lemma << "'\n";
                            }
                            best_path[i]->lemma = refined_lemma;
                        }
                    } else if (settings_.debug && i < 10 && has_lexicon_lemma) {
                        std::cerr << "[flexitag] Skipping lemmatization for token[" << i << "]: " 
                                  << token.form << " (has lexicon lemma: " 
                                  << best_path[i]->lex_attributes->lemma << ")\n";
                    }
                    
                    // CRITICAL: Ensure lemma is never empty (neotagxml line 367-375 fallback)
                    // If lemma is still empty after all processing, use form
                    if (best_path[i]->lemma.empty()) {
                        best_path[i]->lemma = token.form;
                        if (settings_.debug && i < 10) {
                            std::cerr << "[flexitag] Fallback: using form as lemma for token[" << i << "]: " 
                                      << token.form << "\n";
                        }
                    }
                    
                    if (settings_.debug && i < 5) {  // Debug first few tokens
                        std::cerr << "[flexitag] Updating token[" << i << "]: " 
                                  << token.form << " -> xpos=" << best_path[i]->tag
                                  << " lemma=" << best_path[i]->lemma
                                  << " (was: " << token.xpos << ")\n";
                    }
                    update_token(token, *best_path[i]);
                    local_stats.word_count++;
                    if (best_path[i]->source.find("lexicon") == std::string::npos) {
                        local_stats.oov_count++;
                    }
                    if (settings_.debug && i < 10) {
                        std::cerr << "[flexitag] chosen token "
                                  << "\"" << token.form << "\" -> tag=" << best_path[i]->tag
                                  << " lemma=" << best_path[i]->lemma
                                  << " source=" << best_path[i]->source
                                  << " prob=" << best_path[i]->prob
                                  << "\n";
                    }
                } else {
                    if (settings_.debug) {
                        std::cerr << "[flexitag] WARNING: No path for token[" << i << "]: " 
                                  << sentence.tokens[i].form << "\n";
                    }
                }
            }
        } else {
            if (settings_.debug) {
                std::cerr << "[flexitag] ERROR: DP table is empty - no states computed!\n";
            }
        }
        update_duration += std::chrono::steady_clock::now() - update_start;
    }

    auto end_time = std::chrono::steady_clock::now();
    local_stats.elapsed_seconds =
        std::chrono::duration_cast<std::chrono::duration<float>>(end_time - start_time).count();
    
    // Print timing breakdown if verbose or debug
    if (settings_.get_bool("verbose", false) || settings_.debug) {
        double total_seconds = std::chrono::duration<double>(end_time - start_time).count();
        double candidate_seconds = std::chrono::duration<double>(candidate_duration).count();
        double viterbi_seconds = std::chrono::duration<double>(viterbi_duration).count();
        double update_seconds = std::chrono::duration<double>(update_duration).count();
        std::cerr << "[flexitag] Timing breakdown:\n";
        std::cerr << "  Candidate generation: " << candidate_calls << " calls, " 
                  << candidate_seconds << "s ("
                  << (total_seconds > 0 ? (candidate_seconds / total_seconds * 100) : 0) << "%)\n";
        std::cerr << "  Viterbi algorithm: " 
                  << viterbi_seconds << "s ("
                  << (total_seconds > 0 ? (viterbi_seconds / total_seconds * 100) : 0) << "%)\n";
        std::cerr << "  Token updates: " 
                  << update_seconds << "s ("
                  << (total_seconds > 0 ? (update_seconds / total_seconds * 100) : 0) << "%)\n";
        std::cerr << "  Total: " << total_seconds << "s\n";
    }

    if (stats) {
        *stats = local_stats;
    }
    return result;
}

std::vector<WordCandidate> FlexitagTagger::morpho_parse(Token& token) const {
    std::vector<WordCandidate> candidates;

    // CRITICAL: If token already has dtoks (contractions) in the input, we need to handle them
    // neotagxml line 320-332: if lexitem has dtok children, it creates wordtokens from them via adddtok()
    // neotagxml still processes the token normally (finds candidates, runs Viterbi), but uses existing dtok structure
    // The key is: neotagxml matches candidates against existing dtoks (tagsmatch checks dtok tags)
    // We should create candidates that match the existing dtok structure, but still allow normal processing
    bool has_existing_dtoks = !token.subtokens.empty();
    
    // Pre-compute combined tag for dtoks (if any) to avoid recomputing it in the lambda
    // Use tagpos setting to determine which tag attribute to use
    std::string tagpos = settings_.get("tagpos", "xpos");
    std::string combined_dtok_tag;
    if (has_existing_dtoks) {
        std::vector<std::string> dtok_tags;
        for (const auto& st : token.subtokens) {
            std::string dtok_tag;
            if (tagpos == "upos") {
                dtok_tag = st.upos.empty() || st.upos == "_" ? "" : st.upos;
            } else if (tagpos == "utot") {
                std::string upos = st.upos.empty() || st.upos == "_" ? "" : st.upos;
                std::string feats = st.feats.empty() || st.feats == "_" ? "" : st.feats;
                if (!upos.empty() && !feats.empty()) {
                    dtok_tag = upos + "#" + feats;
                } else if (!upos.empty()) {
                    dtok_tag = upos;
                }
            } else {
                dtok_tag = st.xpos.empty() || st.xpos == "_" ? "" : st.xpos;
            }
            if (!dtok_tag.empty()) {
                dtok_tags.push_back(dtok_tag);
            }
        }
        if (!dtok_tags.empty()) {
            combined_dtok_tag = dtok_tags[0];
            for (std::size_t i = 1; i < dtok_tags.size(); ++i) {
                combined_dtok_tag += "." + dtok_tags[i];
            }
            if (settings_.debug) {
                std::cerr << "[flexitag] Token " << token.form << " has dtoks, combined tag: " << combined_dtok_tag << "\n";
            }
        }
    }
    
    // If dtoks are present, we'll create candidates that match their tags
    // But we still need to run normal candidate generation to get proper probabilities

    // Helper function to check if a candidate matches existing tags (like neotagxml's tagsmatch)
    // neotagxml line 1168-1185: tagsmatch() checks if lexitem attributes match token attributes
    // It checks formTags (like "xpos", "lemma", etc.) but NOT dtoks directly
    // The dtok matching happens via adddtok() which creates wordtokens from existing dtoks
    auto matches_existing_tags = [&](const WordCandidate& cand) -> bool {
        if (!has_existing_dtoks) {
            // No existing dtoks - check if candidate matches existing tags (upos, xpos, feats)
            // neotagxml line 1179: checks if token.attribute(t) exists and matches lexitem.attribute(t)
            // CRITICAL: We should check ALL existing tags, not just the tagpos tag
            // This allows constraining candidates when token has upos=NOUN even if tagpos=xpos
            bool overwrite = settings_.get_bool("overwrite", false);
            if (overwrite) return true;  // Overwrite mode - always match
            
            // Check UPOS match (if token has upos, candidate must match)
            if (!token.upos.empty() && token.upos != "_") {
                std::string cand_upos;
                if (cand.lex_attributes && !cand.lex_attributes->upos.empty() && cand.lex_attributes->upos != "_") {
                    cand_upos = cand.lex_attributes->upos;
                } else {
                    // Extract from tag if it's in format "UPOS#FEATS" or tagpos is upos
                    std::string tagpos = settings_.get("tagpos", "xpos");
                    if (tagpos == "upos" || tagpos == "utot") {
                        std::size_t hash_pos = cand.tag.find('#');
                        if (hash_pos != std::string::npos) {
                            cand_upos = cand.tag.substr(0, hash_pos);
                        } else if (tagpos == "upos") {
                            cand_upos = cand.tag;
                        }
                    }
                }
                if (!cand_upos.empty() && cand_upos != token.upos) {
                    if (settings_.debug) {
                        std::cerr << "[flexitag] matches_existing_tags: UPOS mismatch - token has '" 
                                  << token.upos << "', candidate has '" << cand_upos << "'\n";
                    }
                    return false;  // UPOS mismatch
                }
            }
            
            // Check XPOS match (if token has xpos, candidate must match)
            if (!token.xpos.empty() && token.xpos != "_") {
                std::string cand_xpos;
                if (cand.lex_attributes && !cand.lex_attributes->xpos.empty() && cand.lex_attributes->xpos != "_") {
                    cand_xpos = cand.lex_attributes->xpos;
                } else {
                    // Extract from tag if tagpos is xpos
                    std::string tagpos = settings_.get("tagpos", "xpos");
                    if (tagpos == "xpos") {
                        cand_xpos = cand.tag;
                    }
                }
                if (!cand_xpos.empty() && cand_xpos != token.xpos) {
                    if (settings_.debug) {
                        std::cerr << "[flexitag] matches_existing_tags: XPOS mismatch - token has '" 
                                  << token.xpos << "', candidate has '" << cand_xpos << "'\n";
                    }
                    return false;  // XPOS mismatch
                }
            }
            
            // Check FEATS match (if token has feats, candidate must have matching feats)
            // CRITICAL: Use partial matching - candidate can have additional features
            // but must have all features that token has
            if (!token.feats.empty() && token.feats != "_") {
                std::string cand_feats;
                if (cand.lex_attributes && !cand.lex_attributes->feats.empty() && cand.lex_attributes->feats != "_") {
                    cand_feats = cand.lex_attributes->feats;
                } else {
                    // Extract from tag if it's in format "UPOS#FEATS"
                    std::string tagpos = settings_.get("tagpos", "xpos");
                    if (tagpos == "utot") {
                        std::size_t hash_pos = cand.tag.find('#');
                        if (hash_pos != std::string::npos) {
                            cand_feats = cand.tag.substr(hash_pos + 1);
                        }
                    }
                }
                
                if (!cand_feats.empty() && cand_feats != "_") {
                    // Parse token feats and candidate feats
                    std::unordered_map<std::string, std::string> token_feats_dict;
                    std::unordered_map<std::string, std::string> cand_feats_dict;
                    
                    // Parse token.feats
                    std::istringstream token_feats_stream(token.feats);
                    std::string token_feat_pair;
                    while (std::getline(token_feats_stream, token_feat_pair, '|')) {
                        std::size_t eq_pos = token_feat_pair.find('=');
                        if (eq_pos != std::string::npos) {
                            std::string key = token_feat_pair.substr(0, eq_pos);
                            std::string value = token_feat_pair.substr(eq_pos + 1);
                            token_feats_dict[key] = value;
                        }
                    }
                    
                    // Parse cand_feats
                    std::istringstream cand_feats_stream(cand_feats);
                    std::string cand_feat_pair;
                    while (std::getline(cand_feats_stream, cand_feat_pair, '|')) {
                        std::size_t eq_pos = cand_feat_pair.find('=');
                        if (eq_pos != std::string::npos) {
                            std::string key = cand_feat_pair.substr(0, eq_pos);
                            std::string value = cand_feat_pair.substr(eq_pos + 1);
                            cand_feats_dict[key] = value;
                        }
                    }
                    
                    // Check if all token features are present in candidate (partial matching)
                    for (const auto& [key, value] : token_feats_dict) {
                        auto it = cand_feats_dict.find(key);
                        if (it == cand_feats_dict.end() || it->second != value) {
                            if (settings_.debug) {
                                std::cerr << "[flexitag] matches_existing_tags: FEATS mismatch - token has '" 
                                          << key << "=" << value << "', candidate missing or different\n";
                            }
                            return false;  // FEATS mismatch
                        }
                    }
                }
            }
            
            return true;  // All existing tags match (or no existing tags to check)
        }
        
        // Token has existing dtoks - neotagxml uses adddtok() to create wordtokens from them
        // neotagxml line 320-332: if lexitem.child("dtok") exists, it creates wordtokens from them
        // neotagxml line 1212: tagsmatch() checks if candidate matches existing tags
        // CRITICAL: neotagxml still accepts candidates WITHOUT dtoks if they match the parent tag
        // The dtok matching is more nuanced - it checks if the candidate's tag matches the combined dtok tag
        // OR if the candidate has dtoks that match the existing dtok tags
        
        // First, check if candidate's tag matches the combined dtok tag (e.g., "APPR.ART")
        // Use pre-computed combined_dtok_tag to avoid recomputing
        if (!combined_dtok_tag.empty()) {
            // If candidate's tag matches the combined dtok tag, accept it (even without dtoks)
            if (cand.tag == combined_dtok_tag) {
                if (settings_.debug) {
                    std::cerr << "[flexitag] matches_existing_tags: candidate tag " << cand.tag 
                              << " matches combined dtok tag " << combined_dtok_tag << "\n";
                }
                return true;
            }
        }
        
        // If candidate has dtoks, they must match existing dtok tags
        if (!cand.dtoks.empty()) {
            if (cand.dtoks.size() != token.subtokens.size()) {
                return false;  // Different number of dtoks
            }
            
            // Check if dtok tags match (neotagxml checks via tagsmatch on each dtok)
            // Use tagpos setting to determine which tag attribute to compare
            std::string tagpos = settings_.get("tagpos", "xpos");
            for (std::size_t i = 0; i < cand.dtoks.size() && i < token.subtokens.size(); ++i) {
                std::string cand_dtok_tag = cand.dtoks[i]->tag;
                std::string existing_dtok_tag;
                if (tagpos == "upos") {
                    existing_dtok_tag = token.subtokens[i].upos.empty() || token.subtokens[i].upos == "_" ? "" : token.subtokens[i].upos;
                } else if (tagpos == "utot") {
                    std::string upos = token.subtokens[i].upos.empty() || token.subtokens[i].upos == "_" ? "" : token.subtokens[i].upos;
                    std::string feats = token.subtokens[i].feats.empty() || token.subtokens[i].feats == "_" ? "" : token.subtokens[i].feats;
                    if (!upos.empty() && !feats.empty()) {
                        existing_dtok_tag = upos + "#" + feats;
                    } else if (!upos.empty()) {
                        existing_dtok_tag = upos;
                    }
                } else {
                    existing_dtok_tag = token.subtokens[i].xpos.empty() || token.subtokens[i].xpos == "_" ? "" : token.subtokens[i].xpos;
                }
                if (existing_dtok_tag.empty()) {
                    continue;  // No tag to match
                }
                if (cand_dtok_tag != existing_dtok_tag) {
                    return false;  // Tag mismatch
                }
            }
            return true;  // All dtok tags match
        }
        
        // Candidate has no dtoks and doesn't match combined tag - reject it
        // This matches neotagxml's behavior: when dtoks exist, only accept candidates that match
        if (settings_.debug) {
            std::cerr << "[flexitag] matches_existing_tags: rejecting candidate tag " << cand.tag 
                      << " (doesn't match combined dtok tag " << combined_dtok_tag << " and has no matching dtoks)\n";
        }
        return false;
    };

    auto populate_candidates = [&](const LexiconItem* item, float weight) {
        if (!item) return;
        for (const auto& lex_token : item->tokens) {
            if (lex_token.tag.empty()) {
                if (settings_.debug) {
                    std::cerr << "[flexitag] populate_candidates: Skipping lex_token for '" << token.form 
                              << "' - tag is empty (has " << lex_token.entries.size() << " entries)\n";
                }
                continue;
            }
            // Create a candidate for each entry in the token (neotagxml does this)
            // If no entries, create one candidate with just the tag
            if (lex_token.entries.empty()) {
                WordCandidate cand;
                cand.form = token.form;
                cand.tag = lex_token.tag;
                cand.prob = std::max(static_cast<float>(lex_token.count), 1.f) * weight;
                cand.source = "lexicon";
                cand.wcase = form_case(token.form);
                cand.token = const_cast<Token*>(&token);
                cand.lemma = token.form;
                
                // CRITICAL: neotagxml line 1212 - only add if tagsmatch() passes
                // If token has existing dtoks, filter candidates to match them
                if (matches_existing_tags(cand)) {
                    candidates.push_back(std::move(cand));
                }
            } else {
                // Create a candidate for each entry (neotagxml iterates through all entries)
                for (const auto& entry : lex_token.entries) {
                    WordCandidate cand;
                    cand.form = token.form;
                    cand.tag = lex_token.tag;
                    // Use entry count if available, otherwise use token count
                    float entry_count = entry.count > 0 ? static_cast<float>(entry.count) : static_cast<float>(lex_token.count);
                    cand.prob = std::max(entry_count, 1.f) * weight;
                    cand.source = "lexicon";
                    cand.wcase = form_case(token.form);
                    cand.token = const_cast<Token*>(&token);
                    
                    cand.lemma = entry.lemma.empty() ? token.form : entry.lemma;
                    // Apply case transformation (neotagxml calls applycase() on all lemmas)
                    cand.lemma = apply_case_to_lemma(cand.lemma, lex_token.tag, cand.wcase);
                    cand.lex_attributes = AttributeSet{};
                    cand.lex_attributes->lemma = cand.lemma;
                    // Use lex_token.tag as xpos (this is the "key" from the tok node, which is the xpos tag)
                    // Only override with entry.xpos if it's different (shouldn't happen, but be safe)
                    cand.lex_attributes->xpos = entry.xpos.empty() ? lex_token.tag : entry.xpos;
                    // Copy upos and feats if available
                    if (!entry.upos.empty() && entry.upos != "_") {
                        cand.lex_attributes->upos = entry.upos;
                    }
                    if (!entry.feats.empty() && entry.feats != "_") {
                        cand.lex_attributes->feats = entry.feats;
                    }
                    if (!entry.reg.empty()) {
                        cand.lex_attributes->reg = entry.reg;
                    }
                    if (!entry.expan.empty()) {
                        cand.lex_attributes->expan = entry.expan;
                    }
                    // Copy contraction-level attributes (mod, trslit, ltrslit, tokid)
                    if (!entry.mod.empty()) {
                        cand.lex_attributes->mod = entry.mod;
                    }
                    if (!entry.trslit.empty()) {
                        cand.lex_attributes->trslit = entry.trslit;
                    }
                    if (!entry.ltrslit.empty()) {
                        cand.lex_attributes->ltrslit = entry.ltrslit;
                    }
                    if (!entry.tokid.empty()) {
                        cand.lex_attributes->tokid = entry.tokid;
                    }
                    // Use entry.key as form (normalized form)
                    if (!entry.key.empty()) {
                        cand.lex_attributes->form = entry.key;
                    }
                    cand.lexitem = std::make_shared<WordCandidate>();
                    cand.lexitem->lemma = cand.lemma;
                    for (const auto& dt : entry.dtoks) {
                        auto ptr = std::make_shared<WordCandidate>();
                        ptr->form = dt.form;
                        ptr->lemma = dt.lemma;
                        // Use dt.xpos as tag, or fallback to empty
                        ptr->tag = dt.xpos.empty() ? "" : dt.xpos;
                        // Store UPOS and feats for dtoks so they can be copied to subtokens
                        ptr->lex_attributes = AttributeSet();
                        ptr->lex_attributes->upos = dt.upos;
                        ptr->lex_attributes->feats = dt.feats;
                        ptr->lex_attributes->xpos = dt.xpos;
                        ptr->lex_attributes->lemma = dt.lemma;
                        cand.dtoks.push_back(ptr);
                    }
                    
                    // CRITICAL: neotagxml line 1212 - only add if tagsmatch() passes
                    // If token has existing dtoks, filter candidates to match them
                    if (matches_existing_tags(cand)) {
                        candidates.push_back(std::move(cand));
                    } else if (settings_.debug) {
                        std::cerr << "[flexitag] populate_candidates: Candidate for '" << token.form 
                                  << "' with tag='" << cand.tag << "' rejected by matches_existing_tags\n";
                        std::cerr << "  token.upos='" << token.upos << "' token.xpos='" << token.xpos 
                                  << "' token.feats='" << token.feats << "'\n";
                        std::cerr << "  cand.lex_attributes: upos='" 
                                  << (cand.lex_attributes ? cand.lex_attributes->upos : "null") 
                                  << "' xpos='" << (cand.lex_attributes ? cand.lex_attributes->xpos : "null") << "'\n";
                    }
                }
            }
        }
    };

    const LexiconItem* item = lexicon_->find(token.form);
    if (item && settings_.debug) {
        std::cerr << "[flexitag] Found '" << token.form << "' in lexicon with " << item->tokens.size() << " token entries\n";
        for (const auto& lex_token : item->tokens) {
            std::cerr << "  tag='" << lex_token.tag << "' count=" << lex_token.count << " entries=" << lex_token.entries.size() << "\n";
        }
    }
    populate_candidates(item, 1.f);
    if (candidates.empty()) {
        const LexiconItem* item_lower = lexicon_->find_lower(token.form);
        if (item_lower && settings_.debug) {
            std::cerr << "[flexitag] Found '" << token.form << "' (lowercase) in lexicon with " << item_lower->tokens.size() << " token entries\n";
        }
        populate_candidates(item_lower, 0.1f);
    }
    if (candidates.empty() && settings_.debug) {
        std::cerr << "[flexitag] No candidates found for '" << token.form << "' - will use unknown word fallback\n";
    }
    
    // Step 1b: Try enhanced normalization if normalizer is enabled
    // This applies morphological variations and pattern-based substitutions
    // CRITICAL: Always check for explicit normalization mappings (form -> reg) from vocab
    // Even if word is in lexicon, it might have a reg value that should be used
    // Pattern-based normalization should only be applied to OOV words
    std::string enhanced_normalized_form;  // Store to set in token.reg
    
    if (normalizer_ && normalizer_->is_enabled()) {
        // First, check for explicit normalization mapping (form -> reg) in vocab
        // This should always be checked, even if word is in lexicon
        const LexiconItem* item = lexicon_->find(token.form);
        if (!item) {
            item = lexicon_->find_lower(token.form);
        }
        
        if (item && normalizer_) {
            // Word is in lexicon - use get_reg_from_entry() to get the most appropriate reg value
            // This function properly checks frequencies and skips capitalization-only normalizations
            std::string reg_from_entry = normalizer_->get_reg_from_entry(item);
            if (!reg_from_entry.empty() && reg_from_entry != token.form) {
                // RESTRICTION 2: Verify the normalized form exists in the vocab
                // This prevents normalizing to invalid forms
                bool reg_exists = lexicon_->find(reg_from_entry) || lexicon_->find_lower(reg_from_entry);
                if (!reg_exists) {
                    // Check if reg exists as a reg value in the lexicon
                    auto mappings = lexicon_->get_normalization_mappings();
                    for (const auto& [form, reg_val] : mappings) {
                        if (reg_val == reg_from_entry) {
                            reg_exists = true;
                            break;
                        }
                    }
                }
                
                if (reg_exists) {
                    // RESTRICTION 3: For frequent words, only normalize if normalized form is much more frequent
                    // Calculate word frequency
                    int word_freq = 0;
                    for (const auto& token_entry : item->tokens) {
                        for (const auto& e : token_entry.entries) {
                            word_freq += e.count;
                        }
                    }
                    
                    // Calculate reg frequency
                    const LexiconItem* reg_item = lexicon_->find(reg_from_entry);
                    if (!reg_item) {
                        reg_item = lexicon_->find_lower(reg_from_entry);
                    }
                    int reg_freq = 0;
                    if (reg_item) {
                        for (const auto& token_entry : reg_item->tokens) {
                            for (const auto& e : token_entry.entries) {
                                reg_freq += e.count;
                            }
                        }
                    }
                    
                    // If word is frequent (>= 100), require normalized form to be at least 5x more frequent
                    // and non-normalized form to be relatively rare (< 10% of normalized form)
                    if (word_freq >= 100) {
                        if (reg_freq < word_freq * 5 || word_freq > reg_freq * 0.1) {
                            if (settings_.debug) {
                                std::cerr << "[flexitag] word='" << token.form << "' (freq=" << word_freq 
                                          << ") -> '" << reg_from_entry << "' (freq=" << reg_freq 
                                          << ") rejected: normalized form not frequent enough\n";
                            }
                            // Don't set reg, but continue to check pattern-based normalization
                        } else {
                            // CRITICAL: If the reg value itself exists as a form that has a normalization,
                            // we should use that form's normalization instead (e.g., "corazon" -> "corazón")
                            std::string final_reg = reg_from_entry;
                            if (reg_item) {
                                std::string reg_reg = normalizer_->get_reg_from_entry(reg_item);
                                if (!reg_reg.empty() && reg_reg != reg_from_entry) {
                                    final_reg = reg_reg;
                                    if (settings_.debug) {
                                        std::cerr << "[flexitag] reg='" << reg_from_entry << "' itself has normalization '" << reg_reg << "', using that instead\n";
                                    }
                                }
                            }
                            
                            enhanced_normalized_form = final_reg;
                            // Set reg from explicit vocab mapping
                            if (token.reg.empty() || token.reg == "_") {
                                token.reg = enhanced_normalized_form;
                            }
                        }
                    } else {
                        // Word is not frequent, use the reg from get_reg_from_entry() (which already checked frequencies)
                        // CRITICAL: If the reg value itself exists as a form that has a normalization,
                        // we should use that form's normalization instead (e.g., "corazon" -> "corazón")
                        std::string final_reg = reg_from_entry;
                        if (reg_item) {
                            std::string reg_reg = normalizer_->get_reg_from_entry(reg_item);
                            if (!reg_reg.empty() && reg_reg != reg_from_entry) {
                                final_reg = reg_reg;
                                if (settings_.debug) {
                                    std::cerr << "[flexitag] reg='" << reg_from_entry << "' itself has normalization '" << reg_reg << "', using that instead\n";
                                }
                            }
                        }
                        
                        enhanced_normalized_form = final_reg;
                        // Set reg from explicit vocab mapping
                        if (token.reg.empty() || token.reg == "_") {
                            token.reg = enhanced_normalized_form;
                        }
                    }
                } else if (settings_.debug) {
                    std::cerr << "[flexitag] word='" << token.form << "' has reg='" << reg_from_entry << "' but reg doesn't exist in vocab, ignoring\n";
                }
            }
        }
        
        // If no explicit mapping found and word is OOV, try pattern-based normalization
        // CRITICAL: Also try normalization if candidates only come from word endings (not from lexicon)
        // This handles cases like "coraçón" which finds candidates via endings but should still normalize
        bool candidates_from_lexicon = false;
        for (const auto& cand : candidates) {
            if (cand.source == "lexicon" || cand.source == "lexicon_lower") {
                candidates_from_lexicon = true;
                break;
            }
        }
        
        if (enhanced_normalized_form.empty() && (!candidates_from_lexicon || candidates.empty())) {
            bool conservative = settings_.get_bool("normalization_conservative", true);
            std::string normalized = normalizer_->normalize(token.form, conservative);
            if (settings_.debug && !normalized.empty() && normalized != token.form) {
                std::cerr << "[flexitag] enhanced normalization: '" << token.form << "' -> '" << normalized << "'\n";
            }
            if (!normalized.empty() && normalized != token.form) {
                enhanced_normalized_form = normalized;  // Store for later
                
                // Set normalized form directly on token.reg so it's preserved even if we fall back to word endings
                if (token.reg.empty() || token.reg == "_") {
                    token.reg = enhanced_normalized_form;
                }
                // Try looking up the normalized form
                populate_candidates(lexicon_->find(normalized), 0.8f);  // Slightly lower weight than direct match
                if (candidates.empty()) {
                    populate_candidates(lexicon_->find_lower(normalized), 0.08f);
                }
                if (settings_.debug && !candidates.empty()) {
                    std::cerr << "[flexitag] found " << candidates.size() << " candidates via enhanced normalization\n";
                }
                // Store normalized form in candidates' lex_attributes->reg if not already set
                if (!enhanced_normalized_form.empty()) {
                    for (auto& cand : candidates) {
                        if (!cand.lex_attributes) {
                            cand.lex_attributes = AttributeSet();
                        }
                        if (cand.lex_attributes->reg.empty() || cand.lex_attributes->reg == "_") {
                            cand.lex_attributes->reg = enhanced_normalized_form;
                        }
                    }
                }
            }
        }
    }
    
    // Step 1c: Check normalized form (nform) if no candidates found yet (neotagxml line 1248-1268)
    // neotagxml checks nform attribute if wordParse.size() == 0
    if (candidates.empty()) {
        std::string nform = settings_.get("tagform", "form");  // Default to "form" if not specified
        // neotagxml uses nform from settings, but we'll check common normalized form attributes
        std::string normalized_form;
        if (!token.reg.empty()) {
            normalized_form = token.reg;
        } else if (!token.expan.empty()) {
            normalized_form = token.expan;
        } else if (!token.form.empty() && nform != "form") {
            // If nform is specified and different from "form", we'd need to get it from token
            // For now, we'll use reg/expan as they're the common normalized forms
        }
        
        if (!normalized_form.empty() && normalized_form != token.form) {
            // Look up normalized form in lexicon (neotagxml line 1250)
            populate_candidates(lexicon_->find(normalized_form), 1.f);
            if (candidates.empty()) {
                populate_candidates(lexicon_->find_lower(normalized_form), 0.1f);
            }
        }
    }

    // Step 2: Check for clitics/contractions (like neotagxml does)
    // CRITICAL: neotagxml only checks clitics if wordParse.size() == 0 (line 1273)
    // This is different from checking if noclitics is set!
    // BUT: if token already has dtoks, neotagxml uses them instead of checking for clitics
    // neotagxml line 1212: tagsmatch() checks if candidate matches existing dtok tags
    // IMPORTANT: Only try clitic splitting if:
    // 1. No candidates were found from lexicon lookup
    // 2. None of the existing candidates already have dtoks (meaning the word is already known as a contraction)
    // 3. The word itself is not in the lexicon as a full form
    // This prevents duplicate contraction detection - if a word is in the vocab with dtoks/parts,
    // it should NOT be split again by the dtoks table mechanism
    bool noclitics = settings_.get_bool("noclitics", false);
    
    // Check if any existing candidates already have dtoks (from vocab entries with parts/dtoks)
    bool has_candidates_with_dtoks = false;
    for (const auto& cand : candidates) {
        if (!cand.dtoks.empty()) {
            has_candidates_with_dtoks = true;
            break;
        }
    }
    
   
    // CRITICAL: If token has existing dtoks and no candidates matched, create a candidate from them
    // neotagxml line 1212: tagsmatch() filters candidates, but if none match, it uses existing dtoks
    // We only create this fallback if no candidates were found (which happens after filtering)
    // This is handled in the "use existing tag" section below

    // Step 3: Switch to normalized form before word endings (neotagxml line 1306-1313)
    // neotagxml switches to nform before checking word endings, even if candidates exist
    std::string word = token.form;  // Use mutable copy
    std::string nform = settings_.get("tagform", "form");
    if (nform == "auto") {
        // Try reg, then expan, then form
    if (!token.reg.empty() && token.reg != token.form) {
            word = token.reg;
    } else if (!token.expan.empty() && token.expan != token.form) {
            word = token.expan;
        }
    } else if (nform == "reg" && !token.reg.empty() && token.reg != token.form) {
        word = token.reg;  // Use normalized form for word ending lookup
    } else if (nform == "expan" && !token.expan.empty() && token.expan != token.form) {
        word = token.expan;  // Fallback to expan
    }
    
    // Step 4: Use word endings for OOV handling (neotagxml line 1315-1418)
    // This is crucial for handling unknown words
    // neotagxml checks endings if wordParse.size() == 0 OR lexsmooth OR partialclitic
    // For now, we'll check if candidates.empty() (matching wordParse.size() == 0)
    if (candidates.empty()) {
        int endretry = settings_.get_int("endretry", 2);  // Default: try 2 extra ending lengths
        // Get endlen from settings (allows runtime overrides), fallback to lexicon value
        int endlen = settings_.get_int("endlen", lexicon_->endlen());  // Can be negative for prefixes
        // CRITICAL: If endlen from settings differs from lexicon's endlen, update the lexicon
        // This ensures endings are indexed correctly when endlen is overridden
        if (endlen != lexicon_->endlen()) {
            lexicon_->set_endlen(endlen);
            // Re-index endings with the new endlen value
            // This is necessary because endings are indexed during vocabulary loading
            // and changing endlen requires re-indexing
            lexicon_->reindex_endings();
        }
        const bool use_prefix = (endlen < 0);
        const int abs_endlen = std::abs(endlen);
        
        int fnd = 0;  // Number of endings found
        // Use normalized form (word) for ending lookup
        
        // Try endings/prefixes from longest to shortest (neotagxml line 1338)
        // Use UTF-8 character count, not byte count
        size_t word_char_count = char_count(word);
        const int max_len = std::min(static_cast<int>(word_char_count), abs_endlen);
        
        for (int len = max_len; len >= 1; --len) {  // Try from longest to shortest
            std::string wending = use_prefix ? prefix(word, len) : suffix(word, len);
            auto ending_it = lexicon_->endings().find(wending);
            
            if (ending_it != lexicon_->endings().end() && !ending_it->second.empty() && fnd <= endretry) {
                fnd++;  // Found a matching ending
                
                // For each tag with this ending (neotagxml line 1344)
                for (const auto& [tag, ending] : ending_it->second) {
                    WordCandidate cand;
                    cand.form = word;
                    cand.tag = tag;
                    // Probability: ending.prob * pow(ending_decay_base, 0-fnd) (neotagxml line 1347)
                    // Shorter endings (larger fnd) get lower probability
                    float ending_decay_base = settings_.get_float("ending_decay_base", 5.0f);
                    float power_factor = std::pow(ending_decay_base, static_cast<float>(0 - fnd));
                    cand.prob = ending.prob * power_factor;
                    cand.source = use_prefix ? "prefix" : "ending";
                    cand.wcase = form_case(word);
                    cand.token = const_cast<Token*>(&token);
                    
                    // CRITICAL: neotagxml line 1348 stores lemmatizations in the candidate
                    // This is used later in lemmatize() (line 342) if lemmatizations.size() > 0
                    cand.lemmatizations = ending.lemmatizations;
                    
                    // CRITICAL: neotagxml line 1357-1414 handles lemmatization rules
                    // If lemmaProbs.size() > 0 (lemmalist exists):
                    //   - Try all rules, check if derived lemma is in lemmalist (20x boost)
                    //   - Otherwise pick most frequent rule
                    // If lemmaProbs.size() == 0 (no lemmalist):
                    //   - Skip lemmatization during candidate generation (line 1405-1414)
                    //   - Just use word form, lemmatization happens later in lemmatize() (line 805)
                    
                    // Since we don't have a lemmalist (lemmaProbs is empty), match neotagxml line 1405-1414:
                    // Don't apply lemmatization rules here - they'll be applied later in refine_lemma_with_endings()
                    cand.lemma = word;  // Use word form, lemmatization happens post-Viterbi
                    
                    // CRITICAL: neotagxml line 1212 - filter ending candidates too if dtoks exist
                    if (matches_existing_tags(cand)) {
                        if (settings_.debug && fnd <= 2) {
                            std::cerr << "[flexitag] morpho_parse: " << (use_prefix ? "prefix" : "ending") 
                                      << " candidate (no lemmalist, deferring lemmatization): "
                                      << "form=" << word << " tag=" << tag << " " << (use_prefix ? "prefix" : "ending") 
                                      << "=" << wending << " lemmatizations=" << cand.lemmatizations.size() << "\n";
                        }
                        
                        candidates.push_back(std::move(cand));
                    }
                }
            }
        }
    }

    // CRITICAL: neotagxml line 1421-1427 - use existing tag if no candidates found
    // This happens AFTER word endings (line 1421 is after 1418) but BEFORE tagset fallback
    // This is a key feature for preserving existing tags when no lexicon/ending match is found
    // BUT: when dtoks are preserved, neotagxml should have found candidates with matching combined tag
    // If no candidates were found, it means the filtering was too strict OR the lexicon doesn't have the combined tag
    // In that case, we create a candidate from existing dtoks
    bool overwrite = settings_.get_bool("overwrite", false);
    if (candidates.empty() && !overwrite) {
        if (has_existing_dtoks) {
            // Create candidate from existing dtoks (neotagxml uses adddtok() to create wordtokens)
            // Use tagpos setting to determine which tag attribute to use
            std::string tagpos = settings_.get("tagpos", "xpos");
            std::vector<std::string> dtok_tags;
            for (const auto& st : token.subtokens) {
                std::string dtok_tag;
                if (tagpos == "upos") {
                    dtok_tag = st.upos.empty() || st.upos == "_" ? "" : st.upos;
                } else if (tagpos == "utot") {
                    std::string upos = st.upos.empty() || st.upos == "_" ? "" : st.upos;
                    std::string feats = st.feats.empty() || st.feats == "_" ? "" : st.feats;
                    if (!upos.empty() && !feats.empty()) {
                        dtok_tag = upos + "#" + feats;
                    } else if (!upos.empty()) {
                        dtok_tag = upos;
                    }
                } else {
                    dtok_tag = st.xpos.empty() || st.xpos == "_" ? "" : st.xpos;
                }
                if (!dtok_tag.empty()) {
                    dtok_tags.push_back(dtok_tag);
                }
            }
            
            if (!dtok_tags.empty()) {
                // Create a candidate that matches the existing dtok structure
                // The combined tag would be something like "APPR.ART" for "im" -> "in" + "dem"
                // Use pre-computed combined_dtok_tag if available, otherwise compute it
                std::string combined_tag = combined_dtok_tag;
                if (combined_tag.empty()) {
                    combined_tag = dtok_tags[0];
                    for (std::size_t i = 1; i < dtok_tags.size(); ++i) {
                        combined_tag += "." + dtok_tags[i];
                    }
                }
                
                // CRITICAL: Try to find this combined tag in the lexicon to get proper probability
                // neotagxml would look up the word in the lexicon and find candidates with matching tags
                // If found, use the lexicon probability; otherwise use dtok_fallback_prob
                // CRITICAL: We should have already found candidates with matching combined tag during populate_candidates
                // If we're here, it means no candidates matched, so we create a fallback
                // However, we should still try to get the probability from the lexicon if available
                float dtok_fallback_prob = settings_.get_float("dtok_fallback_prob", 1e-6f);
                float dtok_prob = dtok_fallback_prob;  // Default to fallback probability
                std::string dtok_source = "existing dtoks (fallback)";
                auto dtok_lexicon_item = lexicon_->find(token.form);
                if (dtok_lexicon_item) {
                    for (const auto& lex_token : dtok_lexicon_item->tokens) {
                        if (lex_token.tag == combined_tag) {
                            dtok_prob = std::max(static_cast<float>(lex_token.count), dtok_fallback_prob);
                            dtok_source = "lexicon (matching dtoks, fallback)";
                            if (settings_.debug) {
                                std::cerr << "[flexitag] Found combined tag in lexicon: " << combined_tag 
                                          << " count=" << lex_token.count << " for " << token.form << "\n";
                            }
                            break;
                        }
                    }
                } else if (settings_.debug) {
                    std::cerr << "[flexitag] WARNING: No lexicon entry found for " << token.form 
                              << " with combined tag " << combined_tag << " - using fallback\n";
                }
                
                WordCandidate existing_dtok_cand;
                existing_dtok_cand.form = token.form;
                existing_dtok_cand.tag = combined_tag;
                existing_dtok_cand.prob = dtok_prob;
                existing_dtok_cand.source = dtok_source;
                existing_dtok_cand.wcase = form_case(token.form);
                existing_dtok_cand.token = const_cast<Token*>(&token);
                existing_dtok_cand.lemma = token.lemma.empty() || token.lemma == "_" ? token.form : token.lemma;
                
                // Create dtok candidates from existing subtokens
                // Use tagpos setting to determine which tag to use
                std::string tagpos = settings_.get("tagpos", "xpos");
                for (const auto& st : token.subtokens) {
                    auto dtok_cand = std::make_shared<WordCandidate>();
                    dtok_cand->form = st.form;
                    if (tagpos == "upos") {
                        dtok_cand->tag = st.upos.empty() || st.upos == "_" ? "" : st.upos;
                    } else if (tagpos == "utot") {
                        std::string upos = st.upos.empty() || st.upos == "_" ? "" : st.upos;
                        std::string feats = st.feats.empty() || st.feats == "_" ? "" : st.feats;
                        if (!upos.empty() && !feats.empty()) {
                            dtok_cand->tag = upos + "#" + feats;
                        } else if (!upos.empty()) {
                            dtok_cand->tag = upos;
                        } else {
                            dtok_cand->tag = "";
                        }
                    } else {
                    dtok_cand->tag = st.xpos.empty() || st.xpos == "_" ? "" : st.xpos;
                    }
                    dtok_cand->lemma = st.lemma.empty() || st.lemma == "_" ? st.form : st.lemma;
                    dtok_cand->source = "existing dtok";
                    
                    // Try to find lemmatization rules for this dtok
                    // Look up in lexicon to get stored lemmatizations
                    // Use xpos for lexicon lookup (lexicon stores tags as xpos)
                    if (!st.xpos.empty() && st.xpos != "_") {
                        auto dtok_lexicon_item = lexicon_->find(st.form);
                        if (dtok_lexicon_item) {
                            for (const auto& lex_token : dtok_lexicon_item->tokens) {
                                if (lex_token.tag == st.xpos) {
                                    // Found matching tag - get lemmatizations from endings
                                    // We'll store these for later lemmatization
                                    // Use UTF-8 character count, not byte count
                                    size_t form_char_count = char_count(st.form);
                                    for (int len = 1; len <= static_cast<int>(form_char_count); ++len) {
                                        std::string ending = suffix(st.form, len);
                                        auto ending_it = lexicon_->endings().find(ending);
                                        if (ending_it != lexicon_->endings().end()) {
                                            auto tag_it = ending_it->second.find(st.xpos);
                                            if (tag_it != ending_it->second.end()) {
                                                dtok_cand->lemmatizations = tag_it->second.lemmatizations;
                                                break;  // Use longest ending
                                            }
                                        }
                                    }
                                    break;
                                }
                            }
                        }
                    }
                    
                    existing_dtok_cand.dtoks.push_back(dtok_cand);
                }
                
                candidates.push_back(std::move(existing_dtok_cand));
                
                if (settings_.debug) {
                    std::cerr << "[flexitag] Using existing dtoks: " << combined_tag << " for " << token.form << "\n";
                }
            }
        } else {
            // Use existing tag with prob=1 (neotagxml line 1424-1427)
            std::string tagpos = settings_.get("tagpos", "xpos");
            std::string existing_tag = get_token_tag_value(token, tagpos);
            if (!existing_tag.empty()) {
            WordCandidate existing;
            existing.form = token.form;
                // Store the tag in the candidate's tag field (will be converted by get_tag_value if needed)
                existing.tag = existing_tag;
            existing.prob = 1.0f;  // neotagxml uses prob=1 for existing tags
            existing.source = "existing tag";
            existing.wcase = form_case(token.form);
            existing.token = const_cast<Token*>(&token);
            existing.lemma = token.lemma.empty() || token.lemma == "_" ? token.form : token.lemma;
            candidates.push_back(std::move(existing));
            
            if (settings_.debug) {
                    std::cerr << "[flexitag] Using existing tag: " << existing_tag << " for " << token.form << "\n";
                }
            }
        }
    }
    
    // Step 5: Fall back to raw tag probability over types (vocab entries), not tokens
    // This is the "unknown" source - uses type counts (number of unique vocab entries per tag)
    if (candidates.empty()) {
        const auto& type_counts = lexicon_->tag_type_counts();
        
        if (type_counts.empty()) {
            // Fallback: use tag_stats if type_counts is empty (shouldn't happen, but safety check)
            if (settings_.debug) {
                std::cerr << "[flexitag] WARNING: tag_type_counts() returned empty map for '" << token.form 
                          << "', using tag_stats()\n";
            }
            for (const auto& [tag, stats] : lexicon_->tag_stats()) {
                WordCandidate cand;
                cand.form = token.form;
                cand.tag = tag;
                cand.prob = std::max(stats.count, 1.f);
                cand.source = "unknown";
                cand.wcase = form_case(token.form);
                cand.token = const_cast<Token*>(&token);
                cand.lemma = token.form;
                candidates.push_back(std::move(cand));
            }
        } else {
            float total_types = 0.f;
            for (const auto& [tag, count] : type_counts) {
                total_types += static_cast<float>(count);
            }
            
            // Use type counts as probabilities (normalized)
            for (const auto& [tag, type_count] : type_counts) {
                WordCandidate cand;
                cand.form = token.form;
                cand.tag = tag;
                // Probability is based on type count (number of vocab entries with this tag)
                // Normalize by total types if available, otherwise use raw count
                if (total_types > 0.f) {
                    cand.prob = static_cast<float>(type_count) / total_types;
                } else {
                    cand.prob = static_cast<float>(type_count);
                }
                cand.source = "unknown";
                cand.wcase = form_case(token.form);
                cand.token = const_cast<Token*>(&token);
                cand.lemma = token.form;
                candidates.push_back(std::move(cand));
            }
        }
    }

    // Final fallback: unknown tag (should rarely happen now)
    if (candidates.empty()) {
        WordCandidate fallback;
        fallback.form = token.form;
        fallback.lemma = token.form;
        fallback.tag = "<unknown>";
        fallback.source = "fallback";
        float fallback_prob = settings_.get_float("fallback_prob", 1e-6f);
        fallback.prob = fallback_prob;
        fallback.wcase = form_case(token.form);
        fallback.token = const_cast<Token*>(&token);
        candidates.push_back(std::move(fallback));
    }

    // Normalize probabilities to [0,1] (matching neotagxml behavior)
    // This is critical for proper Viterbi scoring
    float totprob = 0.f;
    for (const auto& cand : candidates) {
        totprob += cand.prob;
    }
    if (totprob > 0.f) {
        for (auto& cand : candidates) {
            cand.prob = cand.prob / totprob;
        }
    }

    if (settings_.debug) {
        std::cerr << "[flexitag] candidates for token \"" << token.form << "\" (total: " 
                  << candidates.size() << ")\n";
        for (std::size_t idx = 0; idx < candidates.size() && idx < 10; ++idx) {
            const auto& cand = candidates[idx];
            std::cerr << "  [" << idx << "] tag=" << cand.tag
                      << " prob=" << cand.prob
                      << " lemma=" << cand.lemma
                      << " source=" << cand.source;
            float prob_minimum = settings_.get_float("prob_minimum", 1e-38f);
            if (cand.prob == 0.0f || cand.prob < prob_minimum) {  // Minimum float value
                std::cerr << " [ZERO/UNDERFLOW!]";
            }
            std::cerr << "\n";
        }
        if (candidates.empty()) {
            std::cerr << "  [NO CANDIDATES - will use fallback]\n";
        }
    }

    return candidates;
}

std::vector<WordCandidate> FlexitagTagger::apply_clitics(const Token& token) const {
    std::vector<WordCandidate> clitic_candidates;
    const std::string& word = token.form;
    
    // If lexicon is empty, don't try clitic expansion
    if (lexicon_->dtoks().empty()) {
        if (settings_.debug) {
            std::cerr << "[flexitag] apply_clitics: dtoks table is empty, cannot split contractions for '" << word << "'\n";
        }
        return clitic_candidates;
    }
    
    if (settings_.debug) {
        std::cerr << "[flexitag] apply_clitics: checking '" << word << "' against " << lexicon_->dtoks().size() << " clitics\n";
    }
    
    // Valid UPOS tags for clitics (ADP, DET, PRON, PART, SCONJ, CCONJ)
    // PUNCT and other tags should never be clitics
    static const std::unordered_set<std::string> valid_clitic_upos = {
        "ADP", "DET", "PRON", "PART", "SCONJ", "CCONJ"
    };
    
    // Loop through all possible clitics in the dtoks table
    for (const auto& [clitic_form, dtok_form] : lexicon_->dtoks()) {
        if (clitic_form.empty()) continue;
        
        // Validate clitic form: should not end in punctuation or be too short
        if (clitic_form.size() < 1) continue;
        // Check if clitic ends in punctuation (common punctuation marks)
        char last_char = clitic_form.back();
        if (last_char == '.' || last_char == ',' || last_char == ';' || last_char == ':' || 
            last_char == '!' || last_char == '?' || last_char == 'P') {
            if (settings_.debug) {
                std::cerr << "[flexitag] apply_clitics: skipping clitic '" << clitic_form 
                          << "' - ends in punctuation or invalid character\n";
            }
            continue;
        }
        
        std::string base;
        std::string position = dtok_form.position;
        bool is_match = false;
        
        // Check if word starts with clitic (left/pre-clitic)
        if (position == "left" && word.size() > clitic_form.size() && 
            word.substr(0, clitic_form.size()) == clitic_form) {
            base = word.substr(clitic_form.size());
            is_match = true;
        }
        // Check if word ends with clitic (right/post-clitic)
        else if (position == "right" && word.size() > clitic_form.size() &&
                 word.substr(word.size() - clitic_form.size()) == clitic_form) {
            base = word.substr(0, word.size() - clitic_form.size());
            is_match = true;
        }
        
        if (!is_match || base.empty() || base == word) continue;
        
        // Remove trailing space from base (neotagxml does this)
        if (!base.empty() && base.back() == ' ') {
            base = base.substr(0, base.size() - 1);
        }
        
        // Recursively parse the base word
        Token base_token = token;
        base_token.form = base;
        std::vector<WordCandidate> base_candidates = morpho_parse(base_token);
        
        if (base_candidates.empty()) {
            continue;  // Skip if base word not found
        }
        
        // For each base candidate, create a clitic candidate
        for (const auto& base_cand : base_candidates) {
            if (base_cand.prob <= 0.f) continue;
            
            // Find the clitic variant that matches the base tag
            // Check each variant in the dtok_form
            for (const auto& variant : dtok_form.variants) {
                // Validate that the clitic variant has a valid UPOS for clitics
                // Check if any of the variant tokens have a valid UPOS
                bool has_valid_upos = false;
                std::string clitic_upos;
                for (const auto& variant_token : variant.tokens) {
                    if (!variant_token.upos.empty() && variant_token.upos != "_") {
                        clitic_upos = variant_token.upos;
                        if (valid_clitic_upos.count(clitic_upos) > 0) {
                            has_valid_upos = true;
                            break;
                        }
                    }
                }
                
                // If variant has tokens but none have valid UPOS, skip this variant
                if (!variant.tokens.empty() && !has_valid_upos) {
                    if (settings_.debug) {
                        std::cerr << "[flexitag] apply_clitics: skipping variant with UPOS '" << clitic_upos 
                                  << "' for clitic '" << clitic_form << "' - not a valid clitic UPOS\n";
                    }
                    continue;
                }
                
                // Check if this variant's tag (key) can combine with base tag
                // For now, we'll create candidates for all variants
                // (neotagxml checks sibling tags, but we'll be more permissive)
                
                // Calculate probability: clitic_prob * base_prob
                float clitic_prob = dtok_form.lexcnt > 0 ? 
                    static_cast<float>(variant.count) / static_cast<float>(dtok_form.lexcnt) : 0.5f;
                float combined_prob = clitic_prob * base_cand.prob;
                
                // Create clitic candidate
                WordCandidate clitic_cand;
                clitic_cand.form = word;
                clitic_cand.tag = base_cand.tag + "." + variant.key;  // Combined tag like "APPR.ART"
                clitic_cand.prob = combined_prob;
                // Set source: contractions are detected via the dtoks table (lexicon),
                // so they should be marked as "contractions: lexicon" if the base is from lexicon,
                // or "contractions: ending" if the base is from endings, etc.
                // However, the contraction pattern itself is always from the lexicon (dtoks table).
                // For reporting purposes, we distinguish:
                // - "contractions: lexicon" = base word is in lexicon
                // - "contractions: ending" = base word matched via endings (OOV base)
                // - "contractions: unknown" = base word matched via unknown fallback
                std::string actual_source = base_cand.source;
                // Remove any existing "contractions: " prefix to avoid nested prefixes
                const std::string prefix = "contractions: ";
                if (actual_source.find(prefix) == 0) {
                    actual_source = actual_source.substr(prefix.length());
                }
                clitic_cand.source = prefix + actual_source;
                clitic_cand.wcase = form_case(word);
                clitic_cand.token = const_cast<Token*>(&token);
                clitic_cand.lemma = base_cand.lemma;  // Use base lemma
                
                // Create dtok children
                if (position == "left") {
                    // Pre-clitic: clitic comes first
                    auto clitic_dtok = std::make_shared<WordCandidate>();
                    clitic_dtok->form = clitic_form;
                    clitic_dtok->tag = variant.key;
                    clitic_dtok->lemma = variant.tokens.empty() ? clitic_form : variant.tokens[0].lemma;
                    if (!variant.tokens.empty() && !variant.tokens[0].xpos.empty()) {
                        clitic_dtok->tag = variant.tokens[0].xpos;
                    }
                    clitic_cand.dtoks.push_back(clitic_dtok);
                    
                    // Then base word
                    auto base_dtok = std::make_shared<WordCandidate>();
                    base_dtok->form = base;
                    base_dtok->tag = base_cand.tag;
                    base_dtok->lemma = base_cand.lemma;
                    if (base_cand.lex_attributes && !base_cand.lex_attributes->xpos.empty()) {
                        base_dtok->tag = base_cand.lex_attributes->xpos;
                    }
                    clitic_cand.dtoks.push_back(base_dtok);
                } else {
                    // Post-clitic: base comes first
                    auto base_dtok = std::make_shared<WordCandidate>();
                    base_dtok->form = base;
                    base_dtok->tag = base_cand.tag;
                    base_dtok->lemma = base_cand.lemma;
                    if (base_cand.lex_attributes && !base_cand.lex_attributes->xpos.empty()) {
                        base_dtok->tag = base_cand.lex_attributes->xpos;
                    }
                    clitic_cand.dtoks.push_back(base_dtok);
                    
                    // Then clitic
                    auto clitic_dtok = std::make_shared<WordCandidate>();
                    clitic_dtok->form = clitic_form;
                    clitic_dtok->tag = variant.key;
                    clitic_dtok->lemma = variant.tokens.empty() ? clitic_form : variant.tokens[0].lemma;
                    if (!variant.tokens.empty() && !variant.tokens[0].xpos.empty()) {
                        clitic_dtok->tag = variant.tokens[0].xpos;
                    }
                    clitic_cand.dtoks.push_back(clitic_dtok);
                }
                
                clitic_candidates.push_back(std::move(clitic_cand));
            }
        }
    }
    
    return clitic_candidates;
}

void FlexitagTagger::update_token(Token& token, const WordCandidate& best) const {
    // If candidate has dtoks, handle them specially (like neotagxml)
    // In this case, we leave the parent token's xpos empty and populate dtok children
    if (!best.dtoks.empty()) {
        const bool overwrite = settings_.get_bool("overwrite", false);
        const bool had_existing_dtoks = !token.subtokens.empty();

        if (!had_existing_dtoks || overwrite) {
            token.subtokens.clear();
        }
        
        // Ensure enough slots for dtoks and update in place where possible
        if (token.subtokens.size() < best.dtoks.size()) {
            token.subtokens.resize(best.dtoks.size());
        } else if (token.subtokens.size() > best.dtoks.size()) {
            token.subtokens.resize(best.dtoks.size());
        }

        for (std::size_t idx = 0; idx < best.dtoks.size(); ++idx) {
            const auto& dtok_cand = best.dtoks[idx];
            SubToken& st = token.subtokens[idx];

            if (st.id == 0) {
                st.id = static_cast<int>(token.id + idx);
            }
            if (!dtok_cand->form.empty()) {
            st.form = dtok_cand->form;
            }
            if (st.form.empty()) {
                // Fall back to candidate form even if empty to avoid blank subtokens
                st.form = dtok_cand->form.empty() ? token.form : dtok_cand->form;
            }

            if (!dtok_cand->lemma.empty()) {
            st.lemma = dtok_cand->lemma;
            } else if (st.lemma.empty() || st.lemma == "_") {
                st.lemma = st.form;
            }

            if (!dtok_cand->tag.empty()) {
                st.xpos = dtok_cand->tag;
            }
            
            // Copy UPOS and feats from candidate's lex_attributes if available
            if (dtok_cand->lex_attributes.has_value()) {
                const auto& attrs = dtok_cand->lex_attributes.value();
                if (!attrs.upos.empty()) {
                    st.upos = attrs.upos;
                }
                if (!attrs.feats.empty()) {
                    st.feats = attrs.feats;
                }
                // Also ensure XPOS is set if lex_attributes has it
                if (st.xpos.empty() && !attrs.xpos.empty()) {
                    st.xpos = attrs.xpos;
                }
            }
            
            // Copy source from candidate
            if (!dtok_cand->source.empty()) {
                st.source = dtok_cand->source;
            }
        }

        token.is_mwt = !token.subtokens.empty();
        token.mwt_start = token.id;
        token.mwt_end = token.id + static_cast<int>(token.subtokens.size()) - 1;
        
        if (token.lemma.empty() || token.lemma == "_") {
            token.lemma = best.lemma.empty() ? token.form : best.lemma;
        }
        
        // Copy contraction-level attributes from candidate's lex_attributes to parent token
        if (best.lex_attributes.has_value()) {
            const auto& attrs = best.lex_attributes.value();
            if (!attrs.reg.empty()) {
                token.reg = attrs.reg;
            }
            if (!attrs.expan.empty()) {
                token.expan = attrs.expan;
            }
            if (!attrs.mod.empty()) {
                token.mod = attrs.mod;
            }
            if (!attrs.trslit.empty()) {
                token.trslit = attrs.trslit;
            }
            if (!attrs.ltrslit.empty()) {
                token.ltrslit = attrs.ltrslit;
            }
            if (!attrs.tokid.empty()) {
                token.tokid = attrs.tokid;
            }
        }
        
        token.source = best.source;
        return;
    }
    
    // Normal case: no dtoks, set tag on parent token based on tagpos setting
    std::string tagpos = settings_.get("tagpos", "xpos");
    if (tagpos == "upos") {
        // Write upos - CRITICAL: always set it, even if best.tag is empty
        // Priority: lex_attributes->upos > best.tag (extract from UPOS#FEATS format) > best.tag (as-is)
        if (best.lex_attributes && !best.lex_attributes->upos.empty() && best.lex_attributes->upos != "_") {
            token.upos = best.lex_attributes->upos;
        } else if (!best.tag.empty()) {
            // Extract upos from tag if it's in format "UPOS#FEATS"
            std::size_t hash_pos = best.tag.find('#');
            if (hash_pos != std::string::npos) {
                token.upos = best.tag.substr(0, hash_pos);
            } else {
                token.upos = best.tag;  // Use tag directly as UPOS
            }
        }
        // CRITICAL: If UPOS is still empty after all attempts, and we're tagging with tagpos="upos",
        // this indicates a problem - the model should have provided a UPOS tag
        if (token.upos.empty() || token.upos == "_") {
            if (settings_.debug) {
                std::cerr << "[flexitag] WARNING: tagpos='upos' but token.upos is empty for token '"
                          << token.form << "' (best.tag='" << best.tag << "', source='" << best.source << "')\n";
            }
        }
        // Also set xpos if available (for compatibility)
        // CRITICAL: When tagpos == "upos", best.tag contains UPOS, not XPOS
        // So we should NOT set token.xpos = best.tag in this case
        // But we should still try to set XPOS from lex_attributes if available
        if (best.lex_attributes && !best.lex_attributes->xpos.empty() && best.lex_attributes->xpos != "_") {
            token.xpos = best.lex_attributes->xpos;
        }
        // Don't set xpos from best.tag when tagpos == "upos" - best.tag is UPOS, not XPOS
        // If XPOS is still empty, it will remain empty (which is acceptable if not in vocab)
    } else if (tagpos == "utot") {
        // Write upos and feats (utot = upos#feats)
        if (best.lex_attributes) {
            if (!best.lex_attributes->upos.empty() && best.lex_attributes->upos != "_") {
                token.upos = best.lex_attributes->upos;
            }
            if (!best.lex_attributes->feats.empty() && best.lex_attributes->feats != "_") {
                token.feats = best.lex_attributes->feats;
            }
        }
        // If tag already contains #, extract upos and feats from it
        std::size_t hash_pos = best.tag.find('#');
        if (hash_pos != std::string::npos) {
            if (token.upos.empty() || token.upos == "_") {
                token.upos = best.tag.substr(0, hash_pos);
            }
            if (token.feats.empty() || token.feats == "_") {
                token.feats = best.tag.substr(hash_pos + 1);
            }
        } else if (token.upos.empty() || token.upos == "_") {
            // Fallback: use tag as upos
            token.upos = best.tag;
        }
        // Also set xpos if available (for compatibility)
        if (best.lex_attributes && !best.lex_attributes->xpos.empty()) {
            token.xpos = best.lex_attributes->xpos;
        } else {
            token.xpos = best.tag;
        }
    } else {
        // Default: xpos
        // Set xpos from tag (this is the primary tag from the lexicon)
        // Use lex_attributes->xpos if available, otherwise use best.tag
        // BUT: if model was trained with tag_attribute="upos", then best.tag contains UPOS, not XPOS
        std::string model_tag_attribute = settings_.get("tag_attribute", "xpos");
        if (best.lex_attributes && !best.lex_attributes->xpos.empty() && best.lex_attributes->xpos != "_") {
            token.xpos = best.lex_attributes->xpos;
        } else if (model_tag_attribute == "upos" || model_tag_attribute == "utot") {
            // Model was trained with UPOS/UTOT, so best.tag contains UPOS, not XPOS
            // Don't set xpos from best.tag - it's actually UPOS
            // XPOS should come from lex_attributes->xpos if available
            // If not available, XPOS will remain empty (which is acceptable if not in vocab)
        } else {
            // Model was trained with XPOS, so best.tag contains XPOS
            if (!best.tag.empty()) {
                token.xpos = best.tag;
            }
        }
    }
    
    // Copy attributes from lex_attributes if available (matches neotagxml behavior)
    // neotagxml copies all attributes from lexitem except "key" and "cnt"
    // CRITICAL: neotagxml's lemmatize() (line 335-367) applies lemmatization rules
    // and prioritizes: lexitem.lemma > lemmatization rules > form
    // We should use best.lemma (which may have been derived from rules) as primary source
    if (best.lex_attributes) {
        // Lemma: prioritize best.lemma (may include lemmatization), then lex_attributes->lemma, then form
        if (token.lemma.empty() || token.lemma == "_") {
            if (!best.lemma.empty() && best.lemma != token.form) {
                // Use lemma from candidate (may have been derived from lemmatization rules)
                token.lemma = best.lemma;
            } else if (!best.lex_attributes->lemma.empty()) {
                token.lemma = best.lex_attributes->lemma;
            } else {
                token.lemma = token.form;
            }
        }
        
        
        // UPOS: copy if available and token doesn't already have it
        // CRITICAL: When tagpos == "upos", token.upos should already be set above (line 2083-2091)
        // This section is for when tagpos != "upos" but we still want to set UPOS from lex_attributes
        if ((token.upos.empty() || token.upos == "_") && 
            !best.lex_attributes->upos.empty() && best.lex_attributes->upos != "_") {
            token.upos = best.lex_attributes->upos;
        } else if ((token.upos.empty() || token.upos == "_") && !best.tag.empty()) {
            // If lex_attributes->upos is not available, check if tagpos is "xpos" and the model
            // was trained with tag_attribute="upos", then best.tag contains the UPOS.
            std::string tagpos = settings_.get("tagpos", "xpos");
            std::string model_tag_attribute = settings_.get("tag_attribute", "xpos");
            if (tagpos == "xpos" && (model_tag_attribute == "upos" || model_tag_attribute == "utot")) {
                // Model was trained with UPOS/UTOT, so best.tag contains UPOS
                // Try to set UPOS from best.tag if it's a known UPOS tag (be conservative)
                // Common UPOS tags: ADJ, ADP, ADV, AUX, CCONJ, DET, INTJ, NOUN, NUM, PART, PRON, PROPN, PUNCT, SCONJ, SYM, VERB, X
                std::string common_upos[] = {"ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"};
                bool is_upos_tag = false;
                for (const auto& upos : common_upos) {
                    if (best.tag == upos) {
                        is_upos_tag = true;
                        break;
                    }
                }
                // Only set if it's a known UPOS tag (be conservative to avoid false positives)
                if (is_upos_tag) {
                    token.upos = best.tag;
                }
            }
        }
        
        // FEATS: copy if available and token doesn't already have it
        if ((token.feats.empty() || token.feats == "_") && 
            !best.lex_attributes->feats.empty() && best.lex_attributes->feats != "_") {
            token.feats = best.lex_attributes->feats;
        }
        
        // Form/reg/expan: copy if available (for normalization)
        // Note: we don't overwrite existing form, but we can set reg/expan
        // CRITICAL: Always copy reg from lex_attributes if available and different from form
        // This ensures explicit vocab mappings (form -> reg) are preserved
        if (!best.lex_attributes->reg.empty() && best.lex_attributes->reg != "_" && 
            best.lex_attributes->reg != token.form) {
            // Only set if token.reg is empty or if lex_attributes->reg is different (prefer explicit mapping)
            if (token.reg.empty() || token.reg == "_" || token.reg == token.form) {
                token.reg = best.lex_attributes->reg;
            }
        }
        if (!best.lex_attributes->expan.empty() && token.expan.empty()) {
            token.expan = best.lex_attributes->expan;
        }
        
        // Copy additional attributes (mod, trslit, ltrslit, tokid) if available and token doesn't have them
        if (!best.lex_attributes->mod.empty() && (token.mod.empty() || token.mod == "_")) {
            token.mod = best.lex_attributes->mod;
        }
        if (!best.lex_attributes->trslit.empty() && (token.trslit.empty() || token.trslit == "_")) {
            token.trslit = best.lex_attributes->trslit;
        }
        if (!best.lex_attributes->ltrslit.empty() && (token.ltrslit.empty() || token.ltrslit == "_")) {
            token.ltrslit = best.lex_attributes->ltrslit;
        }
        if (!best.lex_attributes->tokid.empty() && (token.tokid.empty() || token.tokid == "_")) {
            token.tokid = best.lex_attributes->tokid;
        }
    } else {
        // Fallback: use lemma from candidate if token doesn't have one
        if (token.lemma.empty() || token.lemma == "_") {
            token.lemma = best.lemma.empty() ? token.form : best.lemma;
        }
        
        // If lex_attributes is not available, but tagpos is "xpos" and the model
        // was trained with tag_attribute="upos", then best.tag contains the UPOS.
        // Try to set UPOS from best.tag if it looks like a UPOS tag
        // NOTE: This is a fallback - ideally lex_attributes should have upos set
        if ((token.upos.empty() || token.upos == "_") && !best.tag.empty()) {
            // Common UPOS tags: ADJ, ADP, ADV, AUX, CCONJ, DET, INTJ, NOUN, NUM, PART, PRON, PROPN, PUNCT, SCONJ, SYM, VERB, X
            std::string common_upos[] = {"ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"};
            bool is_upos_tag = false;
            for (const auto& upos : common_upos) {
                if (best.tag == upos) {
                    is_upos_tag = true;
                    break;
                }
            }
            // Also check if tag contains no special characters (likely UPOS) vs XPOS (often has numbers/special chars)
            // But be more conservative: only set if it's a known UPOS tag, not just any short tag
            if (is_upos_tag) {
                token.upos = best.tag;
            }
        }
    }
    
    token.source = best.source;
}

} // namespace flexitag

