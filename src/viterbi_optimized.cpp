#include "viterbi_optimized.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>
#include <cctype>

std::vector<std::string> ViterbiTagger::tag_sentence(
    const std::vector<std::string>& sentence,
    const Vocab& vocab,
    const std::string& tag_type) {
    
    if (sentence.empty()) {
        return {};
    }
    
    // Build all tags
    std::vector<std::string> all_tags = build_all_tags(vocab, tag_type);
    if (all_tags.empty()) {
        return std::vector<std::string>(sentence.size(), "_");
    }
    
    // Build default emission probabilities
    std::unordered_map<std::string, double> default_emission = build_default_emission(vocab, all_tags, tag_type);
    
    // Viterbi algorithm: dp[i][tag] = best probability ending at position i with tag
    // We use log probabilities to avoid underflow
    std::vector<std::unordered_map<std::string, double>> dp(sentence.size());
    std::vector<std::unordered_map<std::string, std::string>> backpointer(sentence.size());
    
    // Initialize first word
    std::string first_word = sentence[0];
    std::string first_word_lower = first_word;
    std::transform(first_word_lower.begin(), first_word_lower.end(), first_word_lower.begin(), ::tolower);
    
    const std::vector<VocabAnalysis>* analyses = vocab.get(first_word);
    if (!analyses) {
        analyses = vocab.get(first_word_lower);
    }
    
    // Get emission probabilities for first word
    std::unordered_map<std::string, double> first_emission;
    if (analyses && !analyses->empty()) {
        int total_count = 0;
        std::unordered_map<std::string, int> tag_counts;
        
        for (const auto& analysis : *analyses) {
            std::string tag;
            if (tag_type == "upos") {
                tag = analysis.upos;
            } else {
                tag = analysis.xpos;
            }
            
            if (!tag.empty() && tag != "_") {
                int count = analysis.count > 0 ? analysis.count : 1;
                tag_counts[tag] += count;
                total_count += count;
            }
        }
        
        // Convert counts to probabilities
        double smoothing = 0.1;
        for (const auto& tag_count : tag_counts) {
            double prob = (tag_count.second + smoothing) / (total_count + smoothing * all_tags.size());
            first_emission[tag_count.first] = prob;
        }
    }
    
    // If no emission found, use default
    if (first_emission.empty()) {
        first_emission = default_emission;
    }
    
    // Initialize dp[0] with start probabilities
    for (const auto& tag : all_tags) {
        double start_prob = get_start_prob(tag, vocab);
        double emission = first_emission.count(tag) > 0 ? first_emission[tag] : default_emission[tag];
        dp[0][tag] = std::log(start_prob) + std::log(emission);
    }
    
    // Process remaining words
    for (size_t i = 1; i < sentence.size(); i++) {
        std::string word = sentence[i];
        std::string word_lower = word;
        std::transform(word_lower.begin(), word_lower.end(), word_lower.begin(), ::tolower);
        
        // Get emission probabilities for this word
        std::unordered_map<std::string, double> emission;
        analyses = vocab.get(word);
        if (!analyses) {
            analyses = vocab.get(word_lower);
        }
        
        if (analyses && !analyses->empty()) {
            int total_count = 0;
            std::unordered_map<std::string, int> tag_counts;
            
            for (const auto& analysis : *analyses) {
                std::string tag;
                if (tag_type == "upos") {
                    tag = analysis.upos;
                } else {
                    tag = analysis.xpos;
                }
                
                if (!tag.empty() && tag != "_") {
                    int count = analysis.count > 0 ? analysis.count : 1;
                    tag_counts[tag] += count;
                    total_count += count;
                }
            }
            
            // Convert counts to probabilities
            double smoothing = 0.1;
            for (const auto& tag_count : tag_counts) {
                double prob = (tag_count.second + smoothing) / (total_count + smoothing * all_tags.size());
                emission[tag_count.first] = prob;
            }
        }
        
        // If no emission found, use default
        if (emission.empty()) {
            emission = default_emission;
        }
        
        // For each possible tag at position i
        for (const auto& curr_tag : all_tags) {
            double best_prob = -std::numeric_limits<double>::infinity();
            std::string best_prev_tag;
            
            double emission_prob = emission.count(curr_tag) > 0 ? emission[curr_tag] : default_emission[curr_tag];
            
            // Find best previous tag
            for (const auto& prev_tag : all_tags) {
                if (dp[i-1].count(prev_tag) == 0) {
                    continue;
                }
                
                double transition_prob = get_transition_prob(prev_tag, curr_tag, vocab, tag_type);
                double prob = dp[i-1][prev_tag] + std::log(transition_prob) + std::log(emission_prob);
                
                if (prob > best_prob) {
                    best_prob = prob;
                    best_prev_tag = prev_tag;
                }
            }
            
            if (best_prev_tag.empty()) {
                // No valid previous tag - use uniform transition
                best_prob = std::log(1.0 / all_tags.size()) + std::log(emission_prob);
                best_prev_tag = all_tags[0];
            }
            
            dp[i][curr_tag] = best_prob;
            backpointer[i][curr_tag] = best_prev_tag;
        }
    }
    
    // Backtrack to find best path
    std::vector<std::string> tags(sentence.size());
    
    // Find best tag for last word
    double best_final_prob = -std::numeric_limits<double>::infinity();
    std::string best_final_tag;
    for (const auto& tag : all_tags) {
        if (dp[sentence.size()-1].count(tag) > 0 && dp[sentence.size()-1][tag] > best_final_prob) {
            best_final_prob = dp[sentence.size()-1][tag];
            best_final_tag = tag;
        }
    }
    
    if (best_final_tag.empty()) {
        best_final_tag = all_tags[0];
    }
    
    tags[sentence.size()-1] = best_final_tag;
    
    // Backtrack
    for (int i = sentence.size() - 2; i >= 0; i--) {
        if (backpointer[i+1].count(tags[i+1]) > 0) {
            tags[i] = backpointer[i+1][tags[i+1]];
        } else {
            tags[i] = all_tags[0];
        }
    }
    
    return tags;
}

std::vector<std::string> ViterbiTagger::build_all_tags(const Vocab& vocab, const std::string& tag_type) {
    std::unordered_set<std::string> tag_set;
    
    // Collect tags from vocab
    for (const auto& entry : vocab.entries) {
        for (const auto& analysis : entry.second) {
            std::string tag;
            if (tag_type == "upos") {
                tag = analysis.upos;
            } else {
                tag = analysis.xpos;
            }
            
            if (!tag.empty() && tag != "_") {
                tag_set.insert(tag);
            }
        }
    }
    
    // Collect tags from transitions
    if (vocab.transitions.count(tag_type) > 0) {
        for (const auto& prev_entry : vocab.transitions.at(tag_type)) {
            tag_set.insert(prev_entry.first);
            for (const auto& curr_entry : prev_entry.second) {
                tag_set.insert(curr_entry.first);
            }
        }
    }
    
    std::vector<std::string> tags(tag_set.begin(), tag_set.end());
    std::sort(tags.begin(), tags.end());
    return tags;
}

double ViterbiTagger::get_emission_prob(const std::string& word, const std::string& tag,
                                       const Vocab& vocab, const std::string& tag_type,
                                       const std::unordered_map<std::string, double>& default_emission) {
    const std::vector<VocabAnalysis>* analyses = vocab.get(word);
    if (!analyses) {
        std::string word_lower = word;
        std::transform(word_lower.begin(), word_lower.end(), word_lower.begin(), ::tolower);
        analyses = vocab.get(word_lower);
    }
    
    if (analyses && !analyses->empty()) {
        int total_count = 0;
        int tag_count = 0;
        
        for (const auto& analysis : *analyses) {
            std::string analysis_tag;
            if (tag_type == "upos") {
                analysis_tag = analysis.upos;
            } else {
                analysis_tag = analysis.xpos;
            }
            
            int count = analysis.count > 0 ? analysis.count : 1;
            total_count += count;
            
            if (analysis_tag == tag) {
                tag_count += count;
            }
        }
        
        if (total_count > 0) {
            double smoothing = 0.1;
            return (tag_count + smoothing) / (total_count + smoothing * default_emission.size());
        }
    }
    
    return default_emission.count(tag) > 0 ? default_emission.at(tag) : (1.0 / default_emission.size());
}

std::unordered_map<std::string, double> ViterbiTagger::build_default_emission(
    const Vocab& vocab,
    const std::vector<std::string>& all_tags,
    const std::string& tag_type) {
    
    std::unordered_map<std::string, double> default_emission;
    
    // Count tag frequencies
    std::unordered_map<std::string, int> tag_frequencies;
    int total_count = 0;
    
    for (const auto& entry : vocab.entries) {
        for (const auto& analysis : entry.second) {
            std::string tag;
            if (tag_type == "upos") {
                tag = analysis.upos;
            } else {
                tag = analysis.xpos;
            }
            
            if (!tag.empty() && tag != "_") {
                int count = analysis.count > 0 ? analysis.count : 1;
                tag_frequencies[tag] += count;
                total_count += count;
            }
        }
    }
    
    // Build default emission with smoothing
    double smoothing = 0.1;
    if (total_count > 0) {
        for (const auto& tag : all_tags) {
            int freq = tag_frequencies.count(tag) > 0 ? tag_frequencies[tag] : 0;
            default_emission[tag] = (freq + smoothing) / (total_count + smoothing * all_tags.size());
        }
    } else {
        // Uniform distribution
        double uniform = 1.0 / all_tags.size();
        for (const auto& tag : all_tags) {
            default_emission[tag] = uniform;
        }
    }
    
    return default_emission;
}

double ViterbiTagger::get_transition_prob(const std::string& prev_tag, const std::string& curr_tag,
                                         const Vocab& vocab, const std::string& tag_type) {
    if (vocab.transitions.count(tag_type) > 0) {
        const auto& tag_transitions = vocab.transitions.at(tag_type);
        if (tag_transitions.count(prev_tag) > 0) {
            const auto& prev_transitions = tag_transitions.at(prev_tag);
            if (prev_transitions.count(curr_tag) > 0) {
                double prob = prev_transitions.at(curr_tag);
                // Ensure probability is valid
                if (prob > 0.0 && prob <= 1.0) {
                    return prob;
                }
            }
        }
    }
    
    // Default: very small uniform transition (smoothing)
    return 1e-6;
}

double ViterbiTagger::get_start_prob(const std::string& tag, const Vocab& vocab) {
    if (vocab.start_probs.count(tag) > 0) {
        double prob = vocab.start_probs.at(tag);
        // Ensure probability is valid
        if (prob > 0.0 && prob <= 1.0) {
            return prob;
        }
    }
    
    // Default: very small uniform start probability (smoothing)
    return 1e-6;
}

