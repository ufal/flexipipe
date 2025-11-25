/*
 * C++ implementation of Viterbi tagging algorithm for FlexiPipe
 * Optimized for speed with efficient data structures and algorithms
 */

#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cctype>
#include <regex>
#include <memory>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/pytypes.h>

namespace py = pybind11;

// Constants
const double EPSILON = 1e-10;
const double SMOOTHING = 0.1;
const int MIN_SUFFIX_LEN = 2;
const int MAX_SUFFIX_LEN = 6;

// Helper: Convert Python string to lowercase
std::string to_lower(const std::string& s) {
    std::string result;
    result.reserve(s.size());
    for (char c : s) {
        result += std::tolower(static_cast<unsigned char>(c));
    }
    return result;
}

// Helper: Check if string is punctuation-only
bool is_punctuation_only(const std::string& word) {
    if (word.empty()) return false;
    bool has_alnum = false;
    bool has_non_space = false;
    for (char c : word) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            has_alnum = true;
            break;
        }
        if (!std::isspace(static_cast<unsigned char>(c))) {
            has_non_space = true;
        }
    }
    return !has_alnum && has_non_space;
}

// Helper: Check if string starts with prefix
bool starts_with(const std::string& s, const std::string& prefix) {
    return s.size() >= prefix.size() && 
           s.compare(0, prefix.size(), prefix) == 0;
}

// Helper: Check if string ends with suffix
bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() && 
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// Helper: Check if string ends with any of the suffixes
bool ends_with_any(const std::string& s, const std::vector<std::string>& suffixes) {
    for (const auto& suffix : suffixes) {
        if (ends_with(s, suffix)) return true;
    }
    return false;
}

// Extract tag probabilities from a vocabulary entry (Python dict or list)
std::unordered_map<std::string, double> extract_tag_probs(
    py::object entry_item,
    const std::string& tag_type,
    const std::vector<std::string>& all_tags
) {
    std::unordered_map<std::string, double> probs;
    
    if (py::isinstance<py::list>(entry_item)) {
        // Multiple analyses
        py::list entry_list = entry_item.cast<py::list>();
        std::vector<py::dict> analyses_with_tag;
        int total_count = 0;
        
        for (auto item : entry_list) {
                if (py::isinstance<py::dict>(item)) {
                    py::dict analysis = item.cast<py::dict>();
                    py::str tag_type_py = py::str(tag_type);
                    if (analysis.contains(tag_type_py)) {
                        py::object tag_obj = analysis[tag_type_py];
                        std::string tag = py::str(tag_obj).cast<std::string>();
                    if (tag != "_") {
                        analyses_with_tag.push_back(analysis);
                        int count = analysis.contains("count") ? 
                                   analysis["count"].cast<int>() : 1;
                        total_count += count;
                    }
                }
            }
        }
        
        if (!analyses_with_tag.empty()) {
            for (const auto& analysis : analyses_with_tag) {
                py::str tag_type_py = py::str(tag_type);
                std::string tag = py::str(analysis[tag_type_py]).cast<std::string>();
                int count = analysis.contains("count") ? 
                           analysis["count"].cast<int>() : 1;
                double prob = (count + 0.1) / (total_count + 0.1 * all_tags.size());
                probs[tag] += prob;
            }
        }
    } else if (py::isinstance<py::dict>(entry_item)) {
        // Single analysis
        py::dict analysis = entry_item.cast<py::dict>();
        py::str tag_type_py = py::str(tag_type);
        if (analysis.contains(tag_type_py)) {
            std::string tag = py::str(analysis[tag_type_py]).cast<std::string>();
            if (tag != "_") {
                int count = analysis.contains("count") ? 
                           analysis["count"].cast<int>() : 1;
                double prob = (count + 0.1) / (count + 0.1 * all_tags.size());
                probs[tag] = prob;
            }
        }
    }
    
    return probs;
}

// Get capitalization ratio for a tag
double get_capitalization_ratio(
    const std::string& tag,
    const std::string& tag_type,
    py::object capitalizable_tags
) {
    if (capitalizable_tags.is_none()) return 0.0;
    
    try {
        py::dict cap_tags = capitalizable_tags.cast<py::dict>();
        py::str tag_type_py = py::str(tag_type);
        if (!cap_tags.contains(tag_type_py)) return 0.0;
        
        py::dict tag_type_dict = cap_tags[tag_type_py].cast<py::dict>();
        py::str tag_py = py::str(tag);
        if (!tag_type_dict.contains(tag_py)) return 0.0;
        
        py::dict tag_stats = tag_type_dict[tag_py].cast<py::dict>();
        int capitalized = tag_stats.contains("capitalized") ? 
                         tag_stats["capitalized"].cast<int>() : 0;
        int lowercase = tag_stats.contains("lowercase") ? 
                       tag_stats["lowercase"].cast<int>() : 0;
        int total = capitalized + lowercase;
        
        if (total == 0) return 0.0;
        return static_cast<double>(capitalized) / total;
    } catch (...) {
        return 0.0;
    }
}

// Main Viterbi tagging function
std::vector<std::string> viterbi_tag_sentence_cpp(
    py::list sentence_py,
    py::dict vocab,
    py::dict transition_probs,
    py::object tag_type_obj,
    py::object capitalizable_tags
) {
    try {
        // Convert tag_type with error handling
        std::string tag_type;
        try {
            tag_type = py::str(tag_type_obj).cast<std::string>();
        } catch (...) {
            // Fallback to default
            tag_type = "upos";
        }
        
        // Convert Python list to C++ vector, handling encoding errors gracefully
        std::vector<std::string> sentence;
        sentence.reserve(sentence_py.size());
        
        for (auto item : sentence_py) {
            try {
                // Use py::str() which handles encoding more gracefully
                py::str item_str = py::str(item);
                std::string word = item_str.cast<std::string>();
                sentence.push_back(word);
            } catch (const std::exception& e) {
                // If conversion fails, try to get bytes representation
                try {
                    // Get the raw bytes if it's a bytes object
                    if (py::isinstance<py::bytes>(item)) {
                        py::bytes item_bytes = item.cast<py::bytes>();
                        std::string word = item_bytes.cast<std::string>();
                        sentence.push_back(word);
                    } else {
                        // Use repr() as fallback
                        py::str item_repr = py::repr(item);
                        std::string word = item_repr.cast<std::string>();
                        sentence.push_back(word);
                    }
                } catch (...) {
                    // Absolute last resort: use string representation
                    std::string word = "<?>";
                    sentence.push_back(word);
                }
            }
        }
        
        if (sentence.empty()) {
            return {};
        }
        
        // Get transition probabilities for this tag type
        py::dict trans_probs;
        py::dict start_probs;
        
        try {
            py::str tag_type_py = py::str(tag_type);
            if (transition_probs.contains(tag_type_py)) {
                trans_probs = transition_probs[tag_type_py].cast<py::dict>();
            }
            if (transition_probs.contains("start")) {
                start_probs = transition_probs["start"].cast<py::dict>();
            }
        } catch (const std::exception& e) {
            // If transition_probs access fails, use empty dicts
            trans_probs = py::dict();
            start_probs = py::dict();
        }
    
        // Collect all possible tags from vocab
        std::unordered_set<std::string> all_tags_set;
        
        // Use items() method to iterate more safely (avoids automatic key conversion)
        py::object items = vocab.attr("items")();
        for (auto item : items) {
            try {
                py::tuple item_tuple = item.cast<py::tuple>();
                if (item_tuple.size() != 2) continue;
                
                // Handle vocabulary key conversion with error handling
                py::object key_obj = item_tuple[0];
                std::string vocab_key;
                try {
                    vocab_key = py::str(key_obj).cast<std::string>();
                } catch (...) {
                    // Skip entries with invalid UTF-8 keys
                    continue;
                }
                
                py::object value_obj = item_tuple[1];
                py::object entry = value_obj;
                
                if (py::isinstance<py::list>(entry)) {
                    py::list entry_list = entry.cast<py::list>();
                    for (auto analysis_obj : entry_list) {
                        if (py::isinstance<py::dict>(analysis_obj)) {
                            py::dict analysis = analysis_obj.cast<py::dict>();
                            py::str tag_type_py = py::str(tag_type);
                            if (analysis.contains(tag_type_py)) {
                                std::string tag = py::str(analysis[tag_type_py]).cast<std::string>();
                                if (tag != "_") {
                                    all_tags_set.insert(tag);
                                }
                            }
                        }
                    }
                } else if (py::isinstance<py::dict>(entry)) {
                    py::dict analysis = entry.cast<py::dict>();
                    py::str tag_type_py = py::str(tag_type);
                    if (analysis.contains(tag_type_py)) {
                        std::string tag = py::str(analysis[tag_type_py]).cast<std::string>();
                        if (tag != "_") {
                            all_tags_set.insert(tag);
                        }
                    }
                }
            } catch (...) {
                // Skip entries with errors during processing
                continue;
            }
        }
    
    // Add tags from transitions
    for (auto item : trans_probs) {
        try {
            std::string prev_tag = py::str(item.first).cast<std::string>();
            all_tags_set.insert(prev_tag);
            if (py::isinstance<py::dict>(item.second)) {
                py::dict next_dict = item.second.cast<py::dict>();
                for (auto next_item : next_dict) {
                    try {
                        std::string curr_tag = py::str(next_item.first).cast<std::string>();
                        all_tags_set.insert(curr_tag);
                    } catch (...) {
                        // Skip tags with invalid UTF-8
                        continue;
                    }
                }
            }
        } catch (...) {
            // Skip transition entries with invalid UTF-8 keys
            continue;
        }
    }
    
    // Convert to sorted vector for consistent ordering
    std::vector<std::string> all_tags(all_tags_set.begin(), all_tags_set.end());
    std::sort(all_tags.begin(), all_tags.end());
    
    if (all_tags.empty()) {
        return std::vector<std::string>(sentence.size(), "_");
    }
    
    // Calculate tag frequencies from vocab (for OOV fallback)
    std::unordered_map<std::string, int> tag_frequencies;
    int total_tag_count = 0;
    
    // Use items() method to iterate more safely
    py::object items2 = vocab.attr("items")();
    for (auto item : items2) {
        try {
            py::tuple item_tuple = item.cast<py::tuple>();
            if (item_tuple.size() != 2) continue;
            
            py::object key_obj = item_tuple[0];
            std::string vocab_key;
            try {
                vocab_key = py::str(key_obj).cast<std::string>();
            } catch (...) {
                // Skip entries with invalid UTF-8 keys
                continue;
            }
            
            py::object value_obj = item_tuple[1];
            py::object entry = value_obj;
            
            if (py::isinstance<py::list>(entry)) {
                py::list entry_list = entry.cast<py::list>();
                for (auto analysis_obj : entry_list) {
                    if (py::isinstance<py::dict>(analysis_obj)) {
                        py::dict analysis = analysis_obj.cast<py::dict>();
                        py::str tag_type_py = py::str(tag_type);
                        if (analysis.contains(tag_type_py)) {
                            std::string tag = py::str(analysis[tag_type_py]).cast<std::string>();
                            if (tag != "_") {
                                int count = analysis.contains("count") ? 
                                           analysis["count"].cast<int>() : 1;
                                tag_frequencies[tag] += count;
                                total_tag_count += count;
                            }
                        }
                    }
                }
            } else if (py::isinstance<py::dict>(entry)) {
                py::dict analysis = entry.cast<py::dict>();
                py::str tag_type_py = py::str(tag_type);
                if (analysis.contains(tag_type_py)) {
                    std::string tag = py::str(analysis[tag_type_py]).cast<std::string>();
                    if (tag != "_") {
                        int count = analysis.contains("count") ? 
                                   analysis["count"].cast<int>() : 1;
                        tag_frequencies[tag] += count;
                        total_tag_count += count;
                    }
                }
            }
        } catch (...) {
            // Skip entries with errors during processing
            continue;
        }
    }
    
    // Create default emission probabilities
    std::unordered_map<std::string, double> default_emission;
    if (total_tag_count > 0) {
        for (const auto& tag : all_tags) {
            int freq = tag_frequencies.count(tag) ? tag_frequencies[tag] : 0;
            default_emission[tag] = (freq + SMOOTHING) / 
                                    (total_tag_count + SMOOTHING * all_tags.size());
        }
    } else {
        double uniform = 1.0 / all_tags.size();
        for (const auto& tag : all_tags) {
            default_emission[tag] = uniform;
        }
    }
    
    // Build suffix index for OOV words (suffix -> tag -> count)
    std::unordered_map<std::string, std::unordered_map<std::string, int>> suffix_index;
    
    // Use items() method to iterate more safely
    py::object items3 = vocab.attr("items")();
    for (auto item : items3) {
        try {
            py::tuple item_tuple = item.cast<py::tuple>();
            if (item_tuple.size() != 2) continue;
            
            py::object key_obj = item_tuple[0];
            std::string vocab_word;
            try {
                vocab_word = py::str(key_obj).cast<std::string>();
            } catch (...) {
                // Skip entries with invalid UTF-8 keys
                continue;
            }
            std::string word_lower = to_lower(vocab_word);
            
            if (word_lower.size() < MIN_SUFFIX_LEN) continue;
            
            // Extract tag counts from entry
            std::unordered_map<std::string, int> tag_counts;
            py::object value_obj = item_tuple[1];
            py::object entry = value_obj;
            
            if (py::isinstance<py::list>(entry)) {
                py::list entry_list = entry.cast<py::list>();
                for (auto analysis_obj : entry_list) {
                    if (py::isinstance<py::dict>(analysis_obj)) {
                        py::dict analysis = analysis_obj.cast<py::dict>();
                        py::str tag_type_py = py::str(tag_type);
                        if (analysis.contains(tag_type_py)) {
                            std::string tag = py::str(analysis[tag_type_py]).cast<std::string>();
                            if (tag != "_") {
                                int count = analysis.contains("count") ? 
                                           analysis["count"].cast<int>() : 1;
                                tag_counts[tag] += count;
                            }
                        }
                    }
                }
            } else if (py::isinstance<py::dict>(entry)) {
                py::dict analysis = entry.cast<py::dict>();
                py::str tag_type_py = py::str(tag_type);
                if (analysis.contains(tag_type_py)) {
                    std::string tag = py::str(analysis[tag_type_py]).cast<std::string>();
                    if (tag != "_") {
                        int count = analysis.contains("count") ? 
                                   analysis["count"].cast<int>() : 1;
                        tag_counts[tag] = count;
                    }
                }
            }
            
            // Add to suffix index (suffixes 2-6 chars)
            int max_suffix_len = std::min(MAX_SUFFIX_LEN, static_cast<int>(word_lower.size()));
            for (int suffix_len = MIN_SUFFIX_LEN; suffix_len <= max_suffix_len; ++suffix_len) {
                std::string suffix = word_lower.substr(word_lower.size() - suffix_len);
                for (const auto& tag_count : tag_counts) {
                    suffix_index[suffix][tag_count.first] += tag_count.second;
                }
            }
        } catch (...) {
            // Skip entries with errors during processing
            continue;
        }
    }
    
    // Build emission probabilities for each word
    std::vector<std::unordered_map<std::string, double>> emission;
    emission.reserve(sentence.size());
    
    for (size_t word_idx = 0; word_idx < sentence.size(); ++word_idx) {
        const std::string& word = sentence[word_idx];
        std::string word_lower = to_lower(word);
        
        // Check if word is capitalized and not sentence-initial
        bool is_capitalized = !word.empty() && 
                              std::isupper(static_cast<unsigned char>(word[0])) && 
                              word_lower != word;
        bool is_sentence_initial = (word_idx == 0);
        bool is_capitalized_non_initial = is_capitalized && !is_sentence_initial;
        
        // Get entry from vocab (try exact case first, then lowercase)
        // Use py::str() to properly convert C++ string to Python string for dict lookup
        py::object entry;
        py::object entry_lower;
        
        try {
            py::str word_py = py::str(word);
            if (vocab.contains(word_py)) {
                entry = vocab[word_py];
            }
        } catch (...) {
            // Skip if word conversion or lookup fails
            entry = py::none();
        }
        
        if (word_lower != word) {
            try {
                py::str word_lower_py = py::str(word_lower);
                if (vocab.contains(word_lower_py)) {
                    entry_lower = vocab[word_lower_py];
                }
            } catch (...) {
                // Skip if word_lower conversion or lookup fails
                entry_lower = py::none();
            }
        }
        
        if (entry.is_none() && !entry_lower.is_none()) {
            entry = entry_lower;
            entry_lower = py::none();
        }
        
        std::unordered_map<std::string, double> word_emission;
        
        if (!entry.is_none()) {
            // Extract tag probabilities from exact case entry
            word_emission = extract_tag_probs(entry, tag_type, all_tags);
            
            // If exact case entry doesn't have the tag_type, also check lowercase entry
            if (word_emission.empty() && !entry_lower.is_none()) {
                word_emission = extract_tag_probs(entry_lower, tag_type, all_tags);
            } else if (!entry_lower.is_none()) {
                // Both entries exist - combine probabilities (weighted by counts)
                auto lower_probs = extract_tag_probs(entry_lower, tag_type, all_tags);
                if (!lower_probs.empty()) {
                    // Calculate total counts for weighting
                    int exact_total = 0;
                    if (py::isinstance<py::list>(entry)) {
                        py::list entry_list = entry.cast<py::list>();
                        for (auto item : entry_list) {
                            if (py::isinstance<py::dict>(item)) {
                                exact_total += item.cast<py::dict>().contains("count") ? 
                                              item.cast<py::dict>()["count"].cast<int>() : 1;
                            }
                        }
                    } else if (py::isinstance<py::dict>(entry)) {
                        exact_total = entry.cast<py::dict>().contains("count") ? 
                                     entry.cast<py::dict>()["count"].cast<int>() : 1;
                    }
                    
                    int lower_total = 0;
                    if (py::isinstance<py::list>(entry_lower)) {
                        py::list entry_list = entry_lower.cast<py::list>();
                        for (auto item : entry_list) {
                            if (py::isinstance<py::dict>(item)) {
                                lower_total += item.cast<py::dict>().contains("count") ? 
                                              item.cast<py::dict>()["count"].cast<int>() : 1;
                            }
                        }
                    } else if (py::isinstance<py::dict>(entry_lower)) {
                        lower_total = entry_lower.cast<py::dict>().contains("count") ? 
                                     entry_lower.cast<py::dict>()["count"].cast<int>() : 1;
                    }
                    
                    int total_combined = exact_total + lower_total;
                    
                    // Normalize and combine
                    for (const auto& prob_pair : lower_probs) {
                        double weighted_prob = total_combined > 0 ? 
                                              prob_pair.second * (static_cast<double>(lower_total) / total_combined) : 
                                              prob_pair.second;
                        word_emission[prob_pair.first] += weighted_prob;
                    }
                }
            }
            
            // Adjust probabilities based on capitalization statistics
            if (is_capitalized_non_initial && !capitalizable_tags.is_none() && !word_emission.empty()) {
                std::unordered_map<std::string, double> tag_ratios;
                for (const auto& tag_prob : word_emission) {
                    tag_ratios[tag_prob.first] = get_capitalization_ratio(
                        tag_prob.first, tag_type, capitalizable_tags);
                }
                
                std::vector<std::string> high_ratio_tags;
                std::vector<std::string> low_ratio_tags;
                
                for (const auto& ratio_pair : tag_ratios) {
                    if (ratio_pair.second > 0.5) {
                        high_ratio_tags.push_back(ratio_pair.first);
                    } else {
                        low_ratio_tags.push_back(ratio_pair.first);
                    }
                }
                
                if (!high_ratio_tags.empty()) {
                    // Boost tags with high capitalization ratio
                    for (const auto& tag : high_ratio_tags) {
                        double ratio = tag_ratios[tag];
                        double boost_factor = 1.0 + (ratio * 9.0);
                        word_emission[tag] *= boost_factor;
                    }
                    
                    // Reduce tags with low capitalization ratio
                    for (const auto& tag : low_ratio_tags) {
                        double ratio = tag_ratios[tag];
                        double reduce_factor = 0.1 + (ratio * 0.4);
                        word_emission[tag] *= reduce_factor;
                    }
                } else if (!low_ratio_tags.empty() && !capitalizable_tags.is_none()) {
                    // Check for available high-ratio tags
                    std::vector<std::pair<std::string, double>> available_high_ratio;
                    try {
                        py::dict cap_tags = capitalizable_tags.cast<py::dict>();
                        py::str tag_type_py = py::str(tag_type);
                        if (cap_tags.contains(tag_type_py)) {
                            py::dict tag_type_dict = cap_tags[tag_type_py].cast<py::dict>();
                            for (auto item : tag_type_dict) {
                                std::string tag = py::str(item.first).cast<std::string>();
                                if (std::find(all_tags.begin(), all_tags.end(), tag) != all_tags.end()) {
                                    double ratio = get_capitalization_ratio(tag, tag_type, capitalizable_tags);
                                    if (ratio > 0.5) {
                                        available_high_ratio.push_back({tag, ratio});
                                    }
                                }
                            }
                        }
                    } catch (...) {
                        // Ignore errors
                    }
                    
                    if (!available_high_ratio.empty()) {
                        double total_ratio = 0.0;
                        for (const auto& pair : available_high_ratio) {
                            total_ratio += pair.second;
                        }
                        for (const auto& pair : available_high_ratio) {
                            word_emission[pair.first] = total_ratio > 0 ? 
                                                       0.8 * (pair.second / total_ratio) : 
                                                       0.8 / available_high_ratio.size();
                        }
                        for (const auto& tag : low_ratio_tags) {
                            word_emission[tag] *= 0.01;
                        }
                    }
                }
            }
            
            // Normalize to probabilities
            double total_prob = 0.0;
            for (const auto& prob_pair : word_emission) {
                total_prob += prob_pair.second;
            }
            
            if (total_prob > 0) {
                for (auto& prob_pair : word_emission) {
                    prob_pair.second /= total_prob;
                }
            } else {
                // Entry exists but has no valid tags - check punctuation
                if (is_punctuation_only(word)) {
                    if (tag_type == "upos") {
                        if (std::find(all_tags.begin(), all_tags.end(), "PUNCT") != all_tags.end()) {
                            word_emission["PUNCT"] = 1.0;
                        } else {
                            double uniform = 1.0 / all_tags.size();
                            for (const auto& tag : all_tags) {
                                word_emission[tag] = uniform;
                            }
                        }
                    } else {
                        // For XPOS, find punctuation tags
                        std::vector<std::string> punct_tags;
                        for (const auto& tag : all_tags) {
                            if ((starts_with(tag, "F") && tag.size() <= 4) ||
                                tag == "PUNCT" || tag == "Punct" || tag == "punct") {
                                punct_tags.push_back(tag);
                            }
                        }
                        if (!punct_tags.empty()) {
                            double prob = 1.0 / punct_tags.size();
                            for (const auto& tag : punct_tags) {
                                word_emission[tag] = prob;
                            }
                        } else {
                            double uniform = 1.0 / all_tags.size();
                            for (const auto& tag : all_tags) {
                                word_emission[tag] = uniform;
                            }
                        }
                    }
                } else {
                    double uniform = 1.0 / all_tags.size();
                    for (const auto& tag : all_tags) {
                        word_emission[tag] = uniform;
                    }
                }
            }
        } else {
            // OOV word - check if it's punctuation-only first
            if (is_punctuation_only(word)) {
                if (tag_type == "upos") {
                    if (std::find(all_tags.begin(), all_tags.end(), "PUNCT") != all_tags.end()) {
                        word_emission["PUNCT"] = 1.0;
                    } else {
                        word_emission = default_emission;
                    }
                } else {
                    // For XPOS, find punctuation tags
                    std::vector<std::string> punct_tags;
                    for (const auto& tag : all_tags) {
                        if ((starts_with(tag, "F") && tag.size() <= 4) ||
                            tag == "PUNCT" || tag == "Punct" || tag == "punct") {
                            punct_tags.push_back(tag);
                        }
                    }
                    if (!punct_tags.empty()) {
                        double prob = 1.0 / punct_tags.size();
                        for (const auto& tag : punct_tags) {
                            word_emission[tag] = prob;
                        }
                    } else {
                        word_emission = default_emission;
                    }
                }
            } else {
                // Not punctuation - try suffix-based matching using pre-built index
                std::unordered_map<std::string, int> suffix_tag_counts;
                int total_suffix_count = 0;
                
                // Try suffixes from 6 chars down to 2 chars
                for (int suffix_len = MAX_SUFFIX_LEN; suffix_len >= MIN_SUFFIX_LEN; --suffix_len) {
                    if (static_cast<int>(word_lower.size()) >= suffix_len) {
                        std::string suffix = word_lower.substr(word_lower.size() - suffix_len);
                        
                        if (suffix_index.count(suffix)) {
                            suffix_tag_counts = suffix_index[suffix];
                            for (const auto& count_pair : suffix_tag_counts) {
                                total_suffix_count += count_pair.second;
                            }
                            break;
                        }
                    }
                }
                
                // Convert suffix-based tag counts to probabilities
                if (!suffix_tag_counts.empty() && total_suffix_count > 0) {
                    for (const auto& count_pair : suffix_tag_counts) {
                        word_emission[count_pair.first] = 
                            static_cast<double>(count_pair.second) / total_suffix_count;
                    }
                }
                
                // Pattern-based heuristics for common endings
                if (word_emission.empty() || 
                    std::max_element(word_emission.begin(), word_emission.end(),
                                    [](const auto& a, const auto& b) { return a.second < b.second; })->second < 0.3) {
                    // Spanish/Portuguese verb endings
                    std::vector<std::string> past_participle_endings = {
                        "ado", "ada", "ados", "adas", "ido", "ida", "idos", "idas"
                    };
                    if (ends_with_any(word_lower, past_participle_endings)) {
                        for (const auto& tag : all_tags) {
                            if (starts_with(tag, "A") || starts_with(tag, "V")) {
                                word_emission[tag] += 0.5;
                            }
                        }
                    }
                    
                    // Preposition-like patterns
                    std::vector<std::string> prepositions = {
                        "de", "a", "en", "por", "para", "con", "sin", "sobre"
                    };
                    if (word_lower.size() <= 2 || 
                        std::find(prepositions.begin(), prepositions.end(), word_lower) != prepositions.end()) {
                        for (const auto& tag : all_tags) {
                            if (starts_with(tag, "SP")) {
                                word_emission[tag] += 0.3;
                            }
                        }
                    }
                }
                
                // Fallback: use tag frequency distribution
                if (word_emission.empty()) {
                    word_emission = default_emission;
                } else {
                    // Normalize
                    double total_prob = 0.0;
                    for (const auto& prob_pair : word_emission) {
                        total_prob += prob_pair.second;
                    }
                    if (total_prob > 0) {
                        for (auto& prob_pair : word_emission) {
                            prob_pair.second /= total_prob;
                        }
                    } else {
                        word_emission = default_emission;
                    }
                }
            }
        }
        
        // Convert to log space
        std::unordered_map<std::string, double> log_emission;
        for (const auto& prob_pair : word_emission) {
            log_emission[prob_pair.first] = std::log(std::max(prob_pair.second, EPSILON));
        }
        emission.push_back(log_emission);
    }
    
    // Viterbi algorithm
    size_t n = sentence.size();
    if (n == 0) {
        return {};
    }
    
    // Initialize DP tables
    std::vector<std::unordered_map<std::string, double>> viterbi(n);
    std::vector<std::unordered_map<std::string, std::string>> backpointer(n);
    
    // Pre-compute log transition probabilities
    std::unordered_map<std::string, std::unordered_map<std::string, double>> log_trans_probs;
    for (const auto& prev_tag : all_tags) {
        for (const auto& curr_tag : all_tags) {
            double trans_prob = EPSILON;
            py::str prev_tag_py = py::str(prev_tag);
            if (trans_probs.contains(prev_tag_py)) {
                py::dict prev_dict = trans_probs[prev_tag_py].cast<py::dict>();
                py::str curr_tag_py = py::str(curr_tag);
                if (prev_dict.contains(curr_tag_py)) {
                    trans_prob = prev_dict[curr_tag_py].cast<double>();
                }
            }
            log_trans_probs[prev_tag][curr_tag] = std::log(std::max(trans_prob, EPSILON));
        }
    }
    
    // Initialization: first word
    for (const auto& tag : all_tags) {
        double start_prob = 1.0 / all_tags.size();
        py::str tag_py = py::str(tag);
        if (start_probs.contains(tag_py)) {
            start_prob = start_probs[tag_py].cast<double>();
        }
        double start_log = std::log(std::max(start_prob, EPSILON));
        
        double emit_log = emission[0].count(tag) ? emission[0][tag] : std::log(EPSILON);
        
        viterbi[0][tag] = start_log + emit_log;
        backpointer[0][tag] = "";
    }
    
    // Recursion: fill DP table
    for (size_t t = 1; t < n; ++t) {
        for (const auto& curr_tag : all_tags) {
            double best_prob = -std::numeric_limits<double>::infinity();
            std::string best_prev_tag;
            
            for (const auto& prev_tag : all_tags) {
                double trans_log = log_trans_probs[prev_tag][curr_tag];
                double emit_log = emission[t].count(curr_tag) ? emission[t][curr_tag] : std::log(EPSILON);
                
                double prob = viterbi[t-1][prev_tag] + trans_log + emit_log;
                
                if (prob > best_prob) {
                    best_prob = prob;
                    best_prev_tag = prev_tag;
                }
            }
            
            viterbi[t][curr_tag] = best_prob;
            backpointer[t][curr_tag] = best_prev_tag;
        }
    }
    
    // Termination: find best final tag
    std::string best_final_tag = all_tags[0];
    double best_final_prob = viterbi[n-1][best_final_tag];
    for (const auto& tag : all_tags) {
        if (viterbi[n-1][tag] > best_final_prob) {
            best_final_prob = viterbi[n-1][tag];
            best_final_tag = tag;
        }
    }
    
    // Backtrack: reconstruct best path
    std::vector<std::string> path;
    path.reserve(n);
    path.push_back(best_final_tag);
    
    std::string current_tag = best_final_tag;
    for (int t = static_cast<int>(n) - 2; t >= 0; --t) {
        std::string prev_tag = backpointer[t+1][current_tag];
        path.insert(path.begin(), prev_tag);
        current_tag = prev_tag;
    }
    
        return path;
    } catch (const std::exception& e) {
        // If anything fails, return empty or fallback
        // Re-raise with more context
        throw std::runtime_error(std::string("Viterbi C++ error: ") + e.what());
    } catch (...) {
        throw std::runtime_error("Viterbi C++ error: Unknown exception");
    }
}

PYBIND11_MODULE(viterbi_cpp, m) {
    m.doc() = "C++ implementation of Viterbi tagging algorithm for FlexiPipe";
    
    m.def("viterbi_tag_sentence", &viterbi_tag_sentence_cpp,
          "Tag a sentence using Viterbi algorithm (C++ implementation)",
          py::arg("sentence"),
          py::arg("vocab"),
          py::arg("transition_probs"),
          py::arg("tag_type") = py::str("upos"),
          py::arg("capitalizable_tags") = py::none());
}

