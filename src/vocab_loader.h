#ifndef VOCAB_LOADER_H
#define VOCAB_LOADER_H

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <algorithm>
#include <cctype>
#include "rapidjson/document.h"
#include "rapidjson/error/en.h"

struct VocabAnalysis {
    std::string upos;
    std::string xpos;
    std::string lemma;
    std::string feats;
    std::string reg;  // normalized form
    std::string expan;  // expansion for abbreviations
    int count = 0;
    std::vector<std::string> parts;  // for contractions
};

struct Vocab {
    // Vocabulary entries: form -> list of analyses
    std::unordered_map<std::string, std::vector<VocabAnalysis>> entries;
    
    // Transition probabilities: tag_type -> prev_tag -> curr_tag -> prob
    std::unordered_map<std::string, 
        std::unordered_map<std::string, 
            std::unordered_map<std::string, double>>> transitions;
    
    // Start probabilities: tag -> prob
    std::unordered_map<std::string, double> start_probs;
    
    // Dependency transitions: head_pos|dep_pos|deprel -> count
    std::unordered_map<std::string, int> dep_transitions;
    
    // Capitalizable tags: tag_type -> tag -> {capitalized, lowercase}
    std::unordered_map<std::string,
        std::unordered_map<std::string,
            std::pair<int, int>>> capitalizable_tags;
    
    // Metadata
    std::unordered_map<std::string, std::string> metadata;
    
    // Get analyses for a form (tries exact case, then lowercase)
    const std::vector<VocabAnalysis>* get(const std::string& form) const {
        auto it = entries.find(form);
        if (it != entries.end()) {
            return &it->second;
        }
        // Try lowercase
        std::string lower = form;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        it = entries.find(lower);
        if (it != entries.end()) {
            return &it->second;
        }
        return nullptr;
    }
};

class VocabLoader {
public:
    bool load(const std::string& vocab_file, Vocab& vocab);
    
private:
    bool parse_vocab_object(const rapidjson::Value& vocab_obj, Vocab& vocab);
    bool parse_analysis(const rapidjson::Value& analysis_obj, VocabAnalysis& analysis);
    bool parse_transitions(const rapidjson::Value& transitions_obj, Vocab& vocab);
    bool parse_metadata(const rapidjson::Value& metadata_obj, Vocab& vocab);
    std::string to_lower(const std::string& s) const;
};

#endif // VOCAB_LOADER_H
