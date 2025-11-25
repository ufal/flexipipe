#ifndef VITERBI_OPTIMIZED_H
#define VITERBI_OPTIMIZED_H

#include <string>
#include <vector>
#include <unordered_map>
#include "vocab_loader.h"

class ViterbiTagger {
public:
    // Tag a sentence using Viterbi algorithm
    static std::vector<std::string> tag_sentence(
        const std::vector<std::string>& sentence,
        const Vocab& vocab,
        const std::string& tag_type = "upos"
    );
    
private:
    // Build all tags from vocab
    static std::vector<std::string> build_all_tags(const Vocab& vocab, const std::string& tag_type);
    
    // Get emission probability for word-tag pair
    static double get_emission_prob(const std::string& word, const std::string& tag,
                                   const Vocab& vocab, const std::string& tag_type,
                                   const std::unordered_map<std::string, double>& default_emission);
    
    // Build default emission probabilities
    static std::unordered_map<std::string, double> build_default_emission(
        const Vocab& vocab,
        const std::vector<std::string>& all_tags,
        const std::string& tag_type
    );
    
    // Get transition probability
    static double get_transition_prob(const std::string& prev_tag, const std::string& curr_tag,
                                     const Vocab& vocab, const std::string& tag_type);
    
    // Get start probability
    static double get_start_prob(const std::string& tag, const Vocab& vocab);
};

#endif // VITERBI_OPTIMIZED_H

