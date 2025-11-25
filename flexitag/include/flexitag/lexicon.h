#pragma once

#include "types.h"

#include <unordered_map>
#include <unordered_set>
#include <string>
#include <vector>
#include <optional>

namespace flexitag {

struct LexiconEntry {
    std::string key;
    std::string lemma;
    std::string xpos;
    std::string upos;
    std::string feats;
    std::string reg;
    std::string expan;
    std::string mod;      // ModernForm (for contractions)
    std::string trslit;   // Translit (for contractions)
    std::string ltrslit;  // LTransLit (for contractions)
    std::string tokid;    // TokId (for contractions)
    int count = 0;
    std::vector<SubToken> dtoks;
};

struct LexiconToken {
    std::string tag;
    int count = 0;
    std::vector<LexiconEntry> entries;
};

struct LexiconItem {
    std::string form;
    std::vector<LexiconToken> tokens;
};

struct TransitionStat {
    std::string key;
    float count = 0.f;
};

struct TagCaseStat {
    std::string key;
    float count = 0.f;
};

struct TagStat {
    std::string key;
    float count = 0.f;
    std::vector<TagCaseStat> cases;
};

struct WordEnding {
    std::string tag;
    float prob = 0.f;
    std::string lemma;
    // CRITICAL: neotagxml stores lemmatization rules (like "*er#*") in endingProbs[wending][tag].lemmatizations
    // These rules are built during training using lemrulemake(word, lemma)
    std::unordered_map<std::string, int> lemmatizations;  // rule -> frequency
};

struct DtokVariant {
    std::string key;
    int count = 0;
    std::vector<SubToken> tokens;
};

struct DtokForm {
    std::string key;
    int lexcnt = 0;
    float clitprob = 0.f;
    std::string position;
    std::vector<DtokVariant> variants;
};

class Lexicon {
public:
    bool load(const std::string& params_file, bool merge = false);
    bool load_external(const std::string& lexicon_file, bool merge = false);

    void set_endlen(int endlen) { endlen_ = (endlen != 0) ? endlen : 6; }  // Allow negative values for prefixes
    int endlen() const { return endlen_; }
    void reindex_endings();  // Re-index all endings with the current endlen value

    const LexiconItem* find(const std::string& form) const;
    const LexiconItem* find_lower(const std::string& form) const;

    std::vector<std::string> ending_forms(const std::string& form, std::size_t max_len) const;

    const std::unordered_map<std::string, TransitionStat>& transitions() const { return transitions_; }
    const std::unordered_map<std::string, TagStat>& tag_stats() const { return tag_stats_; }
    const std::unordered_map<std::string, DtokForm>& dtoks() const { return dtoks_; }
    
    // Get type counts (number of unique vocabulary entries) per tag
    // This is used for unknown word tagging based on raw tag probability over types
    // Results are cached after first computation
    const std::unordered_map<std::string, int>& tag_type_counts() const;
    const std::unordered_map<std::string, std::unordered_map<std::string, WordEnding>>& endings() const { return endings_; }
    
    // Get default settings from the loaded model (if available)
    const std::unordered_map<std::string, std::string>& default_settings() const { return default_settings_; }
    
    // Check if lexicon has normalization data (entries with reg fields)
    // This is used to determine if enhanced normalization should be enabled
    bool has_normalizations() const;
    
    // Get all form->reg mappings for pattern extraction
    // Returns a map of form -> reg (only entries where reg exists and differs from form)
    std::unordered_map<std::string, std::string> get_normalization_mappings() const;

private:
    void reset();
    bool load_json(const std::string& params_file, bool merge = false);
    void merge_item(const LexiconItem& item);
    void merge_transitions(const std::unordered_map<std::string, TransitionStat>& new_transitions);
    void merge_tag_stats(const std::unordered_map<std::string, TagStat>& new_tag_stats);
    void merge_dtoks(const std::unordered_map<std::string, DtokForm>& new_dtoks);
    void merge_endings(const std::unordered_map<std::string, std::unordered_map<std::string, WordEnding>>& new_endings);

    std::unordered_map<std::string, LexiconItem> items_;
    std::unordered_map<std::string, LexiconItem> lower_items_;
    std::unordered_map<std::string, TransitionStat> transitions_;
    std::unordered_map<std::string, TagStat> tag_stats_;
    std::unordered_map<std::string, DtokForm> dtoks_;
    std::unordered_map<std::string, std::unordered_map<std::string, WordEnding>> endings_;  // ending -> tag -> WordEnding
    std::unordered_map<std::string, std::string> default_settings_;  // Default tagger settings from model
    int endlen_ = 6;

    void index_item(const LexiconItem& item);
    void index_endings(const LexiconItem& item);
    void compute_tag_type_counts() const;  // Lazy computation of type counts
    
    // Cached type counts (computed lazily)
    mutable std::unordered_map<std::string, int> tag_type_counts_cache_;
    mutable bool tag_type_counts_computed_ = false;
};

} // namespace flexitag

