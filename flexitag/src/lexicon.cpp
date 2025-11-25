#include "flexitag/lexicon.h"
#include "flexitag/unicode_utils.h"

#include <pugixml.hpp>

#include <algorithm>
#include <cctype>
#include <fstream>

#include <nlohmann/json.hpp>

namespace flexitag {

namespace {
// Using ICU-based Unicode utilities instead of manual UTF-8 handling
using flexitag::unicode::char_count;
using flexitag::unicode::suffix;
using flexitag::unicode::prefix;
using flexitag::unicode::substr_from_char;

std::string to_lower(const std::string& input) {
    std::string copy = input;
    std::transform(copy.begin(), copy.end(), copy.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return copy;
}

LexiconEntry parse_entry(const pugi::xml_node& node) {
    LexiconEntry entry;
    entry.key = node.attribute("key").value();
    entry.count = node.attribute("cnt") ? std::atoi(node.attribute("cnt").value()) : 0;
    if (node.attribute("lemma")) entry.lemma = node.attribute("lemma").value();
    if (node.attribute("xpos")) entry.xpos = node.attribute("xpos").value();
    if (node.attribute("upos")) entry.upos = node.attribute("upos").value();
    if (node.attribute("feats")) entry.feats = node.attribute("feats").value();
    if (node.attribute("reg")) entry.reg = node.attribute("reg").value();
    if (node.attribute("expan")) entry.expan = node.attribute("expan").value();

    for (auto dtok : node.children("dtok")) {
        SubToken st;
        st.form = dtok.attribute("form").value();
        if (dtok.attribute("lemma")) st.lemma = dtok.attribute("lemma").value();
        if (dtok.attribute("xpos")) st.xpos = dtok.attribute("xpos").value();
        if (dtok.attribute("upos")) st.upos = dtok.attribute("upos").value();
        if (dtok.attribute("feats")) st.feats = dtok.attribute("feats").value();
        entry.dtoks.push_back(std::move(st));
    }
    return entry;
}

LexiconToken parse_token(const pugi::xml_node& node) {
    LexiconToken token;
    token.tag = node.attribute("key").value();
    token.count = node.attribute("cnt") ? std::atoi(node.attribute("cnt").value()) : 0;
    for (auto child : node.children("tok")) {
        token.entries.push_back(parse_entry(child));
    }
    return token;
}

LexiconItem parse_item(const pugi::xml_node& node) {
    LexiconItem item;
    item.form = node.attribute("key").value();
    for (auto tok : node.children("tok")) {
        LexiconToken token;
        token.tag = tok.attribute("key").value();
        token.count = tok.attribute("cnt") ? std::atoi(tok.attribute("cnt").value()) : 0;
        token.entries.push_back(parse_entry(tok));
        item.tokens.push_back(std::move(token));
    }
    return item;
}

TransitionStat parse_transition(const pugi::xml_node& node) {
    TransitionStat ts;
    ts.key = node.attribute("key").value();
    ts.count = node.attribute("cnt") ? std::atof(node.attribute("cnt").value()) : 0.f;
    return ts;
}

TagStat parse_tagstat(const pugi::xml_node& node) {
    TagStat ts;
    ts.key = node.attribute("key").value();
    ts.count = node.attribute("cnt") ? std::atof(node.attribute("cnt").value()) : 0.f;
    for (auto cs : node.children("case")) {
        TagCaseStat cstat;
        cstat.key = cs.attribute("key").value();
        cstat.count = cs.attribute("cnt") ? std::atof(cs.attribute("cnt").value()) : 0.f;
        ts.cases.push_back(std::move(cstat));
    }
    return ts;
}

DtokVariant parse_dtok_variant(const pugi::xml_node& node) {
    DtokVariant variant;
    variant.key = node.attribute("key").value();
    variant.count = node.attribute("cnt") ? std::atoi(node.attribute("cnt").value()) : 0;
    for (auto dtok : node.children("dtok")) {
        SubToken st;
        st.form = dtok.attribute("form").value();
        if (dtok.attribute("lemma")) st.lemma = dtok.attribute("lemma").value();
        if (dtok.attribute("xpos")) st.xpos = dtok.attribute("xpos").value();
        if (dtok.attribute("upos")) st.upos = dtok.attribute("upos").value();
        if (dtok.attribute("feats")) st.feats = dtok.attribute("feats").value();
        variant.tokens.push_back(std::move(st));
    }
    return variant;
}

DtokForm parse_dtok_form(const pugi::xml_node& node) {
    DtokForm form;
    form.key = node.attribute("key").value();
    form.lexcnt = node.attribute("lexcnt") ? std::atoi(node.attribute("lexcnt").value()) : 0;
    if (node.attribute("clitprob")) form.clitprob = std::atof(node.attribute("clitprob").value());
    if (node.attribute("position")) form.position = node.attribute("position").value();
    for (auto item : node.children("item")) {
        form.variants.push_back(parse_dtok_variant(item));
    }
    return form;
}

// CRITICAL: Build lemmatization rule from word form to lemma (neotagxml line 643-694)
// Creates rules like "*er#*" meaning: remove "er" from end, result is lemma
// Example: "Geißler" -> "Geißl" creates rule like "**ler#**l" or "*er#*"
std::string make_lemmatization_rule(const std::string& word, const std::string& lemma) {
    if (word.empty() || lemma.empty()) {
        return "";
    }
    // CRITICAL: neotagxml builds rule "*#*" when word == lemma (no transformation)
    // This rule is important for words that should stay unchanged
    if (word == lemma) {
        return "*#*";  // Rule meaning "no change"
    }
    
    // Match as many characters between lemma and form as possible (neotagxml line 651-669)
    int wrdidx = 0;
    int lemidx = 0;
    std::string wrdroot = word;
    std::string lemroot = lemma;
    
    // Find each char of lemma in turn in the word
    while (lemidx < static_cast<int>(lemroot.size())) {
        // Walk through the form until we find a match for the lemchar
        while (wrdidx < static_cast<int>(wrdroot.size()) && wrdroot[wrdidx] != lemroot[lemidx]) {
            wrdidx++;
        }
        if (wrdidx < static_cast<int>(wrdroot.size())) {  // Match found
            wrdroot[wrdidx] = '*';
            lemroot[lemidx] = '*';
        } else {  // No match found - rewind to just after the last * in the form and skip a lemchar
            while (wrdidx > 0 && wrdroot[wrdidx] != '*') {
                wrdidx--;
            }
            wrdidx++;
        }
        lemidx++;
    }
    
    if (wrdroot.empty() || lemroot.empty()) {
        return "";
    }
    
    std::string lemrule = wrdroot + '#' + lemroot;
    
    // CRITICAL: Sanitize UTF-8 - rule building works on bytes and can create invalid UTF-8
    // We need to ensure the rule itself is valid UTF-8, but since rules only contain ASCII
    // characters (*, #) and the matched characters, we should be OK. However, if the original
    // word/lemma had invalid UTF-8, it could propagate. Let's validate.
    // Actually, since rules are built from valid UTF-8 words/lemmas, and we only replace
    // matching characters with '*', the rule should remain valid UTF-8. But to be safe,
    // we'll validate the result.
    
    // Remove multiple * in the rule (neotagxml line 679-690)
    int wrdidx2 = 0;
    bool star = false;
    while (wrdidx2 < static_cast<int>(lemrule.size())) {
        if (lemrule[wrdidx2] == '*') {
            if (star) {
                lemrule.erase(wrdidx2, 1);
            } else {
                wrdidx2++;
            }
            star = true;
        } else {
            wrdidx2++;
            star = false;
        }
    }
    
    return lemrule;
}

} // namespace

bool Lexicon::load(const std::string& params_file, bool merge) {
    std::string lower = to_lower(params_file);
    auto dot = lower.find_last_of('.');
    if (dot != std::string::npos) {
        std::string ext = lower.substr(dot + 1);
        if (ext == "json") {
            return load_json(params_file, merge);
        }
    }

    pugi::xml_document doc;
    if (!doc.load_file(params_file.c_str())) {
        return false;
    }

    if (!merge) {
        reset();
    }

    for (auto node : doc.child("neotag").child("lexicon").children("item")) {
        LexiconItem item = parse_item(node);
        if (merge) {
            merge_item(item);
        } else {
        index_item(item);
        }
    }

    auto transitions_node = doc.child("neotag").child("transitions");
    if (transitions_node) {
        std::unordered_map<std::string, TransitionStat> new_transitions;
        for (auto node : transitions_node.children("item")) {
            TransitionStat ts = parse_transition(node);
            new_transitions[ts.key] = ts;
        }
        if (merge) {
            merge_transitions(new_transitions);
        } else {
            transitions_ = std::move(new_transitions);
        }
    }

    auto tags_node = doc.child("neotag").child("tags");
    if (tags_node) {
        std::unordered_map<std::string, TagStat> new_tag_stats;
        for (auto node : tags_node.children("item")) {
            TagStat ts = parse_tagstat(node);
            new_tag_stats[ts.key] = ts;
        }
        if (merge) {
            merge_tag_stats(new_tag_stats);
        } else {
            tag_stats_ = std::move(new_tag_stats);
        }
    }

    auto dtoks_node = doc.child("neotag").child("dtoks");
    if (dtoks_node) {
        std::unordered_map<std::string, DtokForm> new_dtoks;
        for (auto node : dtoks_node.children("item")) {
            DtokForm form = parse_dtok_form(node);
            new_dtoks[form.key] = form;
        }
        if (merge) {
            merge_dtoks(new_dtoks);
        } else {
            dtoks_ = std::move(new_dtoks);
        }
    }

    // Load word endings (neotagxml line 554)
    auto endings_node = doc.child("neotag").child("endings");
    if (endings_node) {
        std::unordered_map<std::string, std::unordered_map<std::string, WordEnding>> new_endings;
        for (auto ending_item : endings_node.children("item")) {
            std::string ending_key = ending_item.attribute("key").value();
            for (auto tag_item : ending_item.children("item")) {
                std::string tag = tag_item.attribute("key").value();
                WordEnding we;
                we.tag = tag;
                we.prob = tag_item.attribute("cnt") ? std::atof(tag_item.attribute("cnt").value()) : 0.f;
                if (tag_item.attribute("lemma")) we.lemma = tag_item.attribute("lemma").value();
                if (tag_item.attribute("lemma") && we.lemma.empty()) {
                    we.lemma = tag_item.attribute("lemma").value();
                }
                new_endings[ending_key][tag] = we;
            }
        }
        if (merge) {
            merge_endings(new_endings);
        } else {
            endings_ = std::move(new_endings);
        }
    }

    return true;
}

void Lexicon::reset() {
    items_.clear();
    lower_items_.clear();
    transitions_.clear();
    tag_stats_.clear();
    dtoks_.clear();
    endings_.clear();
    default_settings_.clear();
    tag_type_counts_cache_.clear();
    tag_type_counts_computed_ = false;
}

bool Lexicon::load_external(const std::string& lexicon_file, bool merge) {
    if (lexicon_file.empty()) {
        return true;
    }
    pugi::xml_document doc;
    if (!doc.load_file(lexicon_file.c_str())) {
        return false;
    }
    for (auto node : doc.child("neotag").child("lexicon").children("item")) {
        LexiconItem item = parse_item(node);
        if (merge) {
            merge_item(item);
        } else {
        index_item(item);
        }
    }
    return true;
}

const LexiconItem* Lexicon::find(const std::string& form) const {
    auto it = items_.find(form);
    if (it == items_.end()) {
        return nullptr;
    }
    return &it->second;
}

const LexiconItem* Lexicon::find_lower(const std::string& form) const {
    auto it = lower_items_.find(to_lower(form));
    if (it == lower_items_.end()) {
        return nullptr;
    }
    return &it->second;
}

std::vector<std::string> Lexicon::ending_forms(const std::string& form, std::size_t max_len) const {
    std::vector<std::string> endings;
    // Use UTF-8 character count, not byte count
    size_t form_char_count = char_count(form);
    if (form_char_count < max_len) {
        max_len = form_char_count;
    }
    // If endlen_ is negative, use prefixes (for front-inflecting languages)
    const bool use_prefix = (endlen_ < 0);
    for (std::size_t len = 1; len <= max_len; ++len) {
        endings.push_back(use_prefix ? prefix(form, len) : suffix(form, len));
    }
    return endings;
}

void Lexicon::index_item(const LexiconItem& item) {
    items_[item.form] = item;
    lower_items_[to_lower(item.form)] = item;
    index_endings(item);
}

void Lexicon::reindex_endings() {
    // Clear existing endings
    endings_.clear();
    // Re-index all items with the current endlen value
    for (const auto& [form, item] : items_) {
        index_endings(item);
    }
}

void Lexicon::index_endings(const LexiconItem& item) {
    const std::string& word = item.form;
    if (word.empty()) {
        return;
    }

    // Use UTF-8 character count, not byte count, for endlen
    size_t word_char_count = char_count(word);
    // If endlen_ is negative, use prefixes (for front-inflecting languages)
    // If positive, use suffixes (for back-inflecting languages)
    const int abs_endlen = std::abs(endlen_);
    const int max_len_utf8 = std::min(static_cast<int>(word_char_count), abs_endlen);
    if (max_len_utf8 <= 0) {
        return;
    }
    const bool use_prefix = (endlen_ < 0);

    for (const auto& token : item.tokens) {
        if (token.tag.empty()) {
            continue;
        }

        if (token.entries.empty()) {
            // No explicit entries (should be rare); fall back to the word itself as lemma
            // Still create a lemmatization rule (word -> word, which is a no-op but consistent)
            std::string lemrule = make_lemmatization_rule(word, word);
            for (int len = 1; len <= max_len_utf8; ++len) {
                // Use UTF-8-aware suffix/prefix extraction
                std::string ending = use_prefix ? prefix(word, len) : suffix(word, len);
                WordEnding& entry = endings_[ending][token.tag];
                entry.tag = token.tag;
                entry.prob += 1.f;
                if (entry.lemma.empty()) {
                    entry.lemma = word;
                }
                if (!lemrule.empty()) {
                    entry.lemmatizations[lemrule] += 1;
                }
            }
            continue;
        }

        for (const auto& entry : token.entries) {
            std::string lemma = entry.lemma.empty() ? word : entry.lemma;
            
            // CRITICAL: Build lemmatization rules
            // For corpora like ode_ps, lemmatization is done from the normalized form (reg), not the original form
            // This is configurable via a setting, defaulting to false (use original form)
            // Check if we should use reg for lemmatization
            std::string lemmatization_source = word;  // Default: use original form
            bool use_reg_for_lemmatization = false;  // Will be set from settings if available
            
            // If entry has reg and it's different from form, check if we should use it
            if (!entry.reg.empty() && entry.reg != "_" && entry.reg != word) {
                // For now, we'll check a setting - but this should be passed from TaggerSettings
                // Since we don't have access to settings here, we'll use a heuristic:
                // If the lemma matches the reg (normalized form), it's likely that lemmatization
                // should be done from reg. Otherwise, use the original form.
                // This is a temporary solution - ideally settings should be passed to index_endings()
                if (lemma == entry.reg || (lemma != word && entry.reg.find(lemma) != std::string::npos)) {
                    use_reg_for_lemmatization = true;
                    lemmatization_source = entry.reg;
                }
            }
            
            // Build lemmatization rule from the appropriate source (form or reg)
            // neotagxml line 701, 643-694
            std::string lemrule = make_lemmatization_rule(lemmatization_source, lemma);
            
            // Use the original word for ending extraction (endings are based on surface form)
            // But lemmatization rules are built from the lemmatization source
            for (int len = 1; len <= max_len_utf8; ++len) {
                // Use UTF-8-aware suffix/prefix extraction
                std::string ending = use_prefix ? prefix(word, len) : suffix(word, len);
                WordEnding& ending_entry = endings_[ending][token.tag];
                ending_entry.tag = token.tag;
                ending_entry.prob += 1.f;
                if (ending_entry.lemma.empty()) {
                    ending_entry.lemma = lemma;
                }
                // Store lemmatization rule with frequency (neotagxml line 719)
                // The rule is built from lemmatization_source (form or reg) to lemma
                if (!lemrule.empty()) {
                    ending_entry.lemmatizations[lemrule] += 1;
                }
            }
        }
    }
}

bool Lexicon::load_json(const std::string& params_file, bool merge) {
    std::ifstream input(params_file);
    if (!input) {
        return false;
    }

    nlohmann::json root;
    try {
        input >> root;
    } catch (const std::exception&) {
        return false;
    }

    if (!merge) {
        reset();
    }

    std::string tag_attribute = "xpos";
    auto metadata_it = root.find("metadata");
    if (metadata_it != root.end() && metadata_it->is_object()) {
        auto attr_it = metadata_it->find("tag_attribute");
        if (attr_it != metadata_it->end() && attr_it->is_string()) {
            tag_attribute = attr_it->get<std::string>();
        }
    }

    auto vocab_it = root.find("vocab");
    if (vocab_it == root.end() || !vocab_it->is_object()) {
        return false;
    }

    const auto& vocab_root = *vocab_it;

    for (const auto& [form, value] : vocab_root.items()) {
        LexiconItem item;
        item.form = form;
        std::unordered_map<std::string, LexiconToken> tokens_by_tag;

        std::vector<LexiconEntry> lemma_only_entries;

        auto add_analysis = [&](const nlohmann::json& analysis) {
            if (!analysis.is_object()) {
                return;
            }

            LexiconEntry entry;
            entry.lemma = analysis.value("lemma", std::string{});
            entry.xpos = analysis.value("xpos", std::string{});
            entry.upos = analysis.value("upos", std::string{});
            entry.feats = analysis.value("feats", std::string{});
            entry.reg = analysis.value("reg", std::string{});
            entry.expan = analysis.value("expan", std::string{});
            entry.mod = analysis.value("mod", std::string{});
            entry.trslit = analysis.value("trslit", std::string{});
            entry.ltrslit = analysis.value("ltrslit", std::string{});
            entry.tokid = analysis.value("tokid", std::string{});
            entry.count = analysis.value("count", 0);

            entry.key = analysis.value("key", std::string{});
            if (entry.key.empty()) {
                entry.key = !entry.reg.empty() ? entry.reg : item.form;
            }

            auto clean = [](const std::string& value) -> std::string {
                if (value.empty() || value == "_" || value == "-") {
                    return std::string{};
                }
                return value;
            };

            std::string tag = analysis.value("tag", std::string{});
            if (tag.empty()) {
                if (tag_attribute == "upos") {
                    tag = clean(entry.upos);
                    if (tag.empty()) {
                        tag = clean(analysis.value("upos", std::string{}));
                    }
                } else if (tag_attribute == "utot") {
                    std::string upos_clean = clean(entry.upos);
                    if (upos_clean.empty()) {
                        upos_clean = clean(analysis.value("upos", std::string{}));
                    }
                    std::string feats_clean = clean(entry.feats);
                    if (feats_clean.empty()) {
                        feats_clean = clean(analysis.value("feats", std::string{}));
                    }
                    if (!upos_clean.empty() && !feats_clean.empty()) {
                        tag = upos_clean + "#" + feats_clean;
                    } else {
                        tag = upos_clean;
                    }
                } else {
                    tag = clean(entry.xpos);
                    if (tag.empty()) {
                        tag = clean(analysis.value("xpos", std::string{}));
                    }
                }
            }
            
            // Extract parts_tags early so we can use them to create a tag if needed
            std::vector<std::string> parts_tags;
            if (analysis.contains("parts_tags") && analysis["parts_tags"].is_array()) {
                for (const auto& part_tag_json : analysis["parts_tags"]) {
                    if (part_tag_json.is_string()) {
                        parts_tags.push_back(part_tag_json.get<std::string>());
                    }
                }
            }
            
            // Process parts (multiword tokens/contractions) before checking if tag is empty
            // This allows us to create a tag from parts_tags if tag is missing
            if (analysis.contains("parts") && analysis["parts"].is_array()) {
                std::size_t part_index = 0;
                for (const auto& part_json : analysis["parts"]) {
                    SubToken st;
                    
                    // Check if part is a full object (new format) or just a string (old format)
                    if (part_json.is_object()) {
                        // New format: full part object with all attributes
                        st.form = part_json.value("form", std::string{});
                        st.lemma = part_json.value("lemma", st.form);
                        st.upos = part_json.value("upos", std::string{});
                        st.xpos = part_json.value("xpos", std::string{});
                        st.feats = part_json.value("feats", std::string{});
                        if (part_json.contains("reg")) {
                            st.reg = part_json.value("reg", std::string{});
                        }
                        if (part_json.contains("expan")) {
                            st.expan = part_json.value("expan", std::string{});
                        }
                        // Normalize empty strings and "_" to empty
                        if (st.lemma == "_" || st.lemma == "-") st.lemma = st.form;
                        if (st.upos == "_") st.upos = "";
                        if (st.xpos == "_") st.xpos = "";
                        if (st.feats == "_") st.feats = "";
                    } else if (part_json.is_string()) {
                        // Old format: just form string, need to look up in vocab
                        st.form = part_json.get<std::string>();

                        const auto vocab_lookup = vocab_root.find(st.form);
                        if (vocab_lookup != vocab_root.end()) {
                            const auto& part_value = *vocab_lookup;
                            const nlohmann::json* best_analysis = nullptr;
                            if (part_value.is_array() && !part_value.empty()) {
                                best_analysis = &part_value.front();
                            } else if (part_value.is_object()) {
                                best_analysis = &part_value;
                            }
                            if (best_analysis) {
                                st.lemma = best_analysis->value("lemma", st.form);
                                st.xpos = best_analysis->value("xpos", std::string{});
                                st.upos = best_analysis->value("upos", std::string{});
                                st.feats = best_analysis->value("feats", std::string{});
                            }
                        }
                        if (st.lemma.empty() || st.lemma == "_" || st.lemma == "-") {
                            st.lemma = st.form;
                        }
                        // Use parts_tags if available (old format fallback)
                        if (part_index < parts_tags.size()) {
                            const std::string& part_tag = parts_tags[part_index];
                            if (!part_tag.empty()) {
                                if (tag_attribute == "upos") {
                                    st.upos = part_tag;
                                } else if (tag_attribute == "utot") {
                                    std::size_t hash = part_tag.find('#');
                                    if (hash != std::string::npos) {
                                        st.upos = part_tag.substr(0, hash);
                                        st.feats = part_tag.substr(hash + 1);
                                    } else {
                                        st.upos = part_tag;
                                    }
                                } else {
                                    st.xpos = part_tag;
                                }
                            }
                        }
                    } else {
                        continue;  // Skip invalid part entries
                    }
                    ++part_index;
                    entry.dtoks.push_back(std::move(st));
                }

                // Create tag from parts_tags if tag is empty or unknown
                if ((tag == "<unknown>" || tag.empty()) && !entry.dtoks.empty()) {
                    std::string combined;
                    for (std::size_t idx = 0; idx < entry.dtoks.size(); ++idx) {
                        if (idx > 0) {
                            combined += ".";
                        }
                        // Use the appropriate tag attribute based on tag_attribute setting
                        if (tag_attribute == "upos") {
                            combined += entry.dtoks[idx].upos.empty() ? entry.dtoks[idx].form : entry.dtoks[idx].upos;
                        } else if (tag_attribute == "utot") {
                            std::string part_tag;
                            if (!entry.dtoks[idx].upos.empty() && !entry.dtoks[idx].feats.empty()) {
                                part_tag = entry.dtoks[idx].upos + "#" + entry.dtoks[idx].feats;
                            } else if (!entry.dtoks[idx].upos.empty()) {
                                part_tag = entry.dtoks[idx].upos;
                            } else {
                                part_tag = entry.dtoks[idx].form;
                            }
                            combined += part_tag;
                        } else {
                            combined += entry.dtoks[idx].xpos.empty() ? entry.dtoks[idx].form : entry.dtoks[idx].xpos;
                        }
                    }
                    if (!combined.empty()) {
                        tag = combined;
                    }
                }
            }
            
            if (tag.empty()) {
                // Keep for normalization/lemmatization but don't contribute to tagging model.
                lemma_only_entries.push_back(std::move(entry));
                return;
            }

            if (entry.lemma.empty() || entry.lemma == "_" || entry.lemma == "-") {
                entry.lemma = item.form;
            }

            auto& token = tokens_by_tag[tag];
            if (token.tag.empty()) {
                token.tag = tag;
            }
            
            // When tagging from upos (or utot), entries with the same tag and lemma are redundant
            // Keep only the highest frequency entry per (tag, lemma) pair
            // This prevents duplicate candidates that will never be selected
            bool should_add = true;
            if (tag_attribute == "upos" || tag_attribute == "utot") {
                // Check if we already have an entry with the same tag and lemma
                for (auto& existing_entry : token.entries) {
                    if (existing_entry.lemma == entry.lemma) {
                        // Same tag and lemma - keep only the highest frequency one
                        if (entry.count > existing_entry.count) {
                            // Replace with higher frequency entry
                            existing_entry = std::move(entry);
                        }
                        // Otherwise, keep the existing one (higher or equal frequency)
                        should_add = false;
                        break;
                    }
                }
            }
            
            if (should_add) {
                if (entry.count > 0) {
                    token.count += entry.count;
                } else {
                    token.count += 1;
                }
                token.entries.push_back(std::move(entry));
            } else {
                // Entry was merged/replaced, update count
                if (entry.count > 0) {
                    token.count += entry.count;
                } else {
                    token.count += 1;
                }
            }
        };

        if (value.is_array()) {
            for (const auto& analysis : value) {
                add_analysis(analysis);
            }
        } else {
            add_analysis(value);
        }

        item.tokens.reserve(tokens_by_tag.size());
        for (auto& [tag, token] : tokens_by_tag) {
            if (token.count <= 0) {
                token.count = static_cast<int>(token.entries.size());
            }
            item.tokens.push_back(std::move(token));
        }

        // Append lemma-only entries without tags so normalization/lemmatization can still use them.
        if (!lemma_only_entries.empty()) {
            LexiconToken lemma_only_token;
            lemma_only_token.tag = "";  // indicates no tag available
            lemma_only_token.count = 0;
            lemma_only_token.entries = std::move(lemma_only_entries);
            item.tokens.push_back(std::move(lemma_only_token));
        }

        if (merge) {
            merge_item(item);
        } else {
            index_item(item);
        }
    }

    auto transitions_it = root.find("transitions");
    if (transitions_it != root.end() && transitions_it->is_object()) {
        std::unordered_map<std::string, TransitionStat> new_transitions;
        auto add_transition_table = [&](const nlohmann::json& table) {
            if (!table.is_object()) {
                return;
            }
            for (const auto& [prev_tag, next_map] : table.items()) {
                if (!next_map.is_object()) {
                    continue;
                }
                for (const auto& [next_tag, count_json] : next_map.items()) {
                    if (!count_json.is_number()) {
                        continue;
                    }
                    float count = static_cast<float>(count_json.get<double>());
                    std::string base_key = prev_tag + "." + next_tag;
                    TransitionStat direct{base_key, count};
                    new_transitions[direct.key] = direct;
                    TransitionStat trailing{base_key + ".", count};
                    new_transitions[trailing.key] = trailing;
                }
            }
        };

        bool loaded_transitions = false;
        auto table_it = transitions_it->find(tag_attribute);
        if (table_it != transitions_it->end()) {
            add_transition_table(*table_it);
            loaded_transitions = true;
        }
        if (!loaded_transitions) {
            // Fallback for legacy models
            if (transitions_it->contains("xpos")) {
                add_transition_table((*transitions_it)["xpos"]);
            }
            if (transitions_it->contains("upos")) {
                add_transition_table((*transitions_it)["upos"]);
            }
        }

        if (transitions_it->contains("start")) {
            const auto& start_map = (*transitions_it)["start"];
            if (start_map.is_object()) {
                for (const auto& [tag, count_json] : start_map.items()) {
                    if (!count_json.is_number()) {
                        continue;
                    }
                    float count = static_cast<float>(count_json.get<double>());
                    std::string key = std::string("START.") + tag;
                    new_transitions[key] = TransitionStat{key, count};
                }
            }
        }
        
        if (merge) {
            merge_transitions(new_transitions);
        } else {
            transitions_ = std::move(new_transitions);
        }
    }

    // Load dtoks (contractions/clitics) from JSON
    auto dtoks_it = root.find("dtoks");
    if (dtoks_it != root.end() && dtoks_it->is_object()) {
        std::unordered_map<std::string, DtokForm> new_dtoks;
        for (const auto& [clitic_form, dtok_data] : dtoks_it->items()) {
            if (!dtok_data.is_object()) {
                continue;
            }
            DtokForm form;
            form.key = clitic_form;
            form.lexcnt = dtok_data.value("lexcnt", 0);
            form.clitprob = dtok_data.value("clitprob", 0.0);
            form.position = dtok_data.value("position", std::string{});
            
            auto variants_it = dtok_data.find("variants");
            if (variants_it != dtok_data.end() && variants_it->is_array()) {
                for (const auto& variant_json : *variants_it) {
                    if (!variant_json.is_object()) {
                        continue;
                    }
                    DtokVariant variant;
                    variant.key = variant_json.value("key", std::string{});
                    variant.count = variant_json.value("count", 0);
                    
                    auto tokens_it = variant_json.find("tokens");
                    if (tokens_it != variant_json.end() && tokens_it->is_array()) {
                        for (const auto& token_json : *tokens_it) {
                            if (!token_json.is_object()) {
                                continue;
                            }
                            SubToken st;
                            st.form = token_json.value("form", std::string{});
                            st.lemma = token_json.value("lemma", std::string{});
                            st.xpos = token_json.value("xpos", std::string{});
                            st.upos = token_json.value("upos", std::string{});
                            st.feats = token_json.value("feats", std::string{});
                            variant.tokens.push_back(std::move(st));
                        }
                    }
                    form.variants.push_back(std::move(variant));
                }
            }
            new_dtoks[form.key] = std::move(form);
        }
        if (merge) {
            merge_dtoks(new_dtoks);
        } else {
            dtoks_ = std::move(new_dtoks);
        }
    }

    if (metadata_it != root.end() && metadata_it->is_object()) {
        auto caps_it = metadata_it->find("capitalizable_tags");
        if (caps_it != metadata_it->end() && caps_it->is_object()) {
            std::unordered_map<std::string, TagStat> new_tag_stats;
            auto process_caps_table = [&](const nlohmann::json& table) {
                if (!table.is_object()) {
                    return;
                }
                for (const auto& [tag, counts_json] : table.items()) {
                    if (!counts_json.is_object()) {
                        continue;
                    }
                    TagStat tag_stat;
                    tag_stat.key = tag;
                    tag_stat.count = 0.f;

                    for (const auto& [case_key, value] : counts_json.items()) {
                        if (!value.is_number()) {
                            continue;
                        }
                        float count = static_cast<float>(value.get<double>());
                        tag_stat.count += count;

                        std::string mapped_case;
                        if (case_key == "capitalized") {
                            mapped_case = "Ul";
                        } else if (case_key == "lowercase") {
                            mapped_case = "ll";
                        } else if (case_key == "uppercase") {
                            mapped_case = "UU";
                        } else {
                            mapped_case = case_key;
                        }

                        TagCaseStat case_stat;
                        case_stat.key = mapped_case;
                        case_stat.count = count;
                        tag_stat.cases.push_back(case_stat);
                    }

                    if (tag_stat.count <= 0.f) {
                        tag_stat.count = 1.f;
                    }

                    new_tag_stats[tag_stat.key] = std::move(tag_stat);
                }
            };

            if (caps_it->contains(tag_attribute)) {
                process_caps_table((*caps_it)[tag_attribute]);
            } else {
                // Backwards compatibility with legacy models
                if (caps_it->contains("xpos")) {
                    process_caps_table((*caps_it)["xpos"]);
                }
                if (caps_it->contains("upos")) {
                    process_caps_table((*caps_it)["upos"]);
                }
            }
            
            if (merge) {
                merge_tag_stats(new_tag_stats);
            } else {
                tag_stats_ = std::move(new_tag_stats);
            }
        }
    }

    // Store tag_attribute in default_settings so Tagger can access it
    default_settings_["tag_attribute"] = tag_attribute;
    
    // Load default settings from JSON (if present)
    auto settings_it = root.find("settings");
    if (settings_it != root.end() && settings_it->is_object()) {
        for (const auto& [key, value] : settings_it->items()) {
            if (value.is_string()) {
                default_settings_[key] = value.get<std::string>();
            } else if (value.is_number()) {
                // Convert numbers to strings
                if (value.is_number_float()) {
                    default_settings_[key] = std::to_string(value.get<float>());
                } else {
                    default_settings_[key] = std::to_string(value.get<int>());
                }
            } else if (value.is_boolean()) {
                default_settings_[key] = value.get<bool>() ? "1" : "0";
            }
        }
    }

    return true;
}

bool Lexicon::has_normalizations() const {
    // Check if any lexicon items have reg fields
    for (const auto& [form, item] : items_) {
        for (const auto& token : item.tokens) {
            for (const auto& entry : token.entries) {
                if (!entry.reg.empty() && entry.reg != "_" && entry.reg != form) {
                    return true;
                }
            }
        }
    }
    return false;
}

std::unordered_map<std::string, std::string> Lexicon::get_normalization_mappings() const {
    std::unordered_map<std::string, std::string> mappings;
    
    for (const auto& [form, item] : items_) {
        for (const auto& token : item.tokens) {
            for (const auto& entry : token.entries) {
                if (!entry.reg.empty() && entry.reg != "_" && entry.reg != form) {
                    // Use the first reg we find for this form
                    if (mappings.find(form) == mappings.end()) {
                        mappings[form] = entry.reg;
                    }
                }
            }
        }
    }
    
    return mappings;
}

void Lexicon::merge_item(const LexiconItem& item) {
    // Check if item already exists
    auto it = items_.find(item.form);
    if (it == items_.end()) {
        // New item - just add it
        index_item(item);
        return;
    }
    
    // Item exists - merge tokens
    LexiconItem& existing = it->second;
    
    // Create a map of existing tokens by tag for efficient lookup
    std::unordered_map<std::string, size_t> token_index_by_tag;
    for (size_t i = 0; i < existing.tokens.size(); ++i) {
        if (!existing.tokens[i].tag.empty()) {
            token_index_by_tag[existing.tokens[i].tag] = i;
        }
    }
    
    // Merge new tokens into existing
    for (const auto& new_token : item.tokens) {
        if (new_token.tag.empty()) {
            // Lemma-only token - append to existing
            existing.tokens.push_back(new_token);
            continue;
        }
        
        auto token_it = token_index_by_tag.find(new_token.tag);
        if (token_it == token_index_by_tag.end()) {
            // New tag - add token
            existing.tokens.push_back(new_token);
            token_index_by_tag[new_token.tag] = existing.tokens.size() - 1;
        } else {
            // Existing tag - merge entries and update counts
            LexiconToken& existing_token = existing.tokens[token_it->second];
            existing_token.count += new_token.count;
            
            // Merge entries (add new entries, avoiding exact duplicates)
            for (const auto& new_entry : new_token.entries) {
                bool found = false;
                for (const auto& existing_entry : existing_token.entries) {
                    if (existing_entry.lemma == new_entry.lemma &&
                        existing_entry.xpos == new_entry.xpos &&
                        existing_entry.upos == new_entry.upos &&
                        existing_entry.feats == new_entry.feats &&
                        existing_entry.reg == new_entry.reg &&
                        existing_entry.expan == new_entry.expan) {
                        // Exact duplicate - just update count
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    existing_token.entries.push_back(new_entry);
                }
            }
        }
    }
    
    // Re-index endings for the merged item
    index_endings(existing);
    
    // Update lowercase index
    lower_items_[to_lower(item.form)] = existing;
}

void Lexicon::merge_transitions(const std::unordered_map<std::string, TransitionStat>& new_transitions) {
    for (const auto& [key, new_stat] : new_transitions) {
        auto it = transitions_.find(key);
        if (it == transitions_.end()) {
            transitions_[key] = new_stat;
        } else {
            // Merge counts (additive)
            it->second.count += new_stat.count;
        }
    }
}

void Lexicon::merge_tag_stats(const std::unordered_map<std::string, TagStat>& new_tag_stats) {
    for (const auto& [key, new_stat] : new_tag_stats) {
        auto it = tag_stats_.find(key);
        if (it == tag_stats_.end()) {
            tag_stats_[key] = new_stat;
        } else {
            // Merge counts and cases
            TagStat& existing = it->second;
            existing.count += new_stat.count;
            
            // Merge cases
            std::unordered_map<std::string, size_t> case_index;
            for (size_t i = 0; i < existing.cases.size(); ++i) {
                case_index[existing.cases[i].key] = i;
            }
            
            for (const auto& new_case : new_stat.cases) {
                auto case_it = case_index.find(new_case.key);
                if (case_it == case_index.end()) {
                    existing.cases.push_back(new_case);
                } else {
                    existing.cases[case_it->second].count += new_case.count;
                }
            }
        }
    }
}

void Lexicon::merge_dtoks(const std::unordered_map<std::string, DtokForm>& new_dtoks) {
    for (const auto& [key, new_form] : new_dtoks) {
        auto it = dtoks_.find(key);
        if (it == dtoks_.end()) {
            dtoks_[key] = new_form;
        } else {
            // Merge dtok forms - update counts and merge variants
            DtokForm& existing = it->second;
            existing.lexcnt += new_form.lexcnt;
            existing.clitprob = (existing.clitprob + new_form.clitprob) / 2.0f;  // Average probability
            
            // Merge variants
            std::unordered_map<std::string, size_t> variant_index;
            for (size_t i = 0; i < existing.variants.size(); ++i) {
                variant_index[existing.variants[i].key] = i;
            }
            
            for (const auto& new_variant : new_form.variants) {
                auto var_it = variant_index.find(new_variant.key);
                if (var_it == variant_index.end()) {
                    existing.variants.push_back(new_variant);
                } else {
                    existing.variants[var_it->second].count += new_variant.count;
                }
            }
        }
    }
}

void Lexicon::compute_tag_type_counts() const {
    if (tag_type_counts_computed_) {
        return;  // Already computed
    }
    
    // Count the number of unique vocabulary entries (types) per tag
    // This is different from token counts - we count each unique form once per tag
    // If a form has multiple analyses with the same tag, we still count it only once per tag
    std::unordered_map<std::string, std::unordered_set<std::string>> tags_by_form;
    
    // First pass: collect all tags for each form
    for (const auto& [form, item] : items_) {
        for (const auto& token : item.tokens) {
            if (!token.tag.empty() && token.tag != "<unknown>") {
                tags_by_form[form].insert(token.tag);
            }
        }
    }
    
    // Second pass: count unique forms per tag
    tag_type_counts_cache_.clear();
    for (const auto& [form, tags] : tags_by_form) {
        for (const auto& tag : tags) {
            tag_type_counts_cache_[tag]++;
        }
    }
    
    tag_type_counts_computed_ = true;
}

const std::unordered_map<std::string, int>& Lexicon::tag_type_counts() const {
    compute_tag_type_counts();
    return tag_type_counts_cache_;
}

void Lexicon::merge_endings(const std::unordered_map<std::string, std::unordered_map<std::string, WordEnding>>& new_endings) {
    for (const auto& [ending_key, new_tag_map] : new_endings) {
        auto ending_it = endings_.find(ending_key);
        if (ending_it == endings_.end()) {
            endings_[ending_key] = new_tag_map;
        } else {
            // Merge tag map for this ending
            std::unordered_map<std::string, WordEnding>& existing_tag_map = ending_it->second;
            for (const auto& [tag, new_ending] : new_tag_map) {
                auto tag_it = existing_tag_map.find(tag);
                if (tag_it == existing_tag_map.end()) {
                    existing_tag_map[tag] = new_ending;
                } else {
                    // Merge ending - add probabilities and merge lemmatizations
                    WordEnding& existing = tag_it->second;
                    existing.prob += new_ending.prob;
                    
                    // Merge lemmatization rules
                    for (const auto& [rule, count] : new_ending.lemmatizations) {
                        existing.lemmatizations[rule] += count;
                    }
                    
                    // Keep lemma from first occurrence (or prefer non-empty)
                    if (existing.lemma.empty() && !new_ending.lemma.empty()) {
                        existing.lemma = new_ending.lemma;
                    }
                }
            }
        }
    }
}

} // namespace flexitag
