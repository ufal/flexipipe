/*
 * Python bindings for FlexiPipe C++ pipeline
 * Direct function calls - no subprocess overhead, no serialization
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "vocab_loader.h"
#include "types.h"
#include "io_conllu.h"
#include "io_teitok.h"
#include "tokenizer.h"
#include "normalizer.h"
#include "contractions.h"
#include "viterbi_optimized.h"
#include <unordered_set>
#include <algorithm>
#include <cctype>
#include <iostream>
#include <climits>

// Include the FlexiPipePipeline implementation
// We'll need to make it accessible - for now, duplicate the key parts
namespace py = pybind11;

// Forward declaration
class FlexiPipePipeline;

// Simplified pipeline class for Python bindings
class PipelineBinding {
private:
    Vocab vocab;
    VocabLoader vocab_loader;
    std::string tag_type = "upos";
    
    std::vector<std::string> viterbi_tag(const std::vector<std::string>& sentence) {
        return ViterbiTagger::tag_sentence(sentence, vocab, tag_type);
    }
    
    std::string normalize(const std::string& form) {
        std::string normalized = Normalizer::normalize(form, vocab, true);
        return normalized.empty() ? form : normalized;
    }
    
    std::vector<std::string> split_contraction(const std::string& form, 
                                               const std::string& upos = "", 
                                               const std::string& xpos = "") {
        return ContractionSplitter::split(form, vocab, upos, xpos);
    }
    
    std::string lemmatize(const std::string& form, const std::string& tag, 
                         const VocabAnalysis* analysis = nullptr) {
        if (analysis && !analysis->lemma.empty() && analysis->lemma != "_") {
            return analysis->lemma;
        }
        
        const std::vector<VocabAnalysis>* analyses = vocab.get(form);
        if (!analyses) {
            std::string form_lower = form;
            std::transform(form_lower.begin(), form_lower.end(), form_lower.begin(), ::tolower);
            analyses = vocab.get(form_lower);
        }
        
        if (analyses && !analyses->empty()) {
            for (const auto& a : *analyses) {
                std::string analysis_tag = (tag_type == "upos") ? a.upos : a.xpos;
                if (analysis_tag == tag && !a.lemma.empty() && a.lemma != "_") {
                    return a.lemma;
                }
            }
            if (!analyses->front().lemma.empty() && analyses->front().lemma != "_") {
                return analyses->front().lemma;
            }
        }
        
        std::string lemma = form;
        std::transform(lemma.begin(), lemma.end(), lemma.begin(), ::tolower);
        return lemma;
    }
    
public:
    bool load_vocab(const std::string& vocab_file) {
        return vocab_loader.load(vocab_file, vocab);
    }
    
    // Convert Token to Python dict
    py::dict token_to_dict(const Token& token) {
        py::dict result;
        
        // Handle MWT with nested subtokens first (before setting regular id)
        if (token.is_mwt) {
            result["_is_mwt"] = true;
            // Check subtokens size - access directly to ensure we get the actual size
            const std::vector<Token>& subtokens_ref = token.subtokens;
            size_t subtokens_size = subtokens_ref.size();
            // Debug output only in debug mode
            
            if (subtokens_size > 0) {
                // MWT token: use range id based on subtoken ids
                int start_id = subtokens_ref[0].id;
                int end_id = subtokens_ref.back().id;
                result["id"] = std::to_string(start_id) + "-" + std::to_string(end_id);
                
                // Add subtokens as nested list - build it explicitly
                py::list subtokens_list;
                // Use size_t iteration to avoid any iterator issues
                for (size_t k = 0; k < subtokens_size; k++) {
                    // Recursively convert each subtoken
                    py::dict subtoken_dict = token_to_dict(subtokens_ref[k]);
                    subtokens_list.append(subtoken_dict);
                }
                result["subtokens"] = subtokens_list;
            } else if (token.mwt_start > 0 && token.mwt_end > 0) {
                // Fallback: use old mwt_start/mwt_end if subtokens are empty
                result["id"] = std::to_string(token.mwt_start) + "-" + std::to_string(token.mwt_end);
                result["subtokens"] = py::list();  // Empty list - subtokens were lost
            } else if (token.id > 0) {
                result["id"] = token.id;
                result["subtokens"] = py::list();  // Empty list
            }
        } else if (token.id > 0) {
            result["id"] = token.id;
        }
        
        if (!token.form.empty()) {
            result["form"] = token.form;
        }
        if (!token.lemma.empty() && token.lemma != "_") {
            result["lemma"] = token.lemma;
        }
        if (!token.upos.empty() && token.upos != "_") {
            result["upos"] = token.upos;
        }
        if (!token.xpos.empty() && token.xpos != "_") {
            result["xpos"] = token.xpos;
        }
        if (!token.feats.empty() && token.feats != "_") {
            result["feats"] = token.feats;
        }
        if (!token.head.empty() && token.head != "_" && token.head != "0") {
            result["head"] = token.head;
        }
        if (!token.deprel.empty() && token.deprel != "_") {
            result["deprel"] = token.deprel;
        }
        if (!token.misc.empty() && token.misc != "_") {
            result["misc"] = token.misc;
        }
        if (!token.norm_form.empty() && token.norm_form != "_") {
            result["norm_form"] = token.norm_form;
        }
        if (!token.expan.empty() && token.expan != "_") {
            result["expan"] = token.expan;
        }
        if (!token.tok_id.empty()) {
            result["tok_id"] = token.tok_id;
        }
        if (!token.dtok_id.empty()) {
            result["dtok_id"] = token.dtok_id;
        }
        
        return result;
    }
    
    // Process text and return Python list of sentences
    py::list process_text(const std::string& text, bool segment, bool tokenize) {
        std::vector<Sentence> sentences;
        
        if (segment) {
            std::vector<std::string> sentence_texts = SentenceSegmenter::segment(text);
            
            for (const auto& sent_text : sentence_texts) {
                Sentence sentence;
                sentence.text = sent_text;
                
                if (tokenize) {
                    std::vector<std::string> tokens = Tokenizer::tokenize_ud_style(sent_text);
                    
                    for (size_t i = 0; i < tokens.size(); i++) {
                        Token token;
                        token.id = i + 1;
                        token.form = tokens[i];
                        
                        std::string normalized = normalize(token.form);
                        if (normalized != token.form) {
                            token.norm_form = normalized;
                        }
                        
                        std::string form_to_check = normalized.empty() ? token.form : normalized;
                        std::vector<std::string> parts = split_contraction(form_to_check);
                        
                        if (!parts.empty() && parts.size() > 1) {
                            // Create nested MWT structure (TEITOK-style)
                            int next_id = sentence.tokens.size() + 1;
                            int mwt_start = next_id;
                            int mwt_end = next_id + parts.size() - 1;
                            
                            Token mwt_token = token;
                            mwt_token.is_mwt = true;
                            mwt_token.subtokens.clear();
                            // Preserve the original contraction form (e.g., "im")
                            mwt_token.form = token.form;  // Original form, not normalized
                            // Set mwt_start/mwt_end as backup in case subtokens are lost
                            mwt_token.mwt_start = mwt_start;
                            mwt_token.mwt_end = mwt_end;
                            
                            // Create subtokens
                            for (size_t j = 0; j < parts.size(); j++) {
                                Token subtoken;
                                subtoken.id = next_id + j;
                                subtoken.form = parts[j];
                                subtoken.norm_form = token.norm_form;  // Inherit normalization
                                mwt_token.subtokens.push_back(subtoken);
                            }
                            
                            // Ensure subtokens are preserved when pushing to sentence
                            // Debug: verify subtokens before push
                            size_t debug_subtokens_before = mwt_token.subtokens.size();
                            sentence.tokens.push_back(mwt_token);
                            // Debug: verify subtokens after push
                            size_t debug_subtokens_after = sentence.tokens.back().subtokens.size();
                            // Debug check (only in debug mode)
                            if (debug_subtokens_before != debug_subtokens_after) {
                                // This should never happen - copy constructor should preserve subtokens
                                // Only print in debug mode (not verbose)
                            }
                        } else {
                            sentence.tokens.push_back(token);
                        }
                    }
                } else {
                    std::vector<std::string> tokens = Tokenizer::tokenize_whitespace(sent_text);
                    for (size_t i = 0; i < tokens.size(); i++) {
                        Token token;
                        token.id = i + 1;
                        token.form = tokens[i];
                        sentence.tokens.push_back(token);
                    }
                }
                
                // Tag the sentence (handle nested subtokens)
                if (!sentence.tokens.empty()) {
                    std::vector<std::string> forms;
                    std::vector<std::pair<size_t, size_t>> token_indices;  // (token_idx, subtoken_idx or -1)
                    
                    // Collect forms from regular tokens and MWT subtokens
                    for (size_t i = 0; i < sentence.tokens.size(); i++) {
                        if (sentence.tokens[i].is_mwt && !sentence.tokens[i].subtokens.empty()) {
                            // MWT: add subtoken forms
                            for (size_t j = 0; j < sentence.tokens[i].subtokens.size(); j++) {
                                std::string form = sentence.tokens[i].subtokens[j].norm_form.empty() ? 
                                                  sentence.tokens[i].subtokens[j].form : sentence.tokens[i].subtokens[j].norm_form;
                                forms.push_back(form);
                                token_indices.push_back({i, j});
                            }
                        } else if (!sentence.tokens[i].is_mwt) {
                            // Regular token
                            forms.push_back(sentence.tokens[i].norm_form.empty() ? 
                                           sentence.tokens[i].form : sentence.tokens[i].norm_form);
                            token_indices.push_back({i, SIZE_MAX});  // SIZE_MAX means regular token
                        }
                    }
                    
                    std::vector<std::string> upos_tags = viterbi_tag(forms);
                    
                    tag_type = "xpos";
                    std::vector<std::string> xpos_tags = viterbi_tag(forms);
                    tag_type = "upos";
                    
                    // Apply tags to tokens/subtokens
                    for (size_t form_idx = 0; form_idx < forms.size() && form_idx < upos_tags.size(); form_idx++) {
                        size_t token_idx = token_indices[form_idx].first;
                        size_t subtoken_idx = token_indices[form_idx].second;
                        
                        if (subtoken_idx != SIZE_MAX) {
                            // Subtoken within MWT
                            Token& subtoken = sentence.tokens[token_idx].subtokens[subtoken_idx];
                            subtoken.upos = upos_tags[form_idx];
                            if (form_idx < xpos_tags.size()) {
                                subtoken.xpos = xpos_tags[form_idx];
                            }
                            
                            std::string form_to_lookup = subtoken.norm_form.empty() ? subtoken.form : subtoken.norm_form;
                            const std::vector<VocabAnalysis>* analyses = vocab.get(form_to_lookup);
                            if (!analyses) {
                                std::string form_lower = form_to_lookup;
                                std::transform(form_lower.begin(), form_lower.end(), form_lower.begin(), ::tolower);
                                analyses = vocab.get(form_lower);
                            }
                            
                            const VocabAnalysis* analysis = nullptr;
                            if (analyses && !analyses->empty()) {
                                int best_count = 0;
                                for (const auto& a : *analyses) {
                                    bool upos_match = (a.upos == upos_tags[form_idx] && !upos_tags[form_idx].empty() && upos_tags[form_idx] != "_");
                                    bool xpos_match = (form_idx < xpos_tags.size() && a.xpos == xpos_tags[form_idx] && !xpos_tags[form_idx].empty() && xpos_tags[form_idx] != "_");
                                    
                                    if (upos_match && xpos_match) {
                                        analysis = &a;
                                        break;
                                    } else if (upos_match && !analysis) {
                                        analysis = &a;
                                    } else if (!analysis && a.count > best_count) {
                                        best_count = a.count;
                                        analysis = &a;
                                    }
                                }
                                if (!analysis) {
                                    analysis = &analyses->front();
                                }
                            }
                            
                            if (analysis) {
                                subtoken.lemma = lemmatize(form_to_lookup, upos_tags[form_idx], analysis);
                                if (!analysis->feats.empty() && analysis->feats != "_") {
                                    subtoken.feats = analysis->feats;
                                }
                                if (!analysis->xpos.empty() && analysis->xpos != "_" && subtoken.xpos.empty()) {
                                    subtoken.xpos = analysis->xpos;
                                }
                            } else {
                                subtoken.lemma = lemmatize(form_to_lookup, upos_tags[form_idx], nullptr);
                            }
                        } else {
                            // Regular token
                            sentence.tokens[token_idx].upos = upos_tags[form_idx];
                            if (form_idx < xpos_tags.size()) {
                                sentence.tokens[token_idx].xpos = xpos_tags[form_idx];
                            }
                            
                            std::string form_to_lookup = sentence.tokens[token_idx].norm_form.empty() ? 
                                                        sentence.tokens[token_idx].form : sentence.tokens[token_idx].norm_form;
                            const std::vector<VocabAnalysis>* analyses = vocab.get(form_to_lookup);
                            if (!analyses) {
                                std::string form_lower = form_to_lookup;
                                std::transform(form_lower.begin(), form_lower.end(), form_lower.begin(), ::tolower);
                                analyses = vocab.get(form_lower);
                            }
                            
                            const VocabAnalysis* analysis = nullptr;
                            if (analyses && !analyses->empty()) {
                                int best_count = 0;
                                for (const auto& a : *analyses) {
                                    bool upos_match = (a.upos == upos_tags[form_idx] && !upos_tags[form_idx].empty() && upos_tags[form_idx] != "_");
                                    bool xpos_match = (form_idx < xpos_tags.size() && a.xpos == xpos_tags[form_idx] && !xpos_tags[form_idx].empty() && xpos_tags[form_idx] != "_");
                                    
                                    if (upos_match && xpos_match) {
                                        analysis = &a;
                                        break;
                                    } else if (upos_match && !analysis) {
                                        analysis = &a;
                                    } else if (!analysis && a.count > best_count) {
                                        best_count = a.count;
                                        analysis = &a;
                                    }
                                }
                                if (!analysis) {
                                    analysis = &analyses->front();
                                }
                            }
                            
                            if (analysis) {
                                sentence.tokens[token_idx].lemma = lemmatize(form_to_lookup, upos_tags[form_idx], analysis);
                                if (!analysis->feats.empty() && analysis->feats != "_") {
                                    sentence.tokens[token_idx].feats = analysis->feats;
                                }
                                if (!analysis->xpos.empty() && analysis->xpos != "_" && sentence.tokens[token_idx].xpos.empty()) {
                                    sentence.tokens[token_idx].xpos = analysis->xpos;
                                }
                            } else {
                                sentence.tokens[token_idx].lemma = lemmatize(form_to_lookup, upos_tags[form_idx], nullptr);
                            }
                        }
                    }
                    
                    // Recalculate token ids (with nested structure)
                    int current_id = 1;
                    for (size_t i = 0; i < sentence.tokens.size(); i++) {
                        if (sentence.tokens[i].is_mwt && !sentence.tokens[i].subtokens.empty()) {
                            // MWT: assign ids to subtokens and update mwt_start/mwt_end
                            int mwt_start = current_id;
                            for (auto& subtoken : sentence.tokens[i].subtokens) {
                                subtoken.id = current_id++;
                            }
                            int mwt_end = current_id - 1;
                            // Update mwt_start/mwt_end as backup
                            sentence.tokens[i].mwt_start = mwt_start;
                            sentence.tokens[i].mwt_end = mwt_end;
                        } else if (!sentence.tokens[i].is_mwt) {
                            // Regular token
                            sentence.tokens[i].id = current_id++;
                        }
                    }
                }
                
                // Verify subtokens are preserved before pushing to sentences
                for (size_t i = 0; i < sentence.tokens.size(); i++) {
                    if (sentence.tokens[i].is_mwt) {
                        size_t subtokens_size = sentence.tokens[i].subtokens.size();
                        if (subtokens_size == 0 && sentence.tokens[i].mwt_start > 0) {
                            // Subtokens are empty but mwt_start is set - this shouldn't happen
                            // but it means subtokens were lost somewhere
                            // Debug warning (only in debug mode)
                        } else if (subtokens_size > 0) {
                            // Debug output only in debug mode (not verbose)
                        }
                    }
                }
                
                sentences.push_back(sentence);
                
                // Verify subtokens are preserved after pushing to sentences
                if (!sentences.empty()) {
                    const Sentence& last_sent = sentences.back();
                    for (size_t i = 0; i < last_sent.tokens.size(); i++) {
                        if (last_sent.tokens[i].is_mwt) {
                            size_t subtokens_size = last_sent.tokens[i].subtokens.size();
                            if (subtokens_size > 0) {
                                // Debug output only in debug mode
                            } else {
                                // Debug error (only in debug mode)
                            }
                        }
                    }
                }
            }
        } else {
            Sentence sentence;
            sentence.text = text;
            
            if (tokenize) {
                std::vector<std::string> tokens = Tokenizer::tokenize_ud_style(text);
                for (size_t i = 0; i < tokens.size(); i++) {
                    Token token;
                    token.id = i + 1;
                    token.form = tokens[i];
                    sentence.tokens.push_back(token);
                }
            }
            
            sentences.push_back(sentence);
        }
        
        // Convert to Python format (nested structure)
        py::list result;
        
        for (size_t sent_idx = 0; sent_idx < sentences.size(); sent_idx++) {
            const Sentence& sentence = sentences[sent_idx];
            py::list sent_list;
            // Use index-based iteration to avoid copy issues with nested vectors
            for (size_t i = 0; i < sentence.tokens.size(); i++) {
                // Access token directly from sentence vector to ensure nested vectors are preserved
                const Token& token = sentence.tokens[i];
                
                // With nested structure, we only add MWT tokens and regular tokens
                // Subtokens are nested inside MWT tokens, not separate in the list
                if (token.is_mwt) {
                    // Access subtokens directly from the sentence vector to ensure they're preserved
                    // Check if subtokens exist in the actual sentence vector
                    size_t subtokens_size = sentence.tokens[i].subtokens.size();
                    // Debug output only in debug mode
                    if (subtokens_size > 0) {
                        // Subtokens exist - convert normally
                        sent_list.append(token_to_dict(sentence.tokens[i]));
                    } else {
                        // Subtokens are empty - use fallback
                        // Debug error (only in debug mode)
                        sent_list.append(token_to_dict(sentence.tokens[i]));
                    }
                } else if (!token.is_mwt) {
                    // Regular token - add it
                    sent_list.append(token_to_dict(token));
                }
                // Skip tokens that are part of old flat structure (shouldn't happen with nested)
            }
            result.append(sent_list);
        }
        
        return result;
    }
    
    // Process file (CoNLL-U or text file)
    py::list process_file(const std::string& input_file, const std::string& input_format,
                         bool segment, bool tokenize,
                         bool skip_split = false, bool skip_tag = false, bool skip_lemma = false) {
        std::vector<Sentence> sentences;
        
        // Read file content
        std::ifstream file(input_file);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open input file: " + input_file);
        }
        std::string content((std::istreambuf_iterator<char>(file)),
                          std::istreambuf_iterator<char>());
        file.close();
        
        // Process based on format
        std::string format = input_format;
        if (format == "auto") {
            if (input_file.find(".conllu") != std::string::npos || input_file.find(".conll") != std::string::npos) {
                format = "conllu";
            } else if (input_file.find(".xml") != std::string::npos) {
                format = "teitok";
            } else {
                format = "text";
            }
        }
        
        if (format == "conllu") {
            // Load and process CoNLL-U (same logic as FlexiPipePipeline::process_conllu)
            sentences = CoNLLUReader::load_string(content);
            
            // Process each sentence
            for (auto& sentence : sentences) {
                if (sentence.tokens.empty()) {
                    continue;
                }
                
                // Apply normalization first (needed for contraction splitting)
                for (auto& token : sentence.tokens) {
                    if (!token.is_mwt) {
                        std::string normalized = normalize(token.form);
                        if (normalized != token.form) {
                            token.norm_form = normalized;
                        }
                    }
                }
                
                // Apply contraction splitting if not skipped
                if (!skip_split) {
                    std::vector<Token> new_tokens;
                    // Track which tokens are part of existing MWTs (to skip them)
                    std::unordered_set<size_t> tokens_in_mwt;
                    // First pass: find existing MWTs and nest their split tokens
                    // Process in order to preserve token sequence
                    for (size_t i = 0; i < sentence.tokens.size(); i++) {
                        const auto& token = sentence.tokens[i];
                        
                        // Skip if already part of an MWT
                        if (tokens_in_mwt.count(i)) {
                            continue;
                        }
                        
                        if (token.is_mwt && token.mwt_start > 0 && token.mwt_end > 0) {
                            // Existing MWT from CoNLL-U - find its split tokens and nest them
                            Token mwt_token = token;
                            mwt_token.subtokens.clear();
                            
                            // Find split tokens that belong to this MWT (in order by ID)
                            std::vector<Token> subtokens_found;
                            for (size_t j = 0; j < sentence.tokens.size(); j++) {
                                if (j == i) continue;  // Skip the MWT token itself
                                const auto& other_token = sentence.tokens[j];
                                if (!other_token.is_mwt && 
                                    other_token.id >= token.mwt_start && 
                                    other_token.id <= token.mwt_end) {
                                    // This is a split token belonging to this MWT
                                    subtokens_found.push_back(other_token);
                                    tokens_in_mwt.insert(j);  // Mark for skipping
                                }
                            }
                            
                            // Sort subtokens by ID to ensure correct order
                            std::sort(subtokens_found.begin(), subtokens_found.end(),
                                     [](const Token& a, const Token& b) { return a.id < b.id; });
                            
                            // Add sorted subtokens to MWT
                            mwt_token.subtokens = subtokens_found;
                            
                            // Add the nested MWT (preserving original position)
                            new_tokens.push_back(mwt_token);
                        } else if (!token.is_mwt) {
                            // Regular token - check if it needs splitting
                            std::string form_to_check = token.norm_form.empty() ? token.form : token.norm_form;
                            std::vector<std::string> parts = split_contraction(form_to_check);
                            
                            if (!parts.empty() && parts.size() > 1) {
                                // Create nested MWT structure (TEITOK-style) - same as standalone tool
                                int next_id = new_tokens.size() + 1;
                                int mwt_start = next_id;
                                int mwt_end = next_id + parts.size() - 1;
                                
                                Token mwt_token = token;
                                mwt_token.is_mwt = true;
                                mwt_token.mwt_start = mwt_start;
                                mwt_token.mwt_end = mwt_end;
                                mwt_token.parts = parts;
                                mwt_token.subtokens.clear();  // Clear any existing subtokens
                                // Preserve the original contraction form (e.g., "im")
                                mwt_token.form = token.form;  // Original form, not normalized
                                
                                // Create subtokens and add them to the MWT token's subtokens vector
                                // This creates the nested structure
                                for (size_t j = 0; j < parts.size(); j++) {
                                    Token subtoken;
                                    subtoken.id = mwt_start + j;
                                    subtoken.form = parts[j];
                                    subtoken.norm_form = token.norm_form;  // Inherit normalization
                                    mwt_token.subtokens.push_back(subtoken);
                                }
                                
                                // Add MWT token to sentence (with nested subtokens)
                                // Only the MWT token is added - subtokens are nested inside it
                                new_tokens.push_back(mwt_token);
                            } else {
                                new_tokens.push_back(token);
                            }
                        }
                    }
                    sentence.tokens = new_tokens;
                }
                
                // Apply tagging if not skipped
                if (!skip_tag) {
                    std::vector<std::string> forms;
                    std::vector<std::pair<size_t, size_t>> token_indices;  // (token_idx, subtoken_idx or SIZE_MAX)
                    
                    // Collect forms from regular tokens and MWT subtokens
                    for (size_t i = 0; i < sentence.tokens.size(); i++) {
                        if (sentence.tokens[i].is_mwt && !sentence.tokens[i].subtokens.empty()) {
                            // MWT: add subtoken forms
                            for (size_t j = 0; j < sentence.tokens[i].subtokens.size(); j++) {
                                std::string form = sentence.tokens[i].subtokens[j].norm_form.empty() ? 
                                                  sentence.tokens[i].subtokens[j].form : sentence.tokens[i].subtokens[j].norm_form;
                                forms.push_back(form);
                                token_indices.push_back({i, j});
                            }
                        } else if (!sentence.tokens[i].is_mwt) {
                            // Regular token
                            std::string form = sentence.tokens[i].norm_form.empty() ? 
                                              sentence.tokens[i].form : sentence.tokens[i].norm_form;
                            forms.push_back(form);
                            token_indices.push_back({i, SIZE_MAX});  // SIZE_MAX means regular token
                        }
                    }
                    
                    if (!forms.empty()) {
                        std::vector<std::string> upos_tags = viterbi_tag(forms);
                        tag_type = "xpos";
                        std::vector<std::string> xpos_tags = viterbi_tag(forms);
                        tag_type = "upos";
                        
                        for (size_t form_idx = 0; form_idx < forms.size() && form_idx < upos_tags.size(); form_idx++) {
                            size_t token_idx = token_indices[form_idx].first;
                            size_t subtoken_idx = token_indices[form_idx].second;
                            
                            if (subtoken_idx != SIZE_MAX) {
                                // Subtoken within MWT
                                Token& subtoken = sentence.tokens[token_idx].subtokens[subtoken_idx];
                                if (subtoken.upos.empty() || subtoken.upos == "_") {
                                    subtoken.upos = upos_tags[form_idx];
                                }
                                if (form_idx < xpos_tags.size()) {
                                    if (subtoken.xpos.empty() || subtoken.xpos == "_") {
                                        subtoken.xpos = xpos_tags[form_idx];
                                    }
                                }
                                
                                std::string form_to_lookup = subtoken.norm_form.empty() ? subtoken.form : subtoken.norm_form;
                                const std::vector<VocabAnalysis>* analyses = vocab.get(form_to_lookup);
                                if (!analyses) {
                                    std::string form_lower = form_to_lookup;
                                    std::transform(form_lower.begin(), form_lower.end(), form_lower.begin(), ::tolower);
                                    analyses = vocab.get(form_lower);
                                }
                                
                                const VocabAnalysis* analysis = nullptr;
                                if (analyses && !analyses->empty()) {
                                    for (const auto& a : *analyses) {
                                        if (a.upos == upos_tags[form_idx] && !upos_tags[form_idx].empty() && upos_tags[form_idx] != "_") {
                                            analysis = &a;
                                            break;
                                        }
                                    }
                                    if (!analysis) {
                                        analysis = &analyses->front();
                                    }
                                }
                                
                                if (analysis) {
                                    if ((subtoken.feats.empty() || subtoken.feats == "_") &&
                                        !analysis->feats.empty() && analysis->feats != "_") {
                                        subtoken.feats = analysis->feats;
                                    }
                                    if (!skip_lemma && (subtoken.lemma.empty() || subtoken.lemma == "_")) {
                                        subtoken.lemma = lemmatize(form_to_lookup, upos_tags[form_idx], analysis);
                                    }
                                } else {
                                    if (!skip_lemma && (subtoken.lemma.empty() || subtoken.lemma == "_")) {
                                        subtoken.lemma = lemmatize(form_to_lookup, upos_tags[form_idx], nullptr);
                                    }
                                }
                            } else {
                                // Regular token
                                if (sentence.tokens[token_idx].upos.empty() || sentence.tokens[token_idx].upos == "_") {
                                    sentence.tokens[token_idx].upos = upos_tags[form_idx];
                                }
                                if (form_idx < xpos_tags.size()) {
                                    if (sentence.tokens[token_idx].xpos.empty() || sentence.tokens[token_idx].xpos == "_") {
                                        sentence.tokens[token_idx].xpos = xpos_tags[form_idx];
                                    }
                                }
                                
                                std::string form_to_lookup = sentence.tokens[token_idx].norm_form.empty() ? 
                                                            sentence.tokens[token_idx].form : sentence.tokens[token_idx].norm_form;
                                const std::vector<VocabAnalysis>* analyses = vocab.get(form_to_lookup);
                                if (!analyses) {
                                    std::string form_lower = form_to_lookup;
                                    std::transform(form_lower.begin(), form_lower.end(), form_lower.begin(), ::tolower);
                                    analyses = vocab.get(form_lower);
                                }
                                
                                const VocabAnalysis* analysis = nullptr;
                                if (analyses && !analyses->empty()) {
                                    for (const auto& a : *analyses) {
                                        if (a.upos == upos_tags[form_idx] && !upos_tags[form_idx].empty() && upos_tags[form_idx] != "_") {
                                            analysis = &a;
                                            break;
                                        }
                                    }
                                    if (!analysis) {
                                        analysis = &analyses->front();
                                    }
                                }
                                
                                if (analysis) {
                                    if ((sentence.tokens[token_idx].feats.empty() || sentence.tokens[token_idx].feats == "_") &&
                                        !analysis->feats.empty() && analysis->feats != "_") {
                                        sentence.tokens[token_idx].feats = analysis->feats;
                                    }
                                    if (!skip_lemma && (sentence.tokens[token_idx].lemma.empty() || sentence.tokens[token_idx].lemma == "_")) {
                                        sentence.tokens[token_idx].lemma = lemmatize(form_to_lookup, upos_tags[form_idx], analysis);
                                    }
                                } else {
                                    if (!skip_lemma && (sentence.tokens[token_idx].lemma.empty() || sentence.tokens[token_idx].lemma == "_")) {
                                        sentence.tokens[token_idx].lemma = lemmatize(form_to_lookup, upos_tags[form_idx], nullptr);
                                    }
                                }
                            }
                        }
                    }
                }
                
                // Recalculate token ids (with nested structure) - same as in process_text
                int current_id = 1;
                for (size_t i = 0; i < sentence.tokens.size(); i++) {
                    if (sentence.tokens[i].is_mwt && !sentence.tokens[i].subtokens.empty()) {
                        // MWT: assign ids to subtokens and update mwt_start/mwt_end
                        int mwt_start = current_id;
                        for (auto& subtoken : sentence.tokens[i].subtokens) {
                            subtoken.id = current_id++;
                        }
                        int mwt_end = current_id - 1;
                        // Update mwt_start/mwt_end as backup
                        sentence.tokens[i].mwt_start = mwt_start;
                        sentence.tokens[i].mwt_end = mwt_end;
                    } else if (!sentence.tokens[i].is_mwt) {
                        // Regular token
                        sentence.tokens[i].id = current_id++;
                    }
                }
            }
        } else if (format == "text") {
            // Process as text
            py::list text_result = process_text(content, segment, tokenize);
            return text_result;
        } else {
            throw std::runtime_error("Unsupported input format: " + format);
        }
        
        // Convert to Python format (nested structure)
        py::list result;
        
        for (const auto& sentence : sentences) {
            py::list sent_list;
            for (const auto& token : sentence.tokens) {
                // With nested structure, we only add MWT tokens and regular tokens
                // Subtokens are nested inside MWT tokens, not separate in the list
                if (token.is_mwt && !token.subtokens.empty()) {
                    // MWT with nested subtokens - add it
                    sent_list.append(token_to_dict(token));
                } else if (!token.is_mwt) {
                    // Regular token - add it
                    sent_list.append(token_to_dict(token));
                }
                // Skip tokens that are part of old flat structure (shouldn't happen with nested)
            }
            result.append(sent_list);
        }
        
        return result;
    }
};

PYBIND11_MODULE(pipeline_cpp, m) {
    m.doc() = "C++ FlexiPipe pipeline with direct Python bindings (no subprocess overhead)";
    
    py::class_<PipelineBinding>(m, "Pipeline")
        .def(py::init<>())
        .def("load_vocab", &PipelineBinding::load_vocab, "Load vocabulary file")
        .def("process_text", &PipelineBinding::process_text, 
             "Process text and return tagged sentences",
             py::arg("text"),
             py::arg("segment") = true,
             py::arg("tokenize") = true)
        .def("process_file", &PipelineBinding::process_file,
             "Process file (CoNLL-U or text) and return tagged sentences",
             py::arg("input_file"),
             py::arg("input_format") = "auto",
             py::arg("segment") = false,
             py::arg("tokenize") = false,
             py::arg("skip_split") = false,
             py::arg("skip_tag") = false,
             py::arg("skip_lemma") = false);
}
