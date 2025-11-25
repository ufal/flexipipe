/*
 * FlexiPipe C++ Standalone Pipeline
 * High-performance implementation of the full NLP pipeline:
 * - Tokenization, segmentation, normalization, contraction splitting
 * - Viterbi tagging, lemmatization
 * - Input: text, CoNLL-U, TEITOK XML
 * - Output: CoNLL-U format
 * 
 * Target: 5k+ tokens/second
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <sstream>
#include <memory>
#include <cmath>
#include <limits>
#include <cctype>

#include "vocab_loader.h"
#include "types.h"
#include "io_conllu.h"
#include "io_teitok.h"
#include "io_teitok_writer.h"
#include "tokenizer.h"
#include "normalizer.h"
#include "contractions.h"
#include "viterbi_optimized.h"

class FlexiPipePipeline {
private:
    Vocab vocab;
    VocabLoader vocab_loader;
    std::string tag_type = "upos";  // or "xpos"
    
    // Viterbi tagging
    std::vector<std::string> viterbi_tag(const std::vector<std::string>& sentence);
    
    // Lemmatization
    std::string lemmatize(const std::string& form, const std::string& tag, 
                       const VocabAnalysis* analysis = nullptr);
    
    // Normalization
    std::string normalize(const std::string& form);
    
    // Contraction splitting
    std::vector<std::string> split_contraction(const std::string& form, 
                                               const std::string& upos = "", 
                                               const std::string& xpos = "");
    
public:
    bool load_vocab(const std::string& vocab_file);
    bool process_file(const std::string& input_file, const std::string& output_file,
                     const std::string& input_format, const std::string& output_format,
                     bool segment, bool tokenize,
                     bool skip_split = false, bool skip_tag = false, bool skip_lemma = false);
    
    std::vector<Sentence> process_text(const std::string& text, bool segment, bool tokenize,
                                      bool skip_split = false, bool skip_tag = false, bool skip_lemma = false);
    std::vector<Sentence> process_conllu(const std::string& content,
                                         bool skip_split = false, bool skip_tag = false, bool skip_lemma = false);
    std::vector<Sentence> process_teitok(const std::string& content);
    
    void write_conllu(const std::vector<Sentence>& sentences, std::ostream& out);
    void write_teitok(const std::vector<Sentence>& sentences, std::ostream& out);
};

bool FlexiPipePipeline::load_vocab(const std::string& vocab_file) {
    return vocab_loader.load(vocab_file, vocab);
}

std::vector<std::string> FlexiPipePipeline::viterbi_tag(const std::vector<std::string>& sentence) {
    return ViterbiTagger::tag_sentence(sentence, vocab, tag_type);
}

std::string FlexiPipePipeline::normalize(const std::string& form) {
    std::string normalized = Normalizer::normalize(form, vocab, true);
    return normalized.empty() ? form : normalized;
}

std::vector<std::string> FlexiPipePipeline::split_contraction(const std::string& form, 
                                                               const std::string& upos, 
                                                               const std::string& xpos) {
    return ContractionSplitter::split(form, vocab, upos, xpos);
}

std::string FlexiPipePipeline::lemmatize(const std::string& form, const std::string& tag, 
                                        const VocabAnalysis* analysis) {
    // Simple lemmatization: use lemma from vocab if available
    if (analysis && !analysis->lemma.empty() && analysis->lemma != "_") {
        return analysis->lemma;
    }
    
    // Try to find lemma in vocab
    const std::vector<VocabAnalysis>* analyses = vocab.get(form);
    if (!analyses) {
        std::string form_lower = form;
        std::transform(form_lower.begin(), form_lower.end(), form_lower.begin(), ::tolower);
        analyses = vocab.get(form_lower);
    }
    
    if (analyses && !analyses->empty()) {
        // Find analysis with matching tag
        for (const auto& a : *analyses) {
            std::string analysis_tag = (tag_type == "upos") ? a.upos : a.xpos;
            if (analysis_tag == tag && !a.lemma.empty() && a.lemma != "_") {
                return a.lemma;
            }
        }
        // If no matching tag, use first lemma
        if (!analyses->front().lemma.empty() && analyses->front().lemma != "_") {
            return analyses->front().lemma;
        }
    }
    
    // Default: return lowercase form
    std::string lemma = form;
    std::transform(lemma.begin(), lemma.end(), lemma.begin(), ::tolower);
    return lemma;
}

std::vector<Sentence> FlexiPipePipeline::process_text(const std::string& text, bool segment, bool tokenize,
                                                      bool skip_split, bool skip_tag, bool skip_lemma) {
    std::vector<Sentence> sentences;
    
    if (segment) {
        // Segment into sentences
        std::vector<std::string> sentence_texts = SentenceSegmenter::segment(text);
        
        for (const auto& sent_text : sentence_texts) {
            Sentence sentence;
            sentence.text = sent_text;
            
            if (tokenize) {
                // Tokenize sentence
                std::vector<std::string> tokens = Tokenizer::tokenize_ud_style(sent_text);
                
                // Process each token
                for (size_t i = 0; i < tokens.size(); i++) {
                    Token token;
                    token.id = i + 1;
                    token.form = tokens[i];
                    
                    // Normalize (always done, needed for lookup even if skipping other steps)
                    std::string normalized = normalize(token.form);
                    if (normalized != token.form) {
                        token.norm_form = normalized;
                    }
                    
                    // Check for contraction splitting (skip if requested)
                    if (!skip_split) {
                        std::string form_to_check = normalized.empty() ? token.form : normalized;
                        std::vector<std::string> parts = split_contraction(form_to_check);
                        
                        if (!parts.empty() && parts.size() > 1) {
                        // Split contraction into multiple tokens
                        // Calculate IDs: MWT spans from next_id to next_id + parts.size() - 1
                        int next_id = sentence.tokens.size() + 1;
                        int mwt_start = next_id;
                        int mwt_end = next_id + parts.size() - 1;
                        
                        // First token is the orthographic form (MWT)
                        Token mwt_token = token;
                        mwt_token.is_mwt = true;
                        mwt_token.mwt_start = mwt_start;
                        mwt_token.mwt_end = mwt_end;
                        mwt_token.parts = parts;
                        sentence.tokens.push_back(mwt_token);
                        
                        // Add split tokens with correct IDs
                        for (size_t j = 0; j < parts.size(); j++) {
                            Token split_token;
                            split_token.id = mwt_start + j;
                            split_token.form = parts[j];
                            sentence.tokens.push_back(split_token);
                        }
                    } else {
                        sentence.tokens.push_back(token);
                    }
                    } else {
                        // Skip splitting - just add token as-is
                        sentence.tokens.push_back(token);
                    }
                }
            } else {
                // Assume whitespace-separated tokens
                std::vector<std::string> tokens = Tokenizer::tokenize_whitespace(sent_text);
                for (size_t i = 0; i < tokens.size(); i++) {
                    Token token;
                    token.id = i + 1;
                    token.form = tokens[i];
                    sentence.tokens.push_back(token);
                }
            }
            
            // Tag the sentence (skip if requested)
            if (!sentence.tokens.empty() && !skip_tag) {
                // Build forms array, skipping MWT tokens (only tag the actual split tokens)
                std::vector<std::string> forms;
                std::vector<size_t> token_indices;  // Map form index to token index
                for (size_t i = 0; i < sentence.tokens.size(); i++) {
                    if (!sentence.tokens[i].is_mwt) {
                        forms.push_back(sentence.tokens[i].norm_form.empty() ? 
                                       sentence.tokens[i].form : sentence.tokens[i].norm_form);
                        token_indices.push_back(i);
                    }
                }
                
                // Tag with UPOS
                std::vector<std::string> upos_tags = viterbi_tag(forms);
                
                // Tag with XPOS (if available)
                tag_type = "xpos";
                std::vector<std::string> xpos_tags = viterbi_tag(forms);
                tag_type = "upos";  // Reset
                
                // Assign tags, FEATS, and lemmatize (only to non-MWT tokens)
                for (size_t form_idx = 0; form_idx < forms.size() && form_idx < upos_tags.size(); form_idx++) {
                    size_t i = token_indices[form_idx];
                    sentence.tokens[i].upos = upos_tags[form_idx];
                    if (form_idx < xpos_tags.size()) {
                        sentence.tokens[i].xpos = xpos_tags[form_idx];
                    }
                    
                    // Find analysis for lemmatization and FEATS
                    std::string form_to_lookup = sentence.tokens[i].norm_form.empty() ? 
                                                sentence.tokens[i].form : sentence.tokens[i].norm_form;
                    const std::vector<VocabAnalysis>* analyses = vocab.get(form_to_lookup);
                    if (!analyses) {
                        std::string form_lower = form_to_lookup;
                        std::transform(form_lower.begin(), form_lower.end(), form_lower.begin(), ::tolower);
                        analyses = vocab.get(form_lower);
                    }
                    
                    const VocabAnalysis* analysis = nullptr;
                    if (analyses && !analyses->empty()) {
                        // Find best matching analysis (prefer UPOS match, then XPOS, then most frequent)
                        int best_count = 0;
                        for (const auto& a : *analyses) {
                            bool upos_match = (a.upos == upos_tags[form_idx] && !upos_tags[form_idx].empty() && upos_tags[form_idx] != "_");
                            bool xpos_match = (form_idx < xpos_tags.size() && a.xpos == xpos_tags[form_idx] && !xpos_tags[form_idx].empty() && xpos_tags[form_idx] != "_");
                            
                            if (upos_match && xpos_match) {
                                // Perfect match
                                analysis = &a;
                                break;
                            } else if (upos_match && !analysis) {
                                // UPOS match
                                analysis = &a;
                            } else if (!analysis && a.count > best_count) {
                                // Most frequent
                                best_count = a.count;
                                analysis = &a;
                            }
                        }
                        if (!analysis) {
                            analysis = &analyses->front();
                        }
                    }
                    
                    if (analysis) {
                        if (!skip_lemma) {
                            sentence.tokens[i].lemma = lemmatize(form_to_lookup, upos_tags[form_idx], analysis);
                        }
                        if (!analysis->feats.empty() && analysis->feats != "_") {
                            sentence.tokens[i].feats = analysis->feats;
                        }
                        if (!analysis->xpos.empty() && analysis->xpos != "_" && sentence.tokens[i].xpos.empty()) {
                            sentence.tokens[i].xpos = analysis->xpos;
                        }
                    } else {
                        if (!skip_lemma) {
                            sentence.tokens[i].lemma = lemmatize(form_to_lookup, upos_tags[form_idx], nullptr);
                        }
                    }
                }
                
                // Recalculate token IDs to account for MWT
                // First pass: identify which tokens are split tokens (come after MWT)
                std::unordered_set<size_t> split_token_indices;
                for (size_t i = 0; i < sentence.tokens.size(); i++) {
                    if (sentence.tokens[i].is_mwt) {
                        // Mark the next N tokens as split tokens (where N = parts.size())
                        for (size_t j = 1; j <= sentence.tokens[i].parts.size() && i + j < sentence.tokens.size(); j++) {
                            split_token_indices.insert(i + j);
                        }
                    }
                }
                
                // Second pass: assign IDs
                int current_id = 1;
                for (size_t i = 0; i < sentence.tokens.size(); i++) {
                    if (sentence.tokens[i].is_mwt) {
                        // MWT token gets range ID
                        sentence.tokens[i].mwt_start = current_id;
                        sentence.tokens[i].mwt_end = current_id + sentence.tokens[i].parts.size() - 1;
                        // Assign IDs to split tokens that follow
                        for (size_t j = 1; j <= sentence.tokens[i].parts.size() && i + j < sentence.tokens.size(); j++) {
                            if (split_token_indices.count(i + j)) {
                                sentence.tokens[i + j].id = current_id + j - 1;
                            }
                        }
                        current_id += sentence.tokens[i].parts.size();
                        // Skip the split tokens in the main loop
                        i += sentence.tokens[i].parts.size();
                    } else if (!split_token_indices.count(i)) {
                        // Regular token (not a split token)
                        sentence.tokens[i].id = current_id++;
                    }
                }
            }
            
            sentences.push_back(sentence);
        }
    } else {
        // Treat entire text as one sentence
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
    
    return sentences;
}

std::vector<Sentence> FlexiPipePipeline::process_conllu(const std::string& content,
                                                        bool skip_split, bool skip_tag, bool skip_lemma) {
    // Load CoNLL-U file
    std::vector<Sentence> sentences = CoNLLUReader::load_string(content);
    
    // Process each sentence: apply splitting, tagging, lemmatization if not skipped
    for (auto& sentence : sentences) {
        if (sentence.tokens.empty()) {
            continue;
        }
        
        // Apply contraction splitting if not skipped
        if (!skip_split) {
            std::vector<Token> new_tokens;
            for (size_t i = 0; i < sentence.tokens.size(); i++) {
                const auto& token = sentence.tokens[i];
                
                // If it's an existing MWT, split it (don't keep as MWT)
                if (token.is_mwt) {
                    // Extract form from MWT token (use the orthographic form)
                    std::string form = token.form;
                    if (form.empty() && !token.parts.empty()) {
                        // If form is empty, reconstruct from parts (e.g., "in" + "dem" -> "im")
                        form = token.parts[0];
                        for (size_t p = 1; p < token.parts.size(); p++) {
                            form += token.parts[p];
                        }
                    }
                    
                    // Try to split the contraction
                    std::string form_to_check = token.norm_form.empty() ? form : token.norm_form;
                    std::vector<std::string> parts = split_contraction(form_to_check);
                    
                    if (!parts.empty() && parts.size() > 1) {
                        // Split the existing MWT
                        int next_id = new_tokens.size() + 1;
                        int mwt_start = next_id;
                        int mwt_end = next_id + parts.size() - 1;
                        
                        // Create new MWT token
                        Token mwt_token = token;
                        mwt_token.is_mwt = true;
                        mwt_token.mwt_start = mwt_start;
                        mwt_token.mwt_end = mwt_end;
                        mwt_token.parts = parts;
                        mwt_token.form = form;  // Keep original orthographic form
                        new_tokens.push_back(mwt_token);
                        
                        // Add split tokens
                        for (size_t j = 0; j < parts.size(); j++) {
                            Token split_token = token;
                            split_token.id = mwt_start + j;
                            split_token.form = parts[j];
                            split_token.is_mwt = false;  // Split tokens are not MWTs
                            split_token.mwt_start = 0;
                            split_token.mwt_end = 0;
                            split_token.parts.clear();
                            new_tokens.push_back(split_token);
                        }
                    } else {
                        // Can't split - keep as regular token (not MWT)
                        Token regular_token = token;
                        regular_token.is_mwt = false;
                        regular_token.mwt_start = 0;
                        regular_token.mwt_end = 0;
                        regular_token.parts.clear();
                        new_tokens.push_back(regular_token);
                    }
                    continue;
                }
                
                // Check if this token is already part of an MWT (by checking if there's an MWT token before it)
                // This should not happen if we're splitting all MWTs, but check anyway
                bool is_part_of_existing_mwt = false;
                for (const auto& existing_token : sentence.tokens) {
                    if (existing_token.is_mwt && existing_token.mwt_start > 0 && existing_token.mwt_end > 0) {
                        if (token.id >= existing_token.mwt_start && token.id <= existing_token.mwt_end) {
                            is_part_of_existing_mwt = true;
                            break;
                        }
                    }
                }
                if (is_part_of_existing_mwt) {
                    // This token is already part of an MWT - skip it (will be handled when we process the MWT)
                    continue;
                }
                
                // Regular token - check if it needs splitting
                std::string form = token.norm_form.empty() ? token.form : token.norm_form;
                std::vector<std::string> parts = split_contraction(form);
                
                if (!parts.empty() && parts.size() > 1) {
                    // Split contraction
                    int next_id = new_tokens.size() + 1;
                    int mwt_start = next_id;
                    int mwt_end = next_id + parts.size() - 1;
                    
                    Token mwt_token = token;
                    mwt_token.is_mwt = true;
                    mwt_token.mwt_start = mwt_start;
                    mwt_token.mwt_end = mwt_end;
                    mwt_token.parts = parts;
                    new_tokens.push_back(mwt_token);
                    
                    for (size_t j = 0; j < parts.size(); j++) {
                        Token split_token = token;
                        split_token.id = mwt_start + j;
                        split_token.form = parts[j];
                        split_token.is_mwt = false;
                        split_token.mwt_start = 0;
                        split_token.mwt_end = 0;
                        split_token.parts.clear();
                        new_tokens.push_back(split_token);
                    }
                } else {
                    new_tokens.push_back(token);
                }
            }
            sentence.tokens = new_tokens;
            
            // Renumber all tokens sequentially to fix IDs after splitting
            // Structure: [MWT_token, split1, split2, next_regular_token, ...]
            int current_id = 1;
            for (size_t i = 0; i < sentence.tokens.size(); i++) {
                if (sentence.tokens[i].is_mwt) {
                    // Update MWT range based on current_id
                    int mwt_size = sentence.tokens[i].parts.size();
                    sentence.tokens[i].mwt_start = current_id;
                    sentence.tokens[i].mwt_end = current_id + mwt_size - 1;
                    
                    // Assign IDs to the split tokens that immediately follow this MWT
                    for (size_t j = i + 1; j < sentence.tokens.size() && (j - i - 1) < mwt_size; j++) {
                        if (!sentence.tokens[j].is_mwt) {
                            sentence.tokens[j].id = current_id + (j - i - 1);
                        }
                    }
                    
                    // Move current_id past the MWT range
                    current_id += mwt_size;
                    // Skip the split tokens in the main loop (they're already assigned)
                    i += mwt_size;
                } else {
                    // Regular token (not part of an MWT)
                    sentence.tokens[i].id = current_id++;
                }
            }
        } else {
            // skip_split=true: Keep MWTs as-is, but ensure split tokens are skipped during tagging
            // (This is already handled in the tagging loop at line 418)
        }
        
        // Apply tagging if not skipped
        // If respect_existing is enabled (skip_tag=false but we should check existing),
        // we should only tag if tags are missing
        if (!skip_tag) {
            // Build forms array for tagging (only tokens that need tagging)
            std::vector<std::string> forms;
            std::vector<size_t> token_indices;
            for (size_t i = 0; i < sentence.tokens.size(); i++) {
                if (!sentence.tokens[i].is_mwt) {
                    // Check if tags already exist (respect existing)
                    // For now, we'll tag everything - respect_existing logic will be handled in Python
                    // TODO: Add respect_existing parameter to C++ pipeline
                    std::string form = sentence.tokens[i].norm_form.empty() ? 
                                      sentence.tokens[i].form : sentence.tokens[i].norm_form;
                    forms.push_back(form);
                    token_indices.push_back(i);
                }
            }
            
            if (!forms.empty()) {
                // Tag with UPOS
                std::vector<std::string> upos_tags = viterbi_tag(forms);
                
                // Tag with XPOS
                tag_type = "xpos";
                std::vector<std::string> xpos_tags = viterbi_tag(forms);
                tag_type = "upos";
                
                // Assign tags and FEATS (only if missing, to respect existing)
                for (size_t form_idx = 0; form_idx < forms.size() && form_idx < upos_tags.size(); form_idx++) {
                    size_t i = token_indices[form_idx];
                    // Only overwrite if tag is missing (respect existing)
                    if (sentence.tokens[i].upos.empty() || sentence.tokens[i].upos == "_") {
                        sentence.tokens[i].upos = upos_tags[form_idx];
                    }
                    if (form_idx < xpos_tags.size()) {
                        if (sentence.tokens[i].xpos.empty() || sentence.tokens[i].xpos == "_") {
                            sentence.tokens[i].xpos = xpos_tags[form_idx];
                        }
                    }
                    
                    // Find analysis for FEATS
                    std::string form_to_lookup = sentence.tokens[i].norm_form.empty() ? 
                                                sentence.tokens[i].form : sentence.tokens[i].norm_form;
                    const std::vector<VocabAnalysis>* analyses = vocab.get(form_to_lookup);
                    if (!analyses) {
                        std::string form_lower = form_to_lookup;
                        std::transform(form_lower.begin(), form_lower.end(), form_lower.begin(), ::tolower);
                        analyses = vocab.get(form_lower);
                    }
                    
                    if (analyses && !analyses->empty()) {
                        // Find best matching analysis
                        const VocabAnalysis* analysis = nullptr;
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
                        
                        if (analysis) {
                            // Only overwrite FEATS if missing (respect existing)
                            if ((sentence.tokens[i].feats.empty() || sentence.tokens[i].feats == "_") &&
                                !analysis->feats.empty() && analysis->feats != "_") {
                                sentence.tokens[i].feats = analysis->feats;
                            }
                        }
                    }
                }
            }
        }
        
        // Apply lemmatization if not skipped
        if (!skip_lemma) {
            for (auto& token : sentence.tokens) {
                if (token.is_mwt) {
                    continue;  // Skip MWT tokens
                }
                
                if (token.lemma.empty() || token.lemma == "_") {
                    // Only lemmatize if lemma is missing
                    std::string form = token.norm_form.empty() ? token.form : token.norm_form;
                    std::string upos = token.upos.empty() ? "_" : token.upos;
                    
                    const std::vector<VocabAnalysis>* analyses = vocab.get(form);
                    if (!analyses) {
                        std::string form_lower = form;
                        std::transform(form_lower.begin(), form_lower.end(), form_lower.begin(), ::tolower);
                        analyses = vocab.get(form_lower);
                    }
                    
                    const VocabAnalysis* analysis = nullptr;
                    if (analyses && !analyses->empty()) {
                        // Find best matching analysis
                        for (const auto& a : *analyses) {
                            if (a.upos == upos && !upos.empty() && upos != "_") {
                                analysis = &a;
                                break;
                            }
                        }
                        if (!analysis) {
                            analysis = &analyses->front();
                        }
                    }
                    
                    token.lemma = lemmatize(form, upos, analysis);
                }
            }
        }
        
        // Recalculate token IDs if splitting was applied
        if (!skip_split) {
            int current_id = 1;
            std::unordered_set<size_t> split_token_indices;
            for (size_t i = 0; i < sentence.tokens.size(); i++) {
                if (sentence.tokens[i].is_mwt) {
                    for (size_t j = 1; j <= sentence.tokens[i].parts.size() && i + j < sentence.tokens.size(); j++) {
                        split_token_indices.insert(i + j);
                    }
                }
            }
            
            for (size_t i = 0; i < sentence.tokens.size(); i++) {
                if (sentence.tokens[i].is_mwt) {
                    sentence.tokens[i].mwt_start = current_id;
                    sentence.tokens[i].mwt_end = current_id + sentence.tokens[i].parts.size() - 1;
                    for (size_t j = 1; j <= sentence.tokens[i].parts.size() && i + j < sentence.tokens.size(); j++) {
                        if (split_token_indices.count(i + j)) {
                            sentence.tokens[i + j].id = current_id + j - 1;
                        }
                    }
                    current_id += sentence.tokens[i].parts.size();
                    i += sentence.tokens[i].parts.size();
                } else if (!split_token_indices.count(i)) {
                    sentence.tokens[i].id = current_id++;
                }
            }
        } else {
            // Just ensure sequential ids
            int current_id = 1;
            for (auto& token : sentence.tokens) {
                if (!token.is_mwt) {
                    token.id = current_id++;
                }
            }
        }
    }
    
    return sentences;
}

std::vector<Sentence> FlexiPipePipeline::process_teitok(const std::string& file_path) {
    return TEITOKReader::load_file(file_path);
}

void FlexiPipePipeline::write_conllu(const std::vector<Sentence>& sentences, std::ostream& out) {
    CoNLLUWriter::write(sentences, out);
}

void FlexiPipePipeline::write_teitok(const std::vector<Sentence>& sentences, std::ostream& out) {
    TEITOKWriter::write(sentences, out);
}

bool FlexiPipePipeline::process_file(const std::string& input_file, const std::string& output_file,
                                     const std::string& input_format, const std::string& output_format,
                                     bool segment, bool tokenize,
                                     bool skip_split, bool skip_tag, bool skip_lemma) {
    std::vector<Sentence> sentences;
    
    // Detect input format if auto
    std::string format = input_format;
    if (format == "auto") {
        if (input_file == "-") {
            format = "text";  // Default to text for stdin
        } else if (input_file.find(".conllu") != std::string::npos) {
            format = "conllu";
        } else if (input_file.find(".xml") != std::string::npos || input_file.find(".teitok") != std::string::npos) {
            format = "teitok";
        } else {
            format = "text";
        }
    }
    
    // For text input, automatically enable tokenization and segmentation if not already set
    if (format == "text") {
        if (!tokenize) {
            tokenize = true;  // Text must be tokenized
        }
        if (!segment) {
            segment = true;  // Text should be segmented into sentences
        }
    }
    
    // Load input
    if (format == "conllu") {
        if (input_file == "-") {
            // Read from stdin
            std::string content((std::istreambuf_iterator<char>(std::cin)),
                               std::istreambuf_iterator<char>());
            sentences = process_conllu(content, skip_split, skip_tag, skip_lemma);
        } else {
            std::ifstream file(input_file);
            if (!file.is_open()) {
                std::cerr << "Error: Cannot open input file: " << input_file << std::endl;
                return false;
            }
            std::string content((std::istreambuf_iterator<char>(file)),
                              std::istreambuf_iterator<char>());
            file.close();
            sentences = process_conllu(content, skip_split, skip_tag, skip_lemma);
        }
    } else if (format == "teitok") {
        if (input_file == "-") {
            std::cerr << "Error: TEITOK format not supported from stdin" << std::endl;
            return false;
        }
        sentences = TEITOKReader::load_file(input_file);
    } else {
        // Text format
        std::string content;
        if (input_file == "-") {
            // Read from stdin
            content = std::string((std::istreambuf_iterator<char>(std::cin)),
                                 std::istreambuf_iterator<char>());
        } else {
            std::ifstream file(input_file);
            if (!file.is_open()) {
                std::cerr << "Error: Cannot open input file: " << input_file << std::endl;
                return false;
            }
            content = std::string((std::istreambuf_iterator<char>(file)),
                                 std::istreambuf_iterator<char>());
            file.close();
        }
        
        sentences = process_text(content, segment, tokenize, skip_split, skip_tag, skip_lemma);
    }
    
    // Write output
    std::string out_format = output_format.empty() ? "conllu" : output_format;
    
    if (output_file == "-" || output_file.empty()) {
        if (out_format == "teitok") {
            // For stdout, write XML declaration first
            std::cout << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
            write_teitok(sentences, std::cout);
        } else {
            write_conllu(sentences, std::cout);
        }
    } else {
        std::ofstream out_file(output_file);
        if (!out_file.is_open()) {
            std::cerr << "Error: Cannot open output file: " << output_file << std::endl;
            return false;
        }
        
        if (out_format == "teitok") {
            // Write XML declaration
            out_file << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
            write_teitok(sentences, out_file);
        } else {
            write_conllu(sentences, out_file);
        }
        out_file.close();
    }
    
    return true;
}

// Main entry point
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: flexipipe_cpp <vocab_file> <input_file> [options]" << std::endl;
        std::cerr << "Options:" << std::endl;
        std::cerr << "  --output <file>     Output file (default: stdout)" << std::endl;
        std::cerr << "  --input-format <format>  Input format: text, conllu, teitok (default: auto)" << std::endl;
        std::cerr << "  --output-format <format>  Output format: conllu, teitok (default: conllu)" << std::endl;
        std::cerr << "  --segment          Segment text into sentences" << std::endl;
        std::cerr << "  --tokenize         Tokenize sentences" << std::endl;
        std::cerr << "  --skip-split       Skip contraction splitting" << std::endl;
        std::cerr << "  --skip-tag         Skip tagging (UPOS/XPOS/FEATS)" << std::endl;
        std::cerr << "  --skip-lemma       Skip lemmatization" << std::endl;
        return 1;
    }
    
    std::string vocab_file = argv[1];
    std::string input_file = argv[2];
    std::string output_file = "-";  // stdout by default
    std::string input_format = "auto";
    std::string output_format = "conllu";  // default to CoNLL-U
    bool segment = false;
    bool tokenize = false;
    bool skip_split = false;
    bool skip_tag = false;
    bool skip_lemma = false;
    
    // Parse arguments
    for (int i = 3; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--output" && i + 1 < argc) {
            output_file = argv[++i];
        } else if (arg == "--input-format" && i + 1 < argc) {
            input_format = argv[++i];
        } else if (arg == "--output-format" && i + 1 < argc) {
            output_format = argv[++i];
        } else if (arg == "--segment") {
            segment = true;
        } else if (arg == "--tokenize") {
            tokenize = true;
        } else if (arg == "--skip-split") {
            skip_split = true;
        } else if (arg == "--skip-tag") {
            skip_tag = true;
        } else if (arg == "--skip-lemma") {
            skip_lemma = true;
        }
    }
    
    FlexiPipePipeline pipeline;
    if (!pipeline.load_vocab(vocab_file)) {
        std::cerr << "Failed to load vocabulary" << std::endl;
        return 1;
    }
    
    if (!pipeline.process_file(input_file, output_file, input_format, output_format, segment, tokenize, skip_split, skip_tag, skip_lemma)) {
        std::cerr << "Failed to process file" << std::endl;
        return 1;
    }
    
    return 0;
}


