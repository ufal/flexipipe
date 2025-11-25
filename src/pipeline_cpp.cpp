#include "pipeline_cpp.h"
#include "vocab_loader.h"
#include "types.h"
#include "io_conllu.h"
#include "io_teitok.h"
#include "tokenizer.h"
#include "normalizer.h"
#include "contractions.h"
#include "viterbi_optimized.h"
#include <unordered_set>

#include <sstream>
#include <fstream>

// Forward declaration - we'll use FlexiPipePipeline from flexipipe_cpp.cpp
// But we need to make it accessible. For now, let's duplicate the minimal needed parts
class FlexiPipePipeline {
private:
    Vocab vocab;
    VocabLoader vocab_loader;
    
public:
    bool load_vocab(const std::string& vocab_file) {
        return vocab_loader.load(vocab_file, vocab);
    }
    
    std::vector<Sentence> process_text(const std::string& text, bool segment, bool tokenize) {
        // This is a simplified version - we'll need to include the full implementation
        // For now, let's create a wrapper that calls the existing functions
        std::vector<Sentence> sentences;
        
        if (tokenize && segment) {
            // Tokenize and segment
            std::vector<std::string> sentence_texts = SentenceSegmenter::segment(text);
            for (const auto& sent_text : sentence_texts) {
                std::vector<std::string> tokens = Tokenizer::tokenize_ud_style(sent_text);
                Sentence sentence;
                sentence.text = sent_text;
                int token_id = 1;
                for (const auto& form : tokens) {
                    Token token;
                    token.form = form;
                    token.id = token_id++;
                    sentence.tokens.push_back(token);
                }
                sentences.push_back(sentence);
            }
        } else if (tokenize) {
            // Just tokenize (assume one sentence)
            std::vector<std::string> tokens = Tokenizer::tokenize_ud_style(text);
            Sentence sentence;
            sentence.text = text;
            int token_id = 1;
            for (const auto& form : tokens) {
                Token token;
                token.form = form;
                token.id = token_id++;
                sentence.tokens.push_back(token);
            }
            sentences.push_back(sentence);
        } else {
            // No tokenization - split on whitespace
            std::istringstream iss(text);
            std::string word;
            Sentence sentence;
            sentence.text = text;
            int token_id = 1;
            while (iss >> word) {
                Token token;
                token.form = word;
                token.id = token_id++;
                sentence.tokens.push_back(token);
            }
            if (!sentence.tokens.empty()) {
                sentences.push_back(sentence);
            }
        }
        
        // Now process: normalize, split contractions, tag, lemmatize
        // This is simplified - the full version is in flexipipe_cpp.cpp
        // We need to refactor that code to be reusable
        
        return sentences;
    }
    
    std::vector<Sentence> process_conllu(const std::string& content) {
        return CoNLLUReader::load_string(content);
    }
    
    std::vector<Sentence> process_teitok(const std::string& file_path) {
        return TEITOKReader::load_file(file_path);
    }
};

namespace PipelineCPP {

// Convert internal Token to Python dict
std::map<std::string, std::string> token_to_dict(const Token& token) {
    std::map<std::string, std::string> result;
    
    if (token.id > 0) {
        result["id"] = std::to_string(token.id);
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
    
    // Handle MWT
    if (token.is_mwt && token.mwt_start > 0 && token.mwt_end > 0) {
        result["_is_mwt"] = "true";
        result["id"] = std::to_string(token.mwt_start) + "-" + std::to_string(token.mwt_end);
    }
    
    return result;
}

// Convert Sentence to Python list of dicts
std::vector<std::map<std::string, std::string>> sentence_to_python(const Sentence& sentence) {
    std::vector<std::map<std::string, std::string>> result;
    
    // Track which token IDs are part of MWT (to skip them as separate tokens)
    std::unordered_set<int> mwt_token_ids;
    for (const auto& token : sentence.tokens) {
        if (token.is_mwt && token.mwt_start > 0 && token.mwt_end > 0) {
            for (int id = token.mwt_start; id <= token.mwt_end; id++) {
                mwt_token_ids.insert(id);
            }
        }
    }
    
    for (const auto& token : sentence.tokens) {
        if (token.is_mwt && token.mwt_start > 0 && token.mwt_end > 0) {
            // MWT token - output as single entry with MWT ID
            auto mwt_dict = token_to_dict(token);
            result.push_back(mwt_dict);
        } else if (!mwt_token_ids.count(token.id)) {
            // Regular token (not part of an MWT)
            result.push_back(token_to_dict(token));
        }
    }
    
    return result;
}

std::vector<std::vector<std::map<std::string, std::string>>> sentences_to_python(
    const std::vector<Sentence>& sentences) {
    std::vector<std::vector<std::map<std::string, std::string>>> result;
    result.reserve(sentences.size());
    
    for (const auto& sentence : sentences) {
        result.push_back(sentence_to_python(sentence));
    }
    
    return result;
}

std::vector<std::vector<std::map<std::string, std::string>>>>
process_text(
    const std::string& vocab_file,
    const std::string& text,
    bool segment,
    bool tokenize) {
    
    // Load vocabulary
    VocabLoader vocab_loader;
    Vocab vocab;
    if (!vocab_loader.load(vocab_file, vocab)) {
        throw std::runtime_error("Failed to load vocabulary from: " + vocab_file);
    }
    
    // Create pipeline instance
    FlexiPipePipeline pipeline;
    if (!pipeline.load_vocab(vocab_file)) {
        throw std::runtime_error("Failed to load vocabulary");
    }
    
    // Process text
    std::vector<Sentence> sentences = pipeline.process_text(text, segment, tokenize);
    
    // Convert to Python format
    return sentences_to_python(sentences);
}

std::vector<std::vector<std::map<std::string, std::string>>>
process_file(
    const std::string& vocab_file,
    const std::string& input_file,
    const std::string& input_format,
    bool segment,
    bool tokenize) {
    
    // Load vocabulary
    VocabLoader vocab_loader;
    Vocab vocab;
    if (!vocab_loader.load(vocab_file, vocab)) {
        throw std::runtime_error("Failed to load vocabulary from: " + vocab_file);
    }
    
    // Create pipeline instance
    FlexiPipePipeline pipeline;
    if (!pipeline.load_vocab(vocab_file)) {
        throw std::runtime_error("Failed to load vocabulary");
    }
    
    // Read input file
    std::ifstream file(input_file);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open input file: " + input_file);
    }
    
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    file.close();
    
    // Process based on format
    std::vector<Sentence> sentences;
    std::string format = input_format;
    
    if (format == "auto") {
        // Detect from file extension
        if (input_file.ends_with(".xml")) {
            format = "teitok";
        } else if (input_file.ends_with(".conllu") || input_file.ends_with(".conll")) {
            format = "conllu";
        } else {
            format = "text";
        }
    }
    
    if (format == "text") {
        sentences = pipeline.process_text(content, segment, tokenize);
    } else if (format == "conllu") {
        sentences = pipeline.process_conllu(content);
    } else if (format == "teitok") {
        sentences = pipeline.process_teitok(input_file);
    } else {
        throw std::runtime_error("Unknown input format: " + format);
    }
    
    // Convert to Python format
    return sentences_to_python(sentences);
}

} // namespace PipelineCPP

