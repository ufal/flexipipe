#include "tokenizer.h"
#include <regex>
#include <algorithm>
#include <sstream>

std::vector<std::string> Tokenizer::tokenize_ud_style(const std::string& text) {
    std::vector<std::string> tokens;
    
    if (text.empty()) {
        return tokens;
    }
    
    // UD-style tokenization with UTF-8 support
    // Instead of using \w (which only matches ASCII), we'll manually tokenize
    // to properly handle UTF-8 multi-byte characters
    
    size_t pos = 0;
    while (pos < text.length()) {
        // Skip whitespace
        while (pos < text.length() && std::isspace(static_cast<unsigned char>(text[pos]))) {
            pos++;
        }
        if (pos >= text.length()) {
            break;
        }
        
        size_t start = pos;
        
        // Check if we're at a punctuation character
        unsigned char c = static_cast<unsigned char>(text[pos]);
        bool is_punct = !std::isalnum(c) && c != '-' && c != '\'' && c != '_';
        
        if (is_punct) {
            // Single punctuation token
            tokens.push_back(text.substr(pos, 1));
            pos++;
        } else {
            // Word token - collect alphanumeric characters, hyphens, apostrophes, and UTF-8 chars
            // We need to handle UTF-8 properly: a UTF-8 character starts with 0x80-0xFF
            // and continuation bytes are 0x80-0xBF
            
            while (pos < text.length()) {
                c = static_cast<unsigned char>(text[pos]);
                
                // Check if this is a valid word character
                if (std::isalnum(c) || c == '-' || c == '\'' || c == '_') {
                    pos++;
                } else if ((c & 0x80) != 0) {
                    // UTF-8 multi-byte character - check if it's a valid start byte
                    // UTF-8 start bytes: 0xC0-0xFF (but not continuation bytes 0x80-0xBF)
                    if ((c & 0xC0) == 0xC0) {
                        // Valid UTF-8 start byte - count continuation bytes
                        int continuation_bytes = 0;
                        if ((c & 0xE0) == 0xC0) continuation_bytes = 1;  // 110xxxxx
                        else if ((c & 0xF0) == 0xE0) continuation_bytes = 2;  // 1110xxxx
                        else if ((c & 0xF8) == 0xF0) continuation_bytes = 3;  // 11110xxx
                        
                        // Skip the UTF-8 character
                        pos++;
                        for (int i = 0; i < continuation_bytes && pos < text.length(); i++) {
                            unsigned char cont = static_cast<unsigned char>(text[pos]);
                            if ((cont & 0xC0) == 0x80) {  // Valid continuation byte
                                pos++;
                            } else {
                                break;  // Invalid UTF-8 sequence
                            }
                        }
                    } else {
                        // Invalid UTF-8 or continuation byte outside word - stop
                        break;
                    }
                } else if (std::isspace(c)) {
                    // Whitespace - end of token
                    break;
                } else {
                    // Other punctuation - end of token
                    break;
                }
            }
            
            if (pos > start) {
                std::string token = text.substr(start, pos - start);
                if (!token.empty()) {
                    tokens.push_back(token);
                }
            }
        }
    }
    
    return tokens;
}

std::vector<std::string> Tokenizer::tokenize_whitespace(const std::string& text) {
    std::vector<std::string> tokens;
    
    if (text.empty()) {
        return tokens;
    }
    
    std::istringstream iss(text);
    std::string token;
    
    while (iss >> token) {
        tokens.push_back(token);
    }
    
    return tokens;
}

std::vector<std::string> SentenceSegmenter::segment(const std::string& text) {
    std::vector<std::string> sentences;
    
    if (text.empty()) {
        return sentences;
    }
    
    // Normalize whitespace
    std::string normalized = text;
    std::regex whitespace_pattern(R"(\s+)");
    normalized = std::regex_replace(normalized, whitespace_pattern, " ");
    
    // Trim
    normalized.erase(0, normalized.find_first_not_of(" \t"));
    normalized.erase(normalized.find_last_not_of(" \t") + 1);
    
    if (normalized.empty()) {
        return sentences;
    }
    
    // Sentence-ending punctuation
    std::regex sentence_endings(R"([.!?]+)");
    
    // Pattern: sentence ending followed by optional whitespace (or end of text)
    std::regex pattern(R"(([.!?]+)(?:\s+|$))");
    
    std::string current_sentence;
    std::sregex_iterator iter(normalized.begin(), normalized.end(), pattern);
    std::sregex_iterator end;
    
    size_t last_pos = 0;
    
    for (; iter != end; ++iter) {
        std::smatch match = *iter;
        size_t match_pos = match.position();
        size_t match_len = match.length();
        
        // Add text from last position to match
        current_sentence += normalized.substr(last_pos, match_pos - last_pos);
        current_sentence += match.str(1);  // Add the punctuation
        
        // Trim and add sentence
        std::string sentence = current_sentence;
        sentence.erase(0, sentence.find_first_not_of(" \t"));
        sentence.erase(sentence.find_last_not_of(" \t") + 1);
        
        if (!sentence.empty()) {
            sentences.push_back(sentence);
        }
        
        current_sentence.clear();
        last_pos = match_pos + match_len;
    }
    
    // Add remaining text as final sentence
    if (last_pos < normalized.length()) {
        current_sentence += normalized.substr(last_pos);
        std::string sentence = current_sentence;
        sentence.erase(0, sentence.find_first_not_of(" \t"));
        sentence.erase(sentence.find_last_not_of(" \t") + 1);
        
        if (!sentence.empty()) {
            sentences.push_back(sentence);
        }
    }
    
    // Fallback: if no sentences found, return entire text as one sentence
    if (sentences.empty()) {
        sentences.push_back(normalized);
    }
    
    return sentences;
}

