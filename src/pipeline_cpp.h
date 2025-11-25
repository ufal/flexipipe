#ifndef PIPELINE_CPP_H
#define PIPELINE_CPP_H

#include <string>
#include <vector>
#include <map>
#include "types.h"

// Python bindings for the full FlexiPipe pipeline
// This provides direct function calls without subprocess overhead

namespace PipelineCPP {

// Process text input and return tagged sentences
// Returns: vector of sentences, where each sentence is a vector of token maps
std::vector<std::vector<std::map<std::string, std::string>>> process_text(
    const std::string& vocab_file,
    const std::string& text,
    bool segment = true,
    bool tokenize = true
);

// Process file input
std::vector<std::vector<std::map<std::string, std::string>>> process_file(
    const std::string& vocab_file,
    const std::string& input_file,
    const std::string& input_format = "auto",
    bool segment = false,
    bool tokenize = false
);

// Convert internal Sentence/Token structures to Python-friendly format
std::vector<std::vector<std::map<std::string, std::string>>> sentences_to_python(
    const std::vector<Sentence>& sentences);

} // namespace PipelineCPP

#endif // PIPELINE_CPP_H

