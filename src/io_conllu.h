#ifndef IO_CONLLU_H
#define IO_CONLLU_H

#include <string>
#include <vector>
#include <fstream>
#include "types.h"

class CoNLLUReader {
public:
    // Parse a single CoNLL-U line
    static bool parse_line(const std::string& line, Token& token);
    
    // Load CoNLL-U file
    static std::vector<Sentence> load_file(const std::string& file_path);
    
    // Load CoNLL-U from string content
    static std::vector<Sentence> load_string(const std::string& content);
};

class CoNLLUWriter {
public:
    // Write sentences to CoNLL-U format
    static void write(const std::vector<Sentence>& sentences, std::ostream& out, 
                      const std::string& generator = "flexipipe", 
                      const std::string& model = "");
    
    // Write to file
    static bool write_file(const std::vector<Sentence>& sentences, const std::string& file_path,
                           const std::string& generator = "flexipipe",
                           const std::string& model = "");
};

#endif // IO_CONLLU_H

