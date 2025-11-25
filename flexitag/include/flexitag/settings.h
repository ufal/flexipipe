#pragma once

#include "types.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace flexitag {

struct TaggerOption {
    std::string key;
    std::string value;
};

struct TaggerSettings {
    std::unordered_map<std::string, std::string> options;
    std::vector<TaggerOption> parameter_overrides;
    std::string pid;
    std::string training_folder;
    std::string xmlfile;
    std::string outfile;
    std::string settings_file;
    std::string lexicon_file;
    std::vector<std::string> extra_vocab_files;  // Additional vocab files to merge
    bool verbose = false;
    bool debug = false;
    bool test = false;

    std::string get(const std::string& key, const std::string& fallback = "") const;
    int get_int(const std::string& key, int fallback) const;
    float get_float(const std::string& key, float fallback) const;
    bool get_bool(const std::string& key, bool fallback) const;
};

TaggerSettings parse_arguments(int argc, char** argv);
TaggerSettings load_settings(const TaggerSettings& base);

} // namespace flexitag

