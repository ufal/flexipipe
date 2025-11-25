#include "flexitag/settings.h"
#include "flexitag/lexicon.h"
#include "flexitag/io_teitok.h"
#include "flexitag/tagger.h"

#include <iostream>

using namespace flexitag;

int main(int argc, char** argv) {
    try {
        auto cli_settings = parse_arguments(argc, argv);
        TaggerSettings settings;

        if (!cli_settings.settings_file.empty()) {
            settings = load_settings(cli_settings);
        } else {
            settings = cli_settings;
        }

        const std::string params_file = settings.get("params");
        if (params_file.empty()) {
            std::cerr << "No parameter file specified (--params=...)" << std::endl;
            return 1;
        }

        auto lexicon = std::make_shared<Lexicon>();
        lexicon->set_endlen(settings.get_int("endlen", 6));
        if (!lexicon->load(params_file)) {
            std::cerr << "Failed to load parameter file: " << params_file << std::endl;
            return 1;
        }
        
        // Load additional vocab files (merge mode)
        for (const auto& extra_vocab : settings.extra_vocab_files) {
            if (settings.verbose) {
                std::cerr << "Loading extra vocabulary: " << extra_vocab << std::endl;
            }
            if (!lexicon->load(extra_vocab, true)) {  // true = merge mode
                std::cerr << "Warning: Failed to load extra vocabulary: " << extra_vocab << std::endl;
            }
        }
        
        // Apply default settings from the model (if available), but don't override CLI arguments
        for (const auto& [key, value] : lexicon->default_settings()) {
            if (settings.get(key).empty()) {
                settings.options[key] = value;
            }
        }

        const std::string xmlfile = settings.xmlfile;
        if (xmlfile.empty()) {
            std::cerr << "--xmlfile option is required" << std::endl;
            return 1;
        }

        Document doc = load_teitok(xmlfile);

        FlexitagTagger tagger;
        tagger.configure(settings);
        tagger.set_lexicon(lexicon);

        TaggerStats stats;
        Document tagged = tagger.tag(doc, &stats);

        std::string outfile = settings.outfile.empty() ? xmlfile : settings.outfile;
        if (settings.test) {
            save_teitok(tagged, outfile);
        } else {
            save_teitok(tagged, outfile);
        }

        if (settings.verbose) {
            float tok_per_sec = stats.elapsed_seconds > 0.f
                ? stats.word_count / stats.elapsed_seconds
                : 0.f;
            std::cout << stats.word_count << " tokens tagged in "
                      << stats.elapsed_seconds << "s ("
                      << tok_per_sec << " tok/s)" << std::endl;
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }
}

