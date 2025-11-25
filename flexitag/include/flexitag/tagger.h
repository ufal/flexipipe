#pragma once

#include "types.h"
#include "settings.h"
#include "lexicon.h"
#include "normalizer.h"

#include <vector>
#include <memory>

namespace flexitag {

struct TaggerStats {
    int word_count = 0;
    int oov_count = 0;
    float elapsed_seconds = 0.f;
};

class FlexitagTagger {
public:
    FlexitagTagger();

    void configure(const TaggerSettings& settings);
    void set_lexicon(std::shared_ptr<Lexicon> lexicon);

    Document tag(const Document& doc, TaggerStats* stats = nullptr);

private:
    TaggerSettings settings_;
    std::shared_ptr<Lexicon> lexicon_;
    std::unique_ptr<Normalizer> normalizer_;

    std::vector<WordCandidate> morpho_parse(Token& token) const;
    std::vector<WordCandidate> apply_clitics(const Token& token) const;
    void update_token(Token& token, const WordCandidate& best) const;
};

} // namespace flexitag

