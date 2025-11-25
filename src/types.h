#ifndef TYPES_H
#define TYPES_H

#include <string>
#include <vector>

struct Token {
    std::string form;
    std::string lemma;
    std::string upos;
    std::string xpos;
    std::string feats;
    std::string head;
    std::string deprel;
    std::string misc;
    std::string norm_form;  // normalized form (from reg/nform)
    std::string expan;  // expansion for abbreviations
    std::vector<std::string> parts;  // for contractions (deprecated, use subtokens instead)
    int id = 0;  // ordinal position in sentence
    bool is_mwt = false;
    int mwt_start = 0;  // deprecated, use subtokens instead
    int mwt_end = 0;    // deprecated, use subtokens instead
    std::vector<Token> subtokens;  // nested subtokens for MWT (TEITOK-style)
    std::string tok_id;  // original token ID from TEITOK (XML @id or @xml:id) - separate from ordinal id
    std::string dtok_id;  // dtok ID from TEITOK
};

struct Sentence {
    std::vector<Token> tokens;
    std::string sent_id;
    std::string text;
};

#endif // TYPES_H

