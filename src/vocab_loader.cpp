#include "vocab_loader.h"
#include <fstream>
#include <algorithm>
#include <cctype>
#include <iostream>

using namespace rapidjson;

bool VocabLoader::load(const std::string& vocab_file, Vocab& vocab) {
    std::ifstream file(vocab_file);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open vocab file: " << vocab_file << std::endl;
        return false;
    }
    
    // Read entire file into string
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    file.close();
    
    Document doc;
    doc.Parse(content.c_str());
    
    if (doc.HasParseError()) {
        std::cerr << "Error: JSON parse error at position " << doc.GetErrorOffset() 
                  << ": " << GetParseError_En(doc.GetParseError()) << std::endl;
        return false;
    }
    
    // Parse metadata
    if (doc.HasMember("metadata") && doc["metadata"].IsObject()) {
        if (!parse_metadata(doc["metadata"], vocab)) {
            std::cerr << "Warning: Failed to parse metadata" << std::endl;
        }
    }
    
    // Parse vocabulary
    if (doc.HasMember("vocab") && doc["vocab"].IsObject()) {
        if (!parse_vocab_object(doc["vocab"], vocab)) {
            std::cerr << "Error: Failed to parse vocabulary" << std::endl;
            return false;
        }
    } else {
        std::cerr << "Error: Missing or invalid 'vocab' field" << std::endl;
        return false;
    }
    
    // Parse transitions
    if (doc.HasMember("transitions") && doc["transitions"].IsObject()) {
        if (!parse_transitions(doc["transitions"], vocab)) {
            std::cerr << "Warning: Failed to parse transitions" << std::endl;
        }
    }
    
    return true;
}

bool VocabLoader::parse_vocab_object(const Value& vocab_obj, Vocab& vocab) {
    if (!vocab_obj.IsObject()) {
        return false;
    }
    
    for (auto it = vocab_obj.MemberBegin(); it != vocab_obj.MemberEnd(); ++it) {
        std::string form = it->name.GetString();
        
        const Value& value = it->value;
        
        if (value.IsObject()) {
            // Single analysis (unambiguous word)
            VocabAnalysis analysis;
            if (parse_analysis(value, analysis)) {
                vocab.entries[form].push_back(analysis);
            }
        } else if (value.IsArray()) {
            // Multiple analyses (ambiguous word)
            for (const auto& analysis_val : value.GetArray()) {
                if (analysis_val.IsObject()) {
                    VocabAnalysis analysis;
                    if (parse_analysis(analysis_val, analysis)) {
                        vocab.entries[form].push_back(analysis);
                    }
                }
            }
        }
    }
    
    return true;
}

bool VocabLoader::parse_analysis(const Value& analysis_obj, VocabAnalysis& analysis) {
    if (!analysis_obj.IsObject()) {
        return false;
    }
    
    // Parse required/optional fields
    if (analysis_obj.HasMember("upos") && analysis_obj["upos"].IsString()) {
        analysis.upos = analysis_obj["upos"].GetString();
    }
    
    if (analysis_obj.HasMember("xpos") && analysis_obj["xpos"].IsString()) {
        analysis.xpos = analysis_obj["xpos"].GetString();
    }
    
    if (analysis_obj.HasMember("lemma") && analysis_obj["lemma"].IsString()) {
        analysis.lemma = analysis_obj["lemma"].GetString();
    }
    
    if (analysis_obj.HasMember("feats") && analysis_obj["feats"].IsString()) {
        analysis.feats = analysis_obj["feats"].GetString();
    }
    
    if (analysis_obj.HasMember("reg") && analysis_obj["reg"].IsString()) {
        analysis.reg = analysis_obj["reg"].GetString();
    }
    
    if (analysis_obj.HasMember("expan") && analysis_obj["expan"].IsString()) {
        analysis.expan = analysis_obj["expan"].GetString();
    }
    
    if (analysis_obj.HasMember("count") && analysis_obj["count"].IsInt()) {
        analysis.count = analysis_obj["count"].GetInt();
    }
    
    if (analysis_obj.HasMember("parts") && analysis_obj["parts"].IsArray()) {
        for (const auto& part_val : analysis_obj["parts"].GetArray()) {
            if (part_val.IsString()) {
                analysis.parts.push_back(part_val.GetString());
            }
        }
    }
    
    return true;
}

bool VocabLoader::parse_transitions(const Value& transitions_obj, Vocab& vocab) {
    if (!transitions_obj.IsObject()) {
        return false;
    }
    
    // Parse UPOS transitions
    if (transitions_obj.HasMember("upos") && transitions_obj["upos"].IsObject()) {
        const Value& upos_trans = transitions_obj["upos"];
        for (auto it = upos_trans.MemberBegin(); it != upos_trans.MemberEnd(); ++it) {
            std::string prev_tag = it->name.GetString();
            if (it->value.IsObject()) {
                for (auto it2 = it->value.MemberBegin(); it2 != it->value.MemberEnd(); ++it2) {
                    std::string curr_tag = it2->name.GetString();
                    if (it2->value.IsNumber()) {
                        double prob = it2->value.GetDouble();
                        vocab.transitions["upos"][prev_tag][curr_tag] = prob;
                    }
                }
            }
        }
    }
    
    // Parse XPOS transitions
    if (transitions_obj.HasMember("xpos") && transitions_obj["xpos"].IsObject()) {
        const Value& xpos_trans = transitions_obj["xpos"];
        for (auto it = xpos_trans.MemberBegin(); it != xpos_trans.MemberEnd(); ++it) {
            std::string prev_tag = it->name.GetString();
            if (it->value.IsObject()) {
                for (auto it2 = it->value.MemberBegin(); it2 != it->value.MemberEnd(); ++it2) {
                    std::string curr_tag = it2->name.GetString();
                    if (it2->value.IsNumber()) {
                        double prob = it2->value.GetDouble();
                        vocab.transitions["xpos"][prev_tag][curr_tag] = prob;
                    }
                }
            }
        }
    }
    
    // Parse start probabilities
    if (transitions_obj.HasMember("start") && transitions_obj["start"].IsObject()) {
        const Value& start_obj = transitions_obj["start"];
        for (auto it = start_obj.MemberBegin(); it != start_obj.MemberEnd(); ++it) {
            std::string tag = it->name.GetString();
            if (it->value.IsNumber()) {
                double prob = it->value.GetDouble();
                vocab.start_probs[tag] = prob;
            }
        }
    }
    
    // Parse dependency transitions (if present)
    if (transitions_obj.HasMember("deprel") && transitions_obj["deprel"].IsObject()) {
        const Value& deprel_obj = transitions_obj["deprel"];
        for (auto it = deprel_obj.MemberBegin(); it != deprel_obj.MemberEnd(); ++it) {
            std::string key = it->name.GetString();
            if (it->value.IsInt()) {
                int count = it->value.GetInt();
                vocab.dep_transitions[key] = count;
            }
        }
    }
    
    return true;
}

bool VocabLoader::parse_metadata(const Value& metadata_obj, Vocab& vocab) {
    if (!metadata_obj.IsObject()) {
        return false;
    }
    
    // Store all metadata as string key-value pairs
    for (auto it = metadata_obj.MemberBegin(); it != metadata_obj.MemberEnd(); ++it) {
        std::string key = it->name.GetString();
        const Value& value = it->value;
        
        if (value.IsString()) {
            vocab.metadata[key] = value.GetString();
        } else if (value.IsNumber()) {
            vocab.metadata[key] = std::to_string(value.GetInt64());
        } else if (value.IsBool()) {
            vocab.metadata[key] = value.GetBool() ? "true" : "false";
        } else if (value.IsNull()) {
            vocab.metadata[key] = "null";
        }
        // For complex objects/arrays, we skip them for now
        // Can be extended if needed
    }
    
    // Parse capitalizable_tags if present
    if (metadata_obj.HasMember("capitalizable_tags") && 
        metadata_obj["capitalizable_tags"].IsObject()) {
        const Value& cap_tags = metadata_obj["capitalizable_tags"];
        
        if (cap_tags.HasMember("upos") && cap_tags["upos"].IsObject()) {
            const Value& upos_cap = cap_tags["upos"];
            for (auto it = upos_cap.MemberBegin(); it != upos_cap.MemberEnd(); ++it) {
                std::string tag = it->name.GetString();
                if (it->value.IsObject()) {
                    int capitalized = 0;
                    int lowercase = 0;
                    if (it->value.HasMember("capitalized") && it->value["capitalized"].IsInt()) {
                        capitalized = it->value["capitalized"].GetInt();
                    }
                    if (it->value.HasMember("lowercase") && it->value["lowercase"].IsInt()) {
                        lowercase = it->value["lowercase"].GetInt();
                    }
                    vocab.capitalizable_tags["upos"][tag] = std::make_pair(capitalized, lowercase);
                }
            }
        }
        
        if (cap_tags.HasMember("xpos") && cap_tags["xpos"].IsObject()) {
            const Value& xpos_cap = cap_tags["xpos"];
            for (auto it = xpos_cap.MemberBegin(); it != xpos_cap.MemberEnd(); ++it) {
                std::string tag = it->name.GetString();
                if (it->value.IsObject()) {
                    int capitalized = 0;
                    int lowercase = 0;
                    if (it->value.HasMember("capitalized") && it->value["capitalized"].IsInt()) {
                        capitalized = it->value["capitalized"].GetInt();
                    }
                    if (it->value.HasMember("lowercase") && it->value["lowercase"].IsInt()) {
                        lowercase = it->value["lowercase"].GetInt();
                    }
                    vocab.capitalizable_tags["xpos"][tag] = std::make_pair(capitalized, lowercase);
                }
            }
        }
    }
    
    return true;
}

std::string VocabLoader::to_lower(const std::string& s) const {
    std::string result = s;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}
