#include "io_teitok.h"
#include "pugixml.hpp"
#include <iostream>
#include <sstream>
#include <algorithm>

std::string TEITOKReader::get_attr_with_fallback(const pugi::xml_node& elem, 
                                                 const std::string& attr_names) {
    if (attr_names.empty()) {
        return "";
    }
    
    std::istringstream iss(attr_names);
    std::string name;
    while (std::getline(iss, name, ',')) {
        // Trim whitespace
        name.erase(0, name.find_first_not_of(" \t"));
        name.erase(name.find_last_not_of(" \t") + 1);
        
        if (!name.empty()) {
            pugi::xml_attribute attr = elem.attribute(name.c_str());
            if (attr) {
                return attr.value();
            }
        }
    }
    return "";
}

std::vector<Sentence> TEITOKReader::load_file(const std::string& file_path,
                                              const std::string& normalization_attr) {
    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file(file_path.c_str());
    
    if (!result) {
        std::cerr << "Error: Cannot parse TEITOK XML file: " << file_path 
                  << " - " << result.description() << std::endl;
        return {};
    }
    
    return load_string(file_path, normalization_attr);
}

std::vector<Sentence> TEITOKReader::load_string(const std::string& file_path,
                                                const std::string& normalization_attr) {
    std::vector<Sentence> sentences;
    
    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file(file_path.c_str());
    
    if (!result) {
        std::cerr << "Error: Cannot parse TEITOK XML file: " << file_path 
                  << " - " << result.description() << std::endl;
        return {};
    }
    
    pugi::xml_node root = doc.root();
    
    // Find all sentence elements using XPath
    pugi::xpath_node_set sentence_nodes = root.select_nodes(".//s");
    
    for (pugi::xpath_node xp_node : sentence_nodes) {
        pugi::xml_node s_node = xp_node.node();
        Sentence sentence;
        int token_num = 1;
        
        // Get sentence ID: try @id first, then @xml:id
        std::string sentence_id = s_node.attribute("id").value();
        if (sentence_id.empty()) {
            sentence_id = s_node.attribute("xml:id").value();
        }
        sentence.sent_id = sentence_id;
        
        // Try to get original text from sentence element
        std::string original_text = s_node.text().as_string();
        if (original_text.empty()) {
            original_text = s_node.attribute("text").value();
        }
        sentence.text = original_text;
        
        // Find all token elements
        for (pugi::xml_node tok : s_node.children("tok")) {
            std::string tok_id = tok.attribute("id").value();
            if (tok_id.empty()) {
                tok_id = tok.attribute("xml:id").value();
            }
            
            // Check for contractions (dtok elements)
            std::vector<pugi::xml_node> dtoks;
            for (pugi::xml_node dtok : tok.children("dtok")) {
                dtoks.push_back(dtok);
            }
            
            if (!dtoks.empty()) {
                // Contraction: collect split forms
                std::vector<std::string> split_forms;
                std::vector<Token> dtok_tokens;
                
                for (const auto& dt : dtoks) {
                    std::string dt_id = dt.attribute("id").value();
                    if (dt_id.empty()) {
                        dt_id = dt.attribute("xml:id").value();
                    }
                    
                    std::string form = dt.attribute("form").value();
                    if (form.empty()) {
                        form = dt.text().as_string();
                        // Trim whitespace
                        form.erase(0, form.find_first_not_of(" \t\n\r"));
                        form.erase(form.find_last_not_of(" \t\n\r") + 1);
                    }
                    
                    if (!form.empty()) {
                        split_forms.push_back(form);
                        
                        // Get normalization
                        std::string nform = get_attr_with_fallback(dt, normalization_attr);
                        if (nform.empty()) {
                            nform = dt.attribute("reg").value();
                            if (nform.empty()) {
                                nform = dt.attribute("nform").value();
                            }
                        }
                        
                        // Get xpos
                        std::string xpos_val = dt.attribute("xpos").value();
                        
                        // Get expan
                        std::string expan_val = dt.attribute("expan").value();
                        if (expan_val.empty()) {
                            expan_val = dt.attribute("fform").value();
                        }
                        
                        Token dtok_token;
                        dtok_token.id = token_num++;
                        dtok_token.form = form;
                        dtok_token.norm_form = nform.empty() ? "_" : nform;
                        dtok_token.lemma = dt.attribute("lemma").value();
                        if (dtok_token.lemma.empty()) {
                            dtok_token.lemma = "_";
                        }
                        dtok_token.upos = dt.attribute("upos").value();
                        if (dtok_token.upos.empty()) {
                            dtok_token.upos = "_";
                        }
                        dtok_token.xpos = xpos_val.empty() ? "_" : xpos_val;
                        dtok_token.feats = dt.attribute("feats").value();
                        if (dtok_token.feats.empty()) {
                            dtok_token.feats = "_";
                        }
                        dtok_token.head = dt.attribute("head").value();
                        if (dtok_token.head.empty()) {
                            dtok_token.head = "0";
                        }
                        dtok_token.deprel = dt.attribute("deprel").value();
                        if (dtok_token.deprel.empty()) {
                            dtok_token.deprel = "_";
                        }
                        dtok_token.tok_id = tok_id;
                        dtok_token.dtok_id = dt_id;
                        dtok_token.expan = expan_val.empty() ? "_" : expan_val;
                        
                        dtok_tokens.push_back(dtok_token);
                    }
                }
                
                // Add orthographic form if we have split forms
                if (split_forms.size() > 1) {
                    std::string ortho_form = tok.attribute("form").value();
                    if (ortho_form.empty()) {
                        ortho_form = tok.text().as_string();
                        ortho_form.erase(0, ortho_form.find_first_not_of(" \t\n\r"));
                        ortho_form.erase(ortho_form.find_last_not_of(" \t\n\r") + 1);
                    }
                    
                    if (!ortho_form.empty()) {
                        Token ortho_token;
                        ortho_token.id = token_num++;
                        ortho_token.form = ortho_form;
                        ortho_token.norm_form = "_";
                        ortho_token.lemma = "_";
                        ortho_token.upos = "_";
                        ortho_token.xpos = "_";
                        ortho_token.feats = "_";
                        ortho_token.head = "0";
                        ortho_token.deprel = "_";
                        ortho_token.tok_id = tok_id;
                        ortho_token.parts = split_forms;
                        ortho_token.expan = "_";
                        
                        sentence.tokens.push_back(ortho_token);
                    }
                }
                
                // Add all dtok tokens
                for (const auto& dtok_token : dtok_tokens) {
                    sentence.tokens.push_back(dtok_token);
                }
            } else {
                // Regular token (no contraction)
                std::string form = tok.attribute("form").value();
                if (form.empty()) {
                    form = tok.text().as_string();
                    form.erase(0, form.find_first_not_of(" \t\n\r"));
                    form.erase(form.find_last_not_of(" \t\n\r") + 1);
                }
                
                if (!form.empty()) {
                    Token token;
                    token.id = token_num++;
                    token.form = form;
                    
                    // Get normalization
                    std::string nform = get_attr_with_fallback(tok, normalization_attr);
                    if (nform.empty()) {
                        nform = tok.attribute("reg").value();
                        if (nform.empty()) {
                            nform = tok.attribute("nform").value();
                        }
                    }
                    token.norm_form = nform.empty() ? "_" : nform;
                    
                    token.lemma = tok.attribute("lemma").value();
                    if (token.lemma.empty()) {
                        token.lemma = "_";
                    }
                    token.upos = tok.attribute("upos").value();
                    if (token.upos.empty()) {
                        token.upos = "_";
                    }
                    token.xpos = tok.attribute("xpos").value();
                    if (token.xpos.empty()) {
                        token.xpos = "_";
                    }
                    token.feats = tok.attribute("feats").value();
                    if (token.feats.empty()) {
                        token.feats = "_";
                    }
                    token.head = tok.attribute("head").value();
                    if (token.head.empty()) {
                        token.head = "0";
                    }
                    token.deprel = tok.attribute("deprel").value();
                    if (token.deprel.empty()) {
                        token.deprel = "_";
                    }
                    token.tok_id = tok_id;
                    
                    // Get expan
                    std::string expan_val = tok.attribute("expan").value();
                    if (expan_val.empty()) {
                        expan_val = tok.attribute("fform").value();
                    }
                    token.expan = expan_val.empty() ? "_" : expan_val;
                    
                    sentence.tokens.push_back(token);
                }
            }
        }
        
        if (!sentence.tokens.empty()) {
            sentences.push_back(sentence);
        }
    }
    
    return sentences;
}

