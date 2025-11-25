#include "io_teitok_writer.h"
#include "pugixml.hpp"
#include <iostream>
#include <sstream>
#include <fstream>
#include <unordered_set>

void TEITOKWriter::write(const std::vector<Sentence>& sentences, std::ostream& out) {
    // Create XML document
    pugi::xml_document doc;
    
    // Create root element (typically <TEI> or <text>)
    pugi::xml_node root = doc.append_child("TEI");
    pugi::xml_node text = root.append_child("text");
    pugi::xml_node body = text.append_child("body");
    
    // Process each sentence
    for (const auto& sentence : sentences) {
        // Create sentence element
        pugi::xml_node s_node = body.append_child("s");
        
        // Set sentence ID if available
        if (!sentence.sent_id.empty()) {
            s_node.append_attribute("id") = sentence.sent_id.c_str();
        }
        
        // Set sentence text if available
        if (!sentence.text.empty()) {
            s_node.append_attribute("text") = sentence.text.c_str();
        }
        
        // Process tokens
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
                // Multi-word token (contraction) - output as <tok> with <dtok> children
                pugi::xml_node tok_node = s_node.append_child("tok");
                
                // Set token ID if available
                if (!token.tok_id.empty()) {
                    tok_node.append_attribute("id") = token.tok_id.c_str();
                }
                
                // Set form
                tok_node.append_attribute("form") = token.form.c_str();
                
                // Add dtok elements for each part
                if (!token.parts.empty()) {
                    for (size_t i = 0; i < token.parts.size(); i++) {
                        pugi::xml_node dtok_node = tok_node.append_child("dtok");
                        
                        // Find corresponding split token
                        int split_token_id = token.mwt_start + i;
                        const Token* split_token = nullptr;
                        for (const auto& t : sentence.tokens) {
                            if (!t.is_mwt && t.id == split_token_id) {
                                split_token = &t;
                                break;
                            }
                        }
                        
                        if (split_token) {
                            // Set dtok ID if available
                            if (!split_token->dtok_id.empty()) {
                                dtok_node.append_attribute("id") = split_token->dtok_id.c_str();
                            }
                            
                            // Set form
                            dtok_node.append_attribute("form") = split_token->form.c_str();
                            
                            // Set normalization if available
                            if (!split_token->norm_form.empty() && split_token->norm_form != "_") {
                                dtok_node.append_attribute("reg") = split_token->norm_form.c_str();
                            }
                            
                            // Set lemma
                            if (!split_token->lemma.empty() && split_token->lemma != "_") {
                                dtok_node.append_attribute("lemma") = split_token->lemma.c_str();
                            }
                            
                            // Set UPOS
                            if (!split_token->upos.empty() && split_token->upos != "_") {
                                dtok_node.append_attribute("upos") = split_token->upos.c_str();
                            }
                            
                            // Set XPOS
                            if (!split_token->xpos.empty() && split_token->xpos != "_") {
                                dtok_node.append_attribute("xpos") = split_token->xpos.c_str();
                            }
                            
                            // Set FEATS
                            if (!split_token->feats.empty() && split_token->feats != "_") {
                                dtok_node.append_attribute("feats") = split_token->feats.c_str();
                            }
                            
                            // Set expansion if available
                            if (!split_token->expan.empty() && split_token->expan != "_") {
                                dtok_node.append_attribute("expan") = split_token->expan.c_str();
                            }
                        } else {
                            // Fallback: use part from parts list
                            dtok_node.append_attribute("form") = token.parts[i].c_str();
                        }
                    }
                }
            } else if (!mwt_token_ids.count(token.id)) {
                // Regular token (not part of an MWT)
                // Regular token
                pugi::xml_node tok_node = s_node.append_child("tok");
                
                // Set token ID if available
                if (!token.tok_id.empty()) {
                    tok_node.append_attribute("id") = token.tok_id.c_str();
                }
                
                // Set form (use text content or attribute)
                tok_node.append_attribute("form") = token.form.c_str();
                
                // Set normalization if available
                if (!token.norm_form.empty() && token.norm_form != "_") {
                    tok_node.append_attribute("reg") = token.norm_form.c_str();
                }
                
                // Set lemma
                if (!token.lemma.empty() && token.lemma != "_") {
                    tok_node.append_attribute("lemma") = token.lemma.c_str();
                }
                
                // Set UPOS
                if (!token.upos.empty() && token.upos != "_") {
                    tok_node.append_attribute("upos") = token.upos.c_str();
                }
                
                // Set XPOS
                if (!token.xpos.empty() && token.xpos != "_") {
                    tok_node.append_attribute("xpos") = token.xpos.c_str();
                }
                
                // Set FEATS
                if (!token.feats.empty() && token.feats != "_") {
                    tok_node.append_attribute("feats") = token.feats.c_str();
                }
                
                // Set expansion if available
                if (!token.expan.empty() && token.expan != "_") {
                    tok_node.append_attribute("expan") = token.expan.c_str();
                }
            }
        }
    }
    
    // Write XML to output stream
    doc.save(out, "  ", pugi::format_indent | pugi::format_no_declaration);
    // Add XML declaration manually
    out << "\n";
}

bool TEITOKWriter::write_file(const std::vector<Sentence>& sentences, const std::string& file_path) {
    std::ofstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open output file: " << file_path << std::endl;
        return false;
    }
    
    // Write XML declaration
    file << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
    
    write(sentences, file);
    file.close();
    return true;
}

