#ifndef IO_TEITOK_H
#define IO_TEITOK_H

#include <string>
#include <vector>
#include "types.h"
#include "pugixml.hpp"

class TEITOKReader {
public:
    // Load TEITOK XML file
    static std::vector<Sentence> load_file(const std::string& file_path, 
                                          const std::string& normalization_attr = "reg");
    
    // Load TEITOK XML from string content
    static std::vector<Sentence> load_string(const std::string& file_path,
                                            const std::string& normalization_attr = "reg");
    
private:
    static std::string get_attr_with_fallback(const pugi::xml_node& elem, 
                                             const std::string& attr_names);
};

#endif // IO_TEITOK_H

