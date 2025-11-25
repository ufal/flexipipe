#pragma once

#include "types.h"

#include <set>
#include <string>

namespace flexitag {

Document load_teitok(const std::string& path);
void save_teitok(const Document& doc, const std::string& path, 
                 const std::set<std::string>& custom_attributes = {},
                 bool pretty_print = false);
std::string dump_teitok(const Document& doc, 
                        const std::set<std::string>& custom_attributes = {},
                        bool pretty_print = false);

} // namespace flexitag

