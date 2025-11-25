#include "flexitag/settings.h"

#include <pugixml.hpp>

#include <stdexcept>
#include <sstream>
#include <iostream>

namespace flexitag {

std::string TaggerSettings::get(const std::string& key, const std::string& fallback) const {
    auto it = options.find(key);
    return it == options.end() ? fallback : it->second;
}

int TaggerSettings::get_int(const std::string& key, int fallback) const {
    auto it = options.find(key);
    if (it == options.end()) {
        return fallback;
    }
    return std::stoi(it->second);
}

float TaggerSettings::get_float(const std::string& key, float fallback) const {
    auto it = options.find(key);
    if (it == options.end()) {
        return fallback;
    }
    return std::stof(it->second);
}

bool TaggerSettings::get_bool(const std::string& key, bool fallback) const {
    auto it = options.find(key);
    if (it == options.end()) {
        return fallback;
    }
    const std::string& val = it->second;
    return val == "1" || val == "true" || val == "TRUE" || val == "yes";
}

namespace {

void push_option(TaggerSettings& settings, const std::string& key, const std::string& value) {
    settings.options[key] = value;
    if (key == "pid") {
        settings.pid = value;
    } else if (key == "xmlfile") {
        settings.xmlfile = value;
    } else if (key == "outfile") {
        settings.outfile = value;
    } else if (key == "settings") {
        settings.settings_file = value;
    } else if (key == "lexicon") {
        settings.lexicon_file = value;
    } else if (key == "extra-vocab") {
        settings.extra_vocab_files.push_back(value);
    } else if (key == "training") {
        settings.training_folder = value;
    } else if (key == "verbose") {
        settings.verbose = true;
    } else if (key == "debug") {
        settings.debug = true;
    } else if (key == "test") {
        settings.test = true;
    }
}

} // namespace

TaggerSettings parse_arguments(int argc, char** argv) {
    TaggerSettings settings;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--", 0) != 0) {
            continue;
        }
        std::size_t eq = arg.find('=');
        if (eq == std::string::npos) {
            std::string key = arg.substr(2);
            push_option(settings, key, "1");
        } else {
            std::string key = arg.substr(2, eq - 2);
            std::string value = arg.substr(eq + 1);
            push_option(settings, key, value);
        }
    }
    return settings;
}

TaggerSettings load_settings(const TaggerSettings& base) {
    TaggerSettings combined = base;
    std::string settings_path = base.settings_file.empty() ? "./Resources/settings.xml" : base.settings_file;

    pugi::xml_document doc;
    if (!doc.load_file(settings_path.c_str())) {
        throw std::runtime_error("Failed to load settings file: " + settings_path);
    }

    pugi::xpath_node_set parameter_nodes = doc.select_nodes("//neotag/parameters/item");
    pugi::xml_node selected;

    for (const auto& node : parameter_nodes) {
        pugi::xml_node param = node.node();
        if (!base.pid.empty()) {
            if (std::string(param.attribute("pid").value()) == base.pid) {
                selected = param;
                break;
            }
        } else if (!param.attribute("restriction") ||
                   doc.select_node(param.attribute("restriction").value()) != nullptr) {
            selected = param;
            break;
        }
    }

    if (!selected) {
        throw std::runtime_error("No matching parameter set found in settings.xml");
    }

    for (const auto& attr : selected.attributes()) {
        push_option(combined, attr.name(), attr.value());
    }
    pugi::xml_node parent = selected.parent().parent();
    if (parent) {
        for (const auto& attr : parent.attributes()) {
            if (!combined.get(attr.name()).empty()) {
                continue;
            }
            push_option(combined, attr.name(), attr.value());
        }
    }

    return combined;
}

} // namespace flexitag

