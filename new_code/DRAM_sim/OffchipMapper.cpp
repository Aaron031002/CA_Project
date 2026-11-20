#include "DRAM_sim/OffchipMapper.h"

#include <stdexcept>

#include <yaml-cpp/yaml.h>

namespace LLM_SIM_DRAM {

void OffchipMapper::load(const std::string& yaml_path) {
    YAML::Node root = YAML::LoadFile(yaml_path); // throws YAML::BadFile if missing

    if (!root["entries"]) {
        throw std::runtime_error("OffchipMapper::load: missing 'entries' key in " + yaml_path);
    }

    std::unordered_map<std::string, int> new_table;

    for (const auto& node : root["entries"]) {
        if (!node["path"]) {
            throw std::runtime_error("OffchipMapper::load: entry missing 'path' in " + yaml_path);
        }
        if (!node["owner"]) {
            throw std::runtime_error("OffchipMapper::load: entry missing 'owner' in " + yaml_path);
        }
        const std::string path = node["path"].as<std::string>();
        const int owner = node["owner"].as<int>();
        new_table[path] = owner;
    }

    std::lock_guard<std::mutex> lk(mutex_);
    table_ = std::move(new_table);
}

int OffchipMapper::owner_of(const std::string& path) const {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = table_.find(path);
    return (it == table_.end()) ? OWNER_NONE : it->second;
}

bool OffchipMapper::is_mapped(const std::string& path) const {
    std::lock_guard<std::mutex> lk(mutex_);
    return table_.find(path) != table_.end();
}

} // namespace LLM_SIM_DRAM
