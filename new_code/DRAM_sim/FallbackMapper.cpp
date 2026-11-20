#include "DRAM_sim/FallbackMapper.h"

#include <stdexcept>

#include <yaml-cpp/yaml.h>

namespace LLM_SIM_DRAM {

void FallbackMapper::load(const std::string& yaml_path) {
    YAML::Node root = YAML::LoadFile(yaml_path);  // throws YAML::BadFile if missing

    if (!root["entries"]) {
        throw std::runtime_error("FallbackMapper::load: missing 'entries' key in " + yaml_path);
    }

    std::unordered_map<std::string, std::unordered_map<int, Entry>> new_table;

    for (const auto& node : root["entries"]) {
        if (!node["path"]) {
            throw std::runtime_error("FallbackMapper::load: entry missing 'path' in " + yaml_path);
        }
        if (!node["core"]) {
            throw std::runtime_error("FallbackMapper::load: entry missing 'core' in " + yaml_path);
        }
        if (!node["dest"]) {
            throw std::runtime_error("FallbackMapper::load: entry missing 'dest' in " + yaml_path);
        }

        const std::string path = node["path"].as<std::string>();
        const int core = node["core"].as<int>();
        const std::string dest_str = node["dest"].as<std::string>();

        Entry entry;
        if (dest_str == "offchip") {
            entry.dest_kind = DestKind::OFFCHIP;
        } else if (dest_str == "core") {
            entry.dest_kind = DestKind::CORE;
            if (!node["dest_core"]) {
                throw std::runtime_error(
                    "FallbackMapper::load: entry with dest: core missing 'dest_core' in " + yaml_path);
            }
            entry.dest_core_id = node["dest_core"].as<int>();
        } else {
            throw std::runtime_error(
                "FallbackMapper::load: unrecognized 'dest' value \"" + dest_str +
                "\" (expected \"offchip\" or \"core\") in " + yaml_path);
        }

        if (node["region"]) {
            const auto& region = node["region"];
            if (!region["row"] || !region["col"]) {
                throw std::runtime_error(
                    "FallbackMapper::load: 'region' entry missing 'row' or 'col' in " + yaml_path);
            }
            entry.has_region = true;
            entry.row_start = region["row"][0].as<int64_t>();
            entry.row_end   = region["row"][1].as<int64_t>();
            entry.col_start = region["col"][0].as<int64_t>();
            entry.col_end   = region["col"][1].as<int64_t>();
        }

        new_table[path][core] = entry;
    }

    std::lock_guard<std::mutex> lk(mutex_);
    table_ = std::move(new_table);
}

bool FallbackMapper::lookup(const std::string& path, int requesting_core_id, Entry* out) const {
    std::lock_guard<std::mutex> lk(mutex_);
    auto path_it = table_.find(path);
    if (path_it == table_.end()) return false;
    auto core_it = path_it->second.find(requesting_core_id);
    if (core_it == path_it->second.end()) return false;
    if (out != nullptr) *out = core_it->second;
    return true;
}

bool FallbackMapper::is_mapped(const std::string& path) const {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = table_.find(path);
    return it != table_.end() && !it->second.empty();
}

} // namespace LLM_SIM_DRAM
