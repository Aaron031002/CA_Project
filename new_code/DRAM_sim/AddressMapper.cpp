#include "DRAM_sim/AddressMapper.h"

#include <fstream>
#include <stdexcept>

#include <yaml-cpp/yaml.h>

namespace LLM_SIM_DRAM {

void AddressMapper::load(const std::string& yaml_path) {
    YAML::Node root = YAML::LoadFile(yaml_path); // throws YAML::BadFile if missing

    if (!root["entries"]) {
        throw std::runtime_error("AddressMapper::load: missing 'entries' key in " + yaml_path);
    }

    std::unordered_map<std::string, OwnerEntry> new_table;

    for (const auto& node : root["entries"]) {
        if (!node["path"]) {
            throw std::runtime_error("AddressMapper::load: entry missing 'path' in " + yaml_path);
        }
        const std::string path = node["path"].as<std::string>();

        OwnerEntry entry;

        if (node["owner"]) {
            entry.owners.push_back(node["owner"].as<int>());
        }
        if (node["owners"]) {
            for (const auto& o : node["owners"]) {
                entry.owners.push_back(o.as<int>());
            }
        }

        if (node["kind"]) {
            entry.kind = node["kind"].as<std::string>();
        }

        if (node["format"]) {
            entry.format = node["format"].as<std::string>();
        }

        if (node["always_resident_owners"]) {
            for (const auto& o : node["always_resident_owners"]) {
                entry.always_resident_owners.push_back(o.as<int>());
            }
        }

        if (node["warm_seed_owners"]) {
            for (const auto& o : node["warm_seed_owners"]) {
                entry.warm_seed_owners.push_back(o.as<int>());
            }
        }

        if (node["no_evict_owners"]) {
            for (const auto& o : node["no_evict_owners"]) {
                entry.no_evict_owners.push_back(o.as<int>());
            }
        }

        // Schema-reserved range-level ownership (see AddressMapper.h). Parsed
        // and stored so round-trips are lossless, but owner_of() does not
        // consult it yet (Phase 1 only does whole-file lookups).
        if (node["ranges"]) {
            for (const auto& r : node["ranges"]) {
                RangeEntry re{};
                re.row_start = r["row"][0].as<int64_t>();
                re.row_end   = r["row"][1].as<int64_t>();
                re.col_start = r["col"][0].as<int64_t>();
                re.col_end   = r["col"][1].as<int64_t>();
                re.owner     = r["owner"].as<int>();
                entry.ranges.push_back(re);
            }
        }

        new_table.emplace(path, std::move(entry));
    }

    std::lock_guard<std::mutex> lk(mutex_);
    table_ = std::move(new_table);
}

int AddressMapper::owner_of(const std::string& path, int64_t /*row*/, int64_t /*col*/) const {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = table_.find(path);
    if (it == table_.end() || it->second.owners.empty()) {
        return OWNER_LOCAL;
    }
    return it->second.owners.front();
}

std::vector<int> AddressMapper::owners_of(const std::string& path) const {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = table_.find(path);
    if (it == table_.end()) return {};
    return it->second.owners;
}

bool AddressMapper::is_mapped(const std::string& path) const {
    std::lock_guard<std::mutex> lk(mutex_);
    return table_.find(path) != table_.end();
}

bool AddressMapper::is_initial_kv(const std::string& path) const {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = table_.find(path);
    return it != table_.end() && it->second.kind == "initial_kv";
}

bool AddressMapper::is_output(const std::string& path) const {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = table_.find(path);
    return it != table_.end() && it->second.kind == "output";
}

int AddressMapper::elem_bits_of(const std::string& path) const {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = table_.find(path);
    if (it != table_.end() && it->second.format == "mxfp4") return 4;
    // (2026-08-14, Bug H
    // continuation): real GPT-OSS attention Q/K/V/Out projection weights
    // and their activations (incl. K/V cache) are F16, per
    // §3/§8.2 ("F16 儲存與相乘"). elem_bits is
    // PURELY a latency/capacity accounting field (Cache::data_bits/
    // latency_cal, DMA::cal_lat, NoC packet sizing) -- it never touches
    // Data_t<T>::value or any arithmetic path, so this only corrects the
    // simulator's bandwidth/capacity model; computed values are completely
    // unaffected. K/V cache files ARE registered via dram_map_writer.add()
    // (see _prefill_real_impl.py/_decode_real_impl.py's kcache_b_core/
    // vcache_b_core/kcache_ws_b_core call sites, now tagged format="f16"
    // there), so a plain table_ lookup suffices -- no path-substring match
    // needed.
    if (it != table_.end() && it->second.format == "f16") return 16;
    return 32;
}

void AddressMapper::set_owner(const std::string& path, int core_id) {
    std::lock_guard<std::mutex> lk(mutex_);
    auto& entry = table_[path];
    entry.owners.assign(1, core_id);
}

void AddressMapper::set_owner_and_format(const std::string& path, int core_id, int elem_bits) {
    std::lock_guard<std::mutex> lk(mutex_);
    auto& entry = table_[path];
    entry.owners.assign(1, core_id);
    // Same string convention as load()'s "format:" YAML field. Only mxfp4
    // and f16 are handled -- these are the only two non-default widths
    // elem_bits_of() (below) actually recognizes on the READ side; a
    // hypothetical "q8_0" tag here would silently do nothing (elem_bits_of
    // has no matching branch, falls through to 32) since Q8_0-quantized
    // data has no DRAM-write path in this simulator today (quantization
    // is a local,
    // transient, never-DRAM-resident step). Any other elem_bits value
    // (including 8) clears the tag rather than writing an unrecognized
    // one, so elem_bits_of() falls through to its own documented 32-bit
    // default -- explicit and consistent, not a silent mismatch.
    if (elem_bits == 4)       entry.format = "mxfp4";
    else if (elem_bits == 16) entry.format = "f16";
    else                      entry.format.clear();
}

// ---------------------------------------------------------------------------
// Phase 3a: DRAM group membership (8x8 mesh, 4-neighbor, no torus wrap)
// ---------------------------------------------------------------------------
void AddressMapper::load_groups(const std::string& bin_path, int num_cores) {
    std::ifstream f(bin_path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        throw std::runtime_error("AddressMapper::load_groups: cannot open " + bin_path);
    }
    const std::streamsize file_size = f.tellg();
    const std::streamsize expected_bytes =
        (static_cast<std::streamsize>(num_cores) * num_cores + 7) / 8;
    if (file_size != expected_bytes) {
        throw std::runtime_error(
            "AddressMapper::load_groups: " + bin_path + " has size " +
            std::to_string(file_size) + " but expected " +
            std::to_string(expected_bytes) + " for num_cores=" + std::to_string(num_cores));
    }
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> bits(static_cast<size_t>(expected_bytes));
    f.read(reinterpret_cast<char*>(bits.data()), expected_bytes);
    if (!f) {
        throw std::runtime_error("AddressMapper::load_groups: read failed for " + bin_path);
    }

    std::lock_guard<std::mutex> lk(group_mutex_);
    group_bits_ = std::move(bits);
    num_cores_g_ = num_cores;
}

int AddressMapper::num_cores() const {
    std::lock_guard<std::mutex> lk(group_mutex_);
    return num_cores_g_;
}

std::vector<std::pair<std::string, std::vector<int>>> AddressMapper::all_entries() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::vector<std::pair<std::string, std::vector<int>>> out;
    out.reserve(table_.size());
    for (const auto& [path, entry] : table_) {
        out.emplace_back(path, entry.owners);
    }
    return out;
}

std::vector<int> AddressMapper::always_resident_owners_of(const std::string& path) const {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = table_.find(path);
    return (it == table_.end()) ? std::vector<int>{} : it->second.always_resident_owners;
}

std::vector<int> AddressMapper::warm_seed_owners_of(const std::string& path) const {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = table_.find(path);
    return (it == table_.end()) ? std::vector<int>{} : it->second.warm_seed_owners;
}

std::vector<int> AddressMapper::no_evict_owners_of(const std::string& path) const {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = table_.find(path);
    return (it == table_.end()) ? std::vector<int>{} : it->second.no_evict_owners;
}

bool AddressMapper::is_group(int core_a, int core_b) const {
    if (core_a == core_b) return false;

    std::lock_guard<std::mutex> lk(group_mutex_);
    if (group_bits_.empty()) return false;
    if (core_a < 0 || core_a >= num_cores_g_ || core_b < 0 || core_b >= num_cores_g_) {
        return false;
    }

    const size_t bit_idx  = static_cast<size_t>(core_a) * num_cores_g_ + core_b;
    const size_t byte_idx = bit_idx / 8;
    const int    bit_off  = static_cast<int>(bit_idx % 8);
    return (group_bits_[byte_idx] >> bit_off) & 0x1;
}

} // namespace LLM_SIM_DRAM
