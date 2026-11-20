#ifndef LLM_SIM_OFFCHIP_MAPPER_H
#define LLM_SIM_OFFCHIP_MAPPER_H

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

namespace LLM_SIM_DRAM {

// ---------------------------------------------------------------------------
// OffchipMapper (Phase 6c)
//
// Deliberately separate from AddressMapper (Phase 1) -- loads its own,
// separate YAML file (never merged with dram_map.yaml, per explicit design
// decision), and answers "which off-chip DRAM node (id 64..71, see
// LLM_SIM_NOC::OFFCHIP_ID_BASE) owns this file" instead of "which core".
//
// Simpler than AddressMapper by design, matching off-chip DRAM's simpler
// ownership model:
//   - Single owner per file (off-chip DRAM is the one authoritative backing
//     store for a piece of data -- unlike local DRAM, there's no TP-sharding
//     scenario where the same off-chip file needs multiple owners).
//   - No range-level entries (schema-reserved in AddressMapper for a future
//     that hasn't arrived; not needed here either).
//   - No is_group()/adjacency concept: off-chip DRAM nodes don't access each
//     other (explicitly out of scope for this round).
//   - No set_owner()/runtime mutation: which node owns which file is a
//     scheduling-time (inst-gen) decision, baked in once at load() time and
//     never reassigned at runtime (explicitly out of scope for this round).
// ---------------------------------------------------------------------------
class OffchipMapper {
public:
    // Sentinel returned by owner_of() when a path has no off-chip entry.
    static constexpr int OWNER_NONE = -1;

    OffchipMapper() = default;

    // Parses offchip_map.yaml at yaml_path and replaces the current table.
    // Throws std::runtime_error on a missing file or malformed schema.
    void load(const std::string& yaml_path);

    // Returns the off-chip node id (LLM_SIM_NOC::OFFCHIP_ID_BASE..+NUM_OFFCHIP_NODES-1)
    // that owns path, or OWNER_NONE if path has no off-chip entry.
    int owner_of(const std::string& path) const;

    // True if path has an entry in the loaded map.
    bool is_mapped(const std::string& path) const;

private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, int> table_; // path -> offchip node id
};

} // namespace LLM_SIM_DRAM

#endif // LLM_SIM_OFFCHIP_MAPPER_H
