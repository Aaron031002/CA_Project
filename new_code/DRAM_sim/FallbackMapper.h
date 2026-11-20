#ifndef LLM_SIM_FALLBACK_MAPPER_H
#define LLM_SIM_FALLBACK_MAPPER_H

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

namespace LLM_SIM_DRAM {

// ---------------------------------------------------------------------------
// FallbackMapper
//
// Compile-time-authored lowest-level-memory fallback target, queried when a
// local HBM miss (and, if not skipped -- see DataManagerConfig::
// skip_group_remote -- the group-remote scan) has already failed to resolve
// a path locally. Answers "for THIS (path, requesting core), where is the
// single lowest-level-memory copy of the data" -- either the existing
// off-chip DRAM path (see OffchipMapper, left completely untouched by this
// class) or a specific (core, sub-region) on another core's local HBM.
//
// Deliberately separate from BOTH AddressMapper (Phase 1, whole-file/range-
// reserved ownership) and OffchipMapper (Phase 6c, single off-chip owner) --
// never merged with either:
//   - AddressMapper answers "who owns this path" (a compile-time-authored,
//     potentially multi-owner fact about the data itself). FallbackMapper
//     answers a DIFFERENT question: "when THIS core misses on this path,
//     where should it go" -- a per-(path, requester) ROUTING decision, not
//     an ownership fact. The same path can have one AddressMapper owner set
//     but many different FallbackMapper entries, one per possible requester.
//   - OffchipMapper's single always-off-chip answer is still available and
//     unchanged for every path/core FallbackMapper has no opinion about
//     (is_mapped() returns false) -- FallbackMapper only needs an entry for
//     a (path, core) pair whose fallback destination should be something
//     OTHER than plain off-chip, or where the CORE destination's exact
//     sub-region needs to be pinned down (row/col into that owner's local
///    copy of the file).
//
// Scope (this round): MoE expert weight fetches only (the entries this
// class ever sees are gate/up/down weight paths keyed by the requesting
// core). Schema is deliberately general (not MoE-specific) so a future
// round can register other data types without a schema change.
//
// Same load-once/query-many lifecycle as AddressMapper/OffchipMapper: C++
// loads the YAML once at DataManager construction, then only ever performs
// read-only lookups -- no runtime set_*() mutation (gen-time authored, like
// OffchipMapper, not runtime-registered like AddressMapper::
// set_owner_and_format()).
// ---------------------------------------------------------------------------
class FallbackMapper {
public:
    enum class DestKind { OFFCHIP, CORE };

    struct Entry {
        DestKind dest_kind = DestKind::OFFCHIP;
        // Valid only when dest_kind == CORE: which core's local HBM to
        // fetch from.
        int dest_core_id = -1;
        // Sub-region within dest_core_id's copy of the file to read.
        // has_region == false means "the whole file" (row_start/row_end/
        // col_start/col_end unused) -- most callers naming a CORE
        // destination will want a specific TP-shard-style sub-region (see
        // DRAMAddr_t's row/col semantics, which this deliberately reuses),
        // but the field is optional so a whole-file CORE fallback doesn't
        // need a synthetic full-range entry.
        bool has_region = false;
        int64_t row_start = 0, row_end = 0;
        int64_t col_start = 0, col_end = 0;
    };

    FallbackMapper() = default;

    // Parses a fallback_map.yaml at yaml_path and replaces the current
    // table. Throws std::runtime_error on a missing file or malformed
    // schema.
    void load(const std::string& yaml_path);

    // Looks up the fallback entry for (path, requesting_core_id). Returns
    // false (and leaves *out untouched) if no entry is registered for this
    // exact (path, core) pair -- callers should treat that as "no opinion,
    // fall back to the existing OffchipMapper/off-chip path unchanged",
    // NOT as an error.
    bool lookup(const std::string& path, int requesting_core_id, Entry* out) const;

    // True if path has AT LEAST ONE entry for ANY requesting core.
    bool is_mapped(const std::string& path) const;

private:
    // path -> (requesting_core_id -> Entry). Nested map (not a single
    // composite-string key) keeps the lookup a plain two-level hash lookup
    // without needing a custom key type or string concatenation on every
    // query.
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::unordered_map<int, Entry>> table_;
};

} // namespace LLM_SIM_DRAM

#endif // LLM_SIM_FALLBACK_MAPPER_H
