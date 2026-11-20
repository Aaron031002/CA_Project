#ifndef LLM_SIM_DATA_MANGER_H
#define LLM_SIM_DATA_MANGER_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <utility>
#include <list>
#include <deque>
#include <memory>
#include <mutex>
#include <atomic>
#include <type_traits>
#include <systemc.h>

#include "utils/Req.h"
#include "DRAM_sim/dram_params.h"
#include "DRAM_sim/AddressMapper.h"
#include "DRAM_sim/OffchipMapper.h"
#include "DRAM_sim/FallbackMapper.h"

// ---------------------------------------------------------------------------
// DataManagerConfig
//
// Construction parameters for DataManager<T>, grouped into a struct so that
// future additions (e.g. a miss-handler seam) only require adding a field
// here rather than changing every call site's constructor signature.
//
// dram_map_path: path to a dram_map.yaml (see AddressMapper) describing
// initial DRAM ownership. Empty string (default) means "no mapping to load"
// -- address_mapper() is still usable, just empty (owner_of() always returns
// AddressMapper::OWNER_LOCAL). This keeps DataManager constructible in
// contexts (unit tests, standalone Core testbenches) that don't care about
// ownership tracking.
//
// dram_group_path: path to a dram_group.bin (Phase 3a) describing which
// cores may access each other's DRAM. Empty string (default) means "no
// group bitmap to load" -- is_group() always returns false, so no cross-
// core reads will ever be attempted (DMA's owner-lookup only redirects when
// an owner is found AND is_group() is true).
// ---------------------------------------------------------------------------
struct DataManagerConfig {
    bool compute = true;
    std::string dram_map_path = "";
    std::string dram_group_path = "";

    // Phase 7a: per-core local DRAM capacity limit, in bytes
    // (not element count -- K/V/weight element sizes are not guaranteed
    // uniform across Q8_0-related paths). 0 (default) means "no capacity
    // limit / eviction disabled", preserving every existing call site's
    // behavior unchanged.
    uint64_t local_dram_capacity_bytes = 0;
    // Optional per-core override (core_id -> bytes). A core not present
    // here falls back to local_dram_capacity_bytes. Empty by default.
    std::unordered_map<int, uint64_t> local_dram_capacity_overrides;

    // Phase 7b: path to offchip_map.yaml (Phase 6c) -- deliberately a
    // SEPARATE file from dram_map_path, never merged (see OffchipMapper's
    // own header comment). Empty (default) means no off-chip ownership
    // mapping is loaded (offchip_mapper().owner_of() always returns
    // OffchipMapper::OWNER_NONE).
    std::string offchip_map_path = "";

    // path to a fallback_map.yaml (see
    // FallbackMapper). Deliberately a SEPARATE file from both dram_map_path
    // and offchip_map_path, never merged (same rationale as OffchipMapper's
    // own header comment -- a different question than either "who owns
    // this path" or "which off-chip node backs this path"). Empty (default)
    // means no fallback mapping is loaded (fallback_mapper().lookup()
    // always returns false, i.e. "no opinion, use the existing off-chip
    // path unchanged").
    std::string fallback_map_path = "";

    // Phase G: optional path to a residency
    // YAML produced by a PREVIOUS chained run's DataManager::dump_residency()
    // (see that method's own header comment for the file schema). Loaded
    // INSIDE the DataManagerConfig constructor, BEFORE warm_start() runs --
    // this ordering is load-bearing, not incidental: touch_lru() always
    // inserts a genuinely new path at the front (most-recently-used) of the
    // LRU list, so loading the prior layer's residency first and then
    // running this layer's own warm_start() naturally leaves the prior
    // layer's files at the BACK of the list (least-recently-used), meaning
    // they are exactly what gets evicted first once this layer's own files
    // push total usage over the capacity budget -- the "older layers get
    // evicted before the current layer's own weights" behavior Phase G's
    // residency chaining is meant to produce. Loading it the other way
    // around (after warm_start) would invert this and evict the CURRENT
    // layer's own just-loaded weights first instead, defeating the point.
    // Empty (default, every pre-Phase-G caller) means no prior residency --
    // this run starts from warm_start() alone.
    std::string dram_residency_in_path = "";

    // compile-time switch to skip the
    // group-remote scan (DMA::try_dispatch()'s call to
    // find_resident_group_member()) entirely, falling straight through to
    // the existing off-chip/lowest-level-memory fallback as if every group
    // member were non-resident. The underlying mechanism (find_resident_
    // group_member(), the 4-neighbor group bitmap, DRAMRouter's group-gated
    // cross-core read path) is left fully intact -- this only gates the one
    // call site that invokes it, so the mechanism can be re-enabled later
    // without any code changes. False (default) preserves every existing
    // call site's behavior unchanged.
    bool skip_group_remote = false;
};

// ---------------------------------------------------------------------------
// DataManager<T>
//
// Converts a DRAMAddr_t into actual byte positions within a binary file:
//   element (r, c)  →  byte offset (r * row_stride + c) * sizeof(T)
//
// Synthetic KV cache file layout (per K_n.bin / V_n.bin, created by
// init_kv_files() below -- NOT what the real gpt-oss decode path uses):
//   rows    = BATCH * HEAD_NUM
//   stride  = MAX_SEQ_LEN * HEAD_DIM
//   row index = batch_id * HEAD_NUM + head_id
//   seq pos s occupies columns [s*HEAD_DIM, (s+1)*HEAD_DIM)
// The REAL decode path (inst_gen/gen_code/_decode_real_impl.py's
// kcache_ws_path()/vcache_path()) uses a completely different, simpler
// scheme instead: one dedicated file per (batch, kv_head) -- e.g.
// "Kcache_ws_real_b{batch}/kv_{kv_head}/K_0.bin", shape [seq_len, 1,
// HEAD_DIM] (row = sequence position, col width == HEAD_DIM exactly). Since
// each such read/write's col range always spans the file's full row width
// (no TP-style column sharding, no `stride` keyword ever generated for
// these paths), row_stride defaults to col_end (Req.h) and consecutive rows
// are physically contiguous in the file -- unlike TP2/TP4's column-sharded
// expert-weight reads (gate_weight.bin/up_weight.bin with an explicit
// `stride` narrower than the row), which DO leave a gap between rows. Don't
// use the BATCH*HEAD_NUM/MAX_SEQ_LEN*HEAD_DIM layout above to reason about
// the real decode path's KV files -- it only describes init_kv_files()'s
// own synthetic scheme.
//
// Thread safety: different files are accessed in parallel; same-file accesses
// are serialised through a per-file mutex.
//
// Phase 2: DataManager is meant to be constructed ONCE per
// simulation (by Acc8x8/Acc) and injected as a non-owning pointer into every
// Core, replacing the previous one-instance-per-Core design. This makes the
// per-file mutex table and the AddressMapper it now owns genuinely global,
// fixing the old bug where per-instance mutexes could not serialise
// cross-core access to the same file (each Core's mutex only protected that
// Core's own instance).
// ---------------------------------------------------------------------------
template <typename T>
class DataManager {
    public:
        // compute: if false, all file I/O is skipped:
        //   read_data  → returns a zero-filled vector (no file opened)
        //   write_data → no-op
        //   init_kv_files → no-op (KV cache files are not created)
        explicit DataManager(bool compute = true);

        // Preferred constructor: loads cfg.dram_map_path into the
        // AddressMapper (if non-empty) in addition to the compute_ behavior
        // above.
        explicit DataManager(const DataManagerConfig& cfg);

        // core_id: Phase 7a LRU tracking. -1 (default) means "not attributed
        // to any core's local DRAM capacity" -- preserves every pre-Phase-7
        // call site's behavior unchanged (no LRU touch, no eviction, no
        // resident-tracking). Pass the calling DRAMSim's core_id to opt in.
        std::vector<T> read_data (const LLM_SIM_DRAM::DRAMAddrPtr_t& addr_ptr,
                                   int core_id = -1);
        // step 6: elem_bits is the caller's
        // already-resolved value (format tag if tagged, else the
        // producer Data_t's own elem_bits as fallback, else 32) -- see
        // DRAMSim::do_write's doc comment for how it's computed. Defaults
        // to 32 so every pre-existing call site (none currently pass a
        // non-default value) is unaffected.
        void           write_data(const LLM_SIM_DRAM::DRAMAddrPtr_t& addr_ptr,
                                  const std::vector<T>& data, int core_id = -1,
                                  int elem_bits = 32);

        // Owner-mapping / group-topology query layer (Phase 1's AddressMapper).
        // Loaded from DataManagerConfig::dram_map_path at construction time;
        // empty/default-constructed (no entries) if no path was given.
        LLM_SIM_DRAM::AddressMapper&       address_mapper()       { return address_mapper_; }
        const LLM_SIM_DRAM::AddressMapper& address_mapper() const { return address_mapper_; }

        // Off-chip ownership mapping (Phase 6c/7b). Loaded from
        // DataManagerConfig::offchip_map_path at construction time;
        // empty/default-constructed (no entries) if no path was given.
        LLM_SIM_DRAM::OffchipMapper&       offchip_mapper()       { return offchip_mapper_; }
        const LLM_SIM_DRAM::OffchipMapper& offchip_mapper() const { return offchip_mapper_; }

        // compile-time lowest-level-
        // memory fallback mapping. Loaded from DataManagerConfig::
        // fallback_map_path at construction time; empty/default-constructed
        // (no entries) if no path was given, in which case every lookup()
        // call returns false and callers fall back to the existing
        // OffchipMapper-only behavior unchanged.
        LLM_SIM_DRAM::FallbackMapper&       fallback_mapper()       { return fallback_mapper_; }
        const LLM_SIM_DRAM::FallbackMapper& fallback_mapper() const { return fallback_mapper_; }

        // ── Phase 7a/7b: LRU/capacity query layer ─────────────────────────
        // True if core_id has capacity tracking enabled (via
        // DataManagerConfig::local_dram_capacity_bytes/_overrides) AND path
        // is currently resident in core_id's LRU state. Used by DRAMSim's
        // do_read (Phase 7b) to distinguish "group owner has this file but
        // it's been evicted" (real Phase 7 miss -> off-chip fetch) from
        // "capacity tracking isn't enabled for this core" (pre-Phase-7
        // behavior: always resident, never miss).
        bool is_resident(int core_id, const std::string& path) const;

        // True if core_id has a nonzero capacity limit configured (either
        // the global default or a per-core override). If false, is_resident
        // always returns true (capacity tracking not opted into for this
        // core) and read_data/write_data never touch lru_states_ for it.
        bool has_capacity_limit(int core_id) const;

        // true if DMA::try_dispatch()
        // should skip calling find_resident_group_member() entirely for
        // this core (see DataManagerConfig::skip_group_remote's doc
        // comment). Global switch (not per-core, unlike capacity) --
        // returns the same value for every core_id today, but takes a
        // core_id parameter for symmetry with has_capacity_limit() and in
        // case a future caller needs a per-core variant.
        bool skip_group_remote(int core_id) const;

        // Phase D: current LRU-tracked bytes
        // for core_id (0 if capacity tracking isn't enabled for it, or it has
        // never been touched). Used by test gates to confirm warm_start()
        // seeded the expected footprint, not a runtime-hot-path query.
        uint64_t used_bytes(int core_id) const;

        // ── Phase 7c/Phase A: eviction-in-flight race guard ───────────────
        // True while (core_id, path) is being written back to its off-chip
        // owner as part of an eviction (set by the eviction routine inside
        // read_data/write_data's LRU-touch path, cleared once the
        // write-back packet has been sent). DRAMSim's do_read/do_write
        // (Phase 3b group-read path) checks this before servicing a
        // request for path -- not a fatal condition, just a signal to
        // requeue and retry later.
        //
        // Phase A: keyed by (core_id, path),
        // not path alone -- a path resident on many cores (e.g. a
        // broadcast weight owned by all 64) must only block the ONE core
        // that is actually evicting its own copy, not every other core
        // that happens to share the same path.
        bool is_evicting(int core_id, const std::string& path) const;

        // Phase 7c: called by DMA's write-back dispatch once an eviction
        // write-back packet has actually been sent, clearing the guard set
        // above for (core_id, path).
        void clear_evicting(int core_id, const std::string& path);

        // Phase 7c: pops the next KV entry queued for write-back on
        // core_id, after an eviction on that core decided to drop it
        // (weight entries never appear here -- see LruEntry::is_kv).
        // Returns false if core_id has nothing pending. Populated
        // internally by the eviction routine inside read_data/write_data;
        // consumed by that core's own DMA's off-chip write-back dispatch.
        //
        // Phase A: partitioned per core_id (not one shared queue) so each
        // core's DMA only ever dequeues write-backs that are actually its
        // own to perform -- avoids a cross-core race over a single queue.
        struct EvictedKv {
            int core_id;
            std::string path;
            std::vector<T> data;
        };
        bool pop_pending_writeback(int core_id, EvictedKv& out);

        // ── Phase C: group-wide residency probe ──
        // Returns the first OTHER core (not core_id itself) in core_id's
        // dram_group.bin neighborhood that currently has `path` resident, or
        // AddressMapper::OWNER_LOCAL (-1) if none do (including core_id
        // itself -- callers wanting to check core_id's own residency first
        // should call is_resident(core_id, path) separately, per the
        // routing rule in DMA::try_dispatch).
        //
        // Single lru_mutex_ acquisition for the whole scan (unlike calling
        // the public is_resident() once per group member, which would each
        // separately lock/unlock via has_capacity_limit()+capacity_for()) --
        // group probes happen on DMA's per-dispatch hot path, so this
        // matters. Group membership itself is looked up in group_members_,
        // precomputed once at construction from dram_group.bin (which is
        // static for the whole run), avoiding AddressMapper::is_group()'s
        // own group_mutex_ on this same hot path.
        int find_resident_group_member(int core_id, const std::string& path) const;

        // ── Phase E: logical width/size ──────
        // Coarse per-element bit width for path (32 fp32 default/unmapped, 4
        // mxfp4) -- delegates to AddressMapper::elem_bits_of(). Suitable for
        // transfer-cycle formulas (LLM_SIM_PRECISION::transfer_cycles), NoC
        // flit_size, and Cache capacity gating. NOT suitable for exact byte
        // accounting -- use logical_bytes() for that (see its own comment for
        // why: real MXFP4 averages 4.25 bits/element, not 4.00).
        int elem_bits_for(const std::string& path) const;

        // Exact logical byte count for n_elems of path, used ONLY for LRU/
        // capacity accounting (touch_lru/evict_if_needed) where a fixed byte
        // budget (e.g. 512 MiB/core) makes the 6% gap between the coarse
        // elem_bits_for()==4 approximation and MXFP4's real 4.25 bits/element
        // meaningful. MXFP4: ceil(n/32)*17 bytes (32-element block: 32 nibbles
        // + 1 shared byte scale = 17 bytes). fp32 (including unmapped paths):
        // n*4 bytes. Deliberately NOT `n * elem_bits_for(path) / 8` -- see
        // AddressMapper::elem_bits_of()'s own doc comment and this plan's
        // Decision 4.
        uint64_t logical_bytes(const std::string& path, uint64_t n_elems) const;

        // step 6: overload taking the
        // producer's own Data_t::elem_bits as a FALLBACK, used only when
        // path carries no format= tag (address_mapper_.elem_bits_of(path)
        // == 32, the untagged default). When path IS tagged (mxfp4/f16),
        // the tag remains the sole authority -- this overload delegates
        // straight to the path-only version above, unchanged, so every
        // already-tagged weight/KV-cache path is completely unaffected.
        // Only an untagged path with a non-default producer_elem_bits
        // (!= 32) uses that value for the byte-count formula (n*bytes-
        // per-elem, no mxfp4-style block encoding since only mxfp4 paths
        // use that and they're always tag-covered by the branch above).
        uint64_t logical_bytes(const std::string& path, uint64_t n_elems,
                                int producer_elem_bits) const;

        // ── Phase D: observability counters ──
        // Both are global (not per-core) running totals for the whole
        // simulation, used by test gates to confirm a run triggered exactly
        // the amount of remote-read/eviction traffic expected (e.g. "warm
        // start means zero of either while capacity comfortably fits every
        // owned file" -- see acc_8x8's local_dram_capacity_bytes doc
        // comment). No reset method: constructed fresh per DataManager
        // instance, which is one per simulation run (Phase 2 singleton).
        uint64_t eviction_count() const;
        uint64_t remote_read_count() const;

        // genuine off-chip fetch
        // counter -- eviction_count() only counts entries being KICKED OUT of
        // local capacity, and remote_read_count() only counts remote-read
        // DISPATCH attempts (before hit/miss is known); neither directly
        // proves data was actually pulled back across the NoC from
        // off-chip DRAM. Incremented inside install_resident() (see its own
        // comment for the two call sites this covers: a core's own local
        // capacity miss, and a remote group-read that was already a known
        // miss) -- every increment corresponds to one real completed
        // DMA::start_offchip_fetch() wire transfer. Global running total,
        // same aggregation convention as eviction_count_/remote_read_count_
        // above.
        uint64_t offchip_fill_count() const;

        // eviction_count() breakdown
        // by victim kind (KV cache entry vs. everything else -- expert/attn
        // weight files) -- added to answer "what's actually getting evicted"
        // instead of guessing from directory sizes. Same global running-total
        // convention as eviction_count_ itself; eviction_count() ==
        // eviction_count_kv() + eviction_count_weight() always.
        uint64_t eviction_count_kv() const;
        uint64_t eviction_count_weight() const;

        // Called by DMA::try_dispatch once a cross-core read has actually
        // been dispatched (not merely attempted -- e.g. blocked-by-busy-slot
        // attempts don't count). Kept here (rather than a DMA-local static)
        // so it aggregates across all 64 per-core DMA instances into one
        // number, matching how eviction_count_ already aggregates across
        // cores.
        void note_remote_read();

        // ── Phase F: fill-and-serve ──────────
        // Marks path as resident on core_id with the LOGICAL size of n_elems
        // (via logical_bytes()) -- a thin public wrapper around the private
        // touch_lru(), used by DRAMSim::fill_thread once an off-chip fill
        // completes. Callers pass the WHOLE FILE's element count (typically
        // std::filesystem::file_size(path)/sizeof(T)), matching warm_start()'s
        // own convention -- the fill's actual wire transfer only moves the
        // requested slice (to avoid paying a full-file off-chip latency for
        // every partial read), but LRU/capacity accounting always tracks
        // whole-file residency, same approximation warm_start() already makes.
        void install_resident(int core_id, const std::string& path, uint64_t n_elems);

        // Fill in-flight guard, keyed by (core_id, path) -- same shape as
        // is_evicting()/clear_evicting() above. Prevents redundant off-chip
        // fetches when multiple pending local reads on the SAME core miss the
        // SAME path before the first fetch lands (e.g. a TP4 shard core's
        // gate/up/down weight reads all missing the same evicted file at
        // once) -- only the FIRST triggers a real fetch; the rest just queue
        // behind it in DRAMSim's existing deferred_ and get retried once
        // fill_thread clears the guard.
        bool is_fill_inflight(int core_id, const std::string& path) const;
        void mark_fill_inflight(int core_id, const std::string& path);
        void clear_fill_inflight(int core_id, const std::string& path);

        // ── Phase G: cross-process residency chaining ──
        // Serializes every capacity-tracked core's current LRU state (front
        // = most-recently-used, back = least-recently-used) to out_path, in
        // this schema:
        //   version: 1
        //   cores:
        //     - core_id: 0
        //       entries:                 # front-to-back == MRU-to-LRU
        //         - path: "..."
        //           size_bytes: 12345
        // Only cores present in lru_states_ (i.e. touched at least once
        // under a nonzero capacity budget) are written -- a core with no
        // capacity limit configured has no LRU state to dump. Meant to be
        // called after a run finishes (e.g. after sc_start() returns), then
        // fed to the NEXT chained run's DataManagerConfig::
        // dram_residency_in_path. Read-only w.r.t. simulation state --
        // does not affect eviction_count()/remote_read_count() or anything
        // else, purely an observability/checkpointing dump.
        void dump_residency(const std::string& out_path) const;

    private:
        // Phase G: loads a residency YAML written by dump_residency() (see
        // its schema comment above) into lru_states_, preserving the
        // dumped front-to-back (MRU-to-LRU) order exactly -- called ONLY
        // from the DataManagerConfig constructor, BEFORE warm_start()
        // (see DataManagerConfig::dram_residency_in_path's header comment
        // for why that ordering matters). Not exposed publicly: calling it
        // after construction would silently break that ordering guarantee.
        // Entries loaded this way are always treated as plain (is_kv=false)
        // for eviction purposes regardless of what they originally were --
        // a stale KV-cache entry from an earlier layer in this chained
        // decode workflow has no further consumer once its layer's run has
        // finished (unlike a same-layer KV eviction, which needs write-back
        // for correctness), so silently dropping it on eviction is correct,
        // not an approximation.
        void load_residency(const std::string& in_path);

    private:
        const bool compute_;
        LLM_SIM_DRAM::AddressMapper address_mapper_;
        LLM_SIM_DRAM::OffchipMapper offchip_mapper_;
        LLM_SIM_DRAM::FallbackMapper fallback_mapper_;

        std::mutex map_mutex_;
        std::unordered_map<std::string, std::unique_ptr<std::mutex>> file_mutexes_;
        std::mutex& get_file_mutex(const std::string& path);

        // Creates K_n.bin and V_n.bin for all layers (only when T == float and compute_=true).
        // Skips files that already exist.
        void init_kv_files();

        // ── Phase 7a: LRU/capacity state, partitioned by core_id ──────────
        // DataManager remains the Phase 2 global singleton; this is an
        // internal per-core-partitioned data structure inside it (same
        // pattern as AddressMapper::is_group()'s internal bitmap), not a
        // reversion to per-core DataManager instances.
        struct LruEntry {
            std::string path;
            uint64_t size_bytes = 0;
            bool is_kv = false;
            // when true, this entry must
            // never be selected as an eviction victim by evict_if_needed()
            // (see AddressMapper::no_evict_owners_of()'s doc comment for the
            // motivating use case). Independent of is_kv/is_evicting --
            // does not change what happens to the entry when it IS evicted,
            // only whether it is eligible to be picked in the first place.
            bool no_evict = false;
        };
        struct LruState {
            uint64_t capacity_bytes = 0;
            uint64_t used_bytes = 0;
            std::list<LruEntry> order;  // front = most-recently-used
            std::unordered_map<std::string, typename std::list<LruEntry>::iterator> index;
        };

        mutable std::mutex lru_mutex_;
        uint64_t local_dram_capacity_default_ = 0;
        std::unordered_map<int, uint64_t> local_dram_capacity_overrides_;
        // mirrors DataManagerConfig::
        // skip_group_remote, set once at construction. Not behind
        // lru_mutex_ -- written only in the constructor, read-only for the
        // rest of the object's lifetime, same convention as the two fields
        // above.
        bool skip_group_remote_ = false;
        std::unordered_map<int, LruState> lru_states_;
        // Phase A: (core_id, path) pair, not path alone -- see is_evicting()'s
        // header comment. std::set (not unordered_set) to avoid needing a
        // custom pair hash for what is a rare, small-cardinality guard set.
        std::set<std::pair<int, std::string>> evicting_paths_;
        // Phase A: partitioned per core_id -- see pop_pending_writeback()'s
        // header comment.
        std::unordered_map<int, std::deque<EvictedKv>> pending_writebacks_;
        // Phase F: (core_id, path) pair, same shape/rationale as
        // evicting_paths_ above -- see is_fill_inflight()'s header comment.
        std::set<std::pair<int, std::string>> fill_inflight_;

        uint64_t capacity_for(int core_id) const;         // 0 = not configured; takes lru_mutex_
        uint64_t capacity_for_locked(int core_id) const;  // same, caller already holds lru_mutex_
        void touch_lru(int core_id, const std::string& path, uint64_t size_bytes);
        void evict_if_needed(int core_id, LruState& state);  // caller holds lru_mutex_

        // ── Phase C: group-wide residency probe support ───────────────────
        // group_members_[i] = every OTHER core j with address_mapper_.
        // is_group(i,j)==true, precomputed once here (constructor, after
        // dram_group_path is loaded) since the bitmap never changes at
        // runtime. Absent key (or empty vector) == no group configured for
        // that core (either no dram_group_path was given, or the bitmap
        // just doesn't list any neighbors for it).
        std::unordered_map<int, std::vector<int>> group_members_;
        void build_group_members();  // called once from both constructors' bodies

        // ── Phase D: warm-start seeding ────────────────────────────────────
        // Pre-populates each owning core's LRU state with the files
        // dram_map.yaml says it owns (address_mapper_.all_entries()), sized
        // by real on-disk file size (fp32 accounting -- Phase E adds
        // MXFP4-aware sizing). Only touches cores with a nonzero capacity
        // configured (has_capacity_limit()); a core with no capacity limit
        // stays fully untracked, exactly as before Phase D. Called once from
        // the DataManagerConfig constructor, after init_kv_files() so any
        // lazily-created KV files already exist on disk.
        // broadcast_only, when
        // true, restricts seeding to entries that are either (a) a genuine
        // broadcast file -- owners list spans (close to) every core, e.g.
        // shared norm/router weights, legitimately needed fresh by every
        // core every layer, NOT part of any per-expert tier-ladder -- or
        // (b) a single-owner entry -- the TP1 base position (core e holds
        // expert e), hardware-out-of-the-box resident, same rationale as a
        // broadcast file, not a tier-ladder position "acquired" by demand.
        // Both stay free to seed even on chained (non-cold-start) layers.
        // Entries with owners.size() in {4,5,6} (ADP's TP2/TP4(+R)
        // tier-ladder shard/replica union positions) are skipped in that
        // mode -- those must be left to load_residency()'s honest
        // prior-layer state / a genuine local-miss fetch, or the
        // local-miss branch that triggers real off-chip refetches can never
        // fire. Default false reproduces the original cold-start (layer 0 /
        // non-chained run) behavior: seed everything.
        void warm_start(bool broadcast_only = false);

        // init-dram-mapping-dram-fetching-nested-blum plan: pre-seeds every
        // owner in warm_seed_owners_of(path) (TP4 replica=1/2 shard-cores
        // under gen_prefill_dp_adaptive(tp4_only=True)) into LRU via plain
        // touch_lru() -- DELIBERATELY NOT the always_resident_owners_of()
        // path, so these entries remain completely normal, evictable LRU
        // entries afterwards (unlike warm_start(broadcast_only=true)'s
        // always-resident seeding, which is permanently exempt from
        // eviction). Called once per layer from the DataManagerConfig
        // constructor, immediately after warm_start(broadcast_only=true)
        // and BEFORE load_residency()/cold-start warm_start() -- so genuine
        // prior-layer residency (or a true cold start's full seed) ends up
        // more-recently-used than this speculative pre-seed, meaning under
        // capacity pressure these entries are evicted before confirmed-
        // in-use data, not instead of it.
        void warm_start_extra();

        // Phase D: observability counters (see public eviction_count()/
        // remote_read_count() doc comments). std::atomic so note_remote_read()
        // can be called concurrently-in-spirit from any of the 64 per-core
        // DMA sc_modules without needing lru_mutex_ (SystemC itself
        // serializes process execution, but these are plain counters with no
        // other invariant to protect, so a dedicated mutex would be
        // overkill).
        std::atomic<uint64_t> eviction_count_{0};
        std::atomic<uint64_t> remote_read_count_{0};
        std::atomic<uint64_t> offchip_fill_count_{0};
        std::atomic<uint64_t> eviction_count_kv_{0};
        std::atomic<uint64_t> eviction_count_weight_{0};

        // Same residency check as the public is_resident(), but callable
        // while already holding lru_mutex_ (single acquisition for a whole
        // group scan) -- is_resident() cannot be reused here since it takes
        // lru_mutex_ itself via has_capacity_limit()/capacity_for(), and
        // std::mutex is not recursive.
        bool resident_locked(int core_id, const std::string& path) const;  // caller holds lru_mutex_
};

#endif // LLM_SIM_DATA_MANGER_H
