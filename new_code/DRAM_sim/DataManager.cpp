#include "DRAM_sim/DataManager.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <yaml-cpp/yaml.h>

template <typename T>
DataManager<T>::DataManager(bool compute) : compute_(compute) {
    init_kv_files();
}

template <typename T>
DataManager<T>::DataManager(const DataManagerConfig& cfg) : compute_(cfg.compute) {
    if (!cfg.dram_map_path.empty()) {
        address_mapper_.load(cfg.dram_map_path);
    }
    // ENABLE_DYNAMIC_DISPATCH: CA_2026_FP student handout keeps only
    // dram_map_path live at runtime. The group/offchip/fallback mapping
    // mechanisms below are disabled here (their sole DataManager-side
    // load call sites) rather than deleted -- see federated-skipping-
    // Front A. Undefine (default) means these three blocks are
    // permanently skipped even if a caller passes a non-empty path.
#ifdef ENABLE_DYNAMIC_DISPATCH
    if (!cfg.dram_group_path.empty()) {
        address_mapper_.load_groups(cfg.dram_group_path);
    }
    if (!cfg.offchip_map_path.empty()) {
        offchip_mapper_.load(cfg.offchip_map_path);
    }
    if (!cfg.fallback_map_path.empty()) {
        fallback_mapper_.load(cfg.fallback_map_path);
    }
#endif // ENABLE_DYNAMIC_DISPATCH
    // Phase 7a: seed per-core capacity limits. lru_states_ entries are only
    // created here for cores with a nonzero capacity (default or override)
    // -- capacity_for()/has_capacity_limit() treat an absent entry as "no
    // limit configured", so this is safe to leave sparse.
    local_dram_capacity_default_ = cfg.local_dram_capacity_bytes;
    local_dram_capacity_overrides_ = cfg.local_dram_capacity_overrides;
    skip_group_remote_ = cfg.skip_group_remote;
    build_group_members();
    init_kv_files();
    // warm_start() unconditionally
    // touch_lru()'s EVERY owner of EVERY dram_map.yaml entry -- for ADP's
    // tier-ladder union ownership (gen_real_prefill_128rp.py registers every
    // tier position for every expert, not just the one this layer's demand
    // actually selected), that re-marks every previously-evicted tier-ladder
    // path as resident again for free, right before any kernel read
    // instruction ever executes. That made DataManager::is_resident() return
    // true unconditionally at read time, so the local-miss branch in
    // DRAMSim::do_read() (the only place a genuine off-chip refetch can be
    // triggered from a local read) could never actually fire -- eviction_
    // count() kept growing every layer, but offchip_fill_count() stayed at
    // 0 across every multi-layer chain tested, even ones with 10-19 experts
    // changing tier between consecutive layers. load_residency() itself is
    // NOT the problem: it already restores exactly the true prior-layer LRU
    // state (including which paths were genuinely evicted), verified via a
    // separate walk of dram_residency_out.yaml's own written entries.
    //
    // Fixing the above (only run the full warm_start() on a true cold start,
    // trust load_residency() otherwise) surfaced a SECOND, pre-existing
    // issue: genuine broadcast files -- e.g. post_attn_norm_weight.bin,
    // owned by all 64 cores, not part of any per-expert tier ladder -- then
    // had to pay a real per-core off-chip fetch on every chained layer
    // (confirmed via an isolated single-layer rerun: 400+ [offchip_fetch]
    // events for ONE broadcast path within the first 700us of sim time,
    // one core at a time, serialized). Every core legitimately needs its
    // own resident copy of a true broadcast file every layer -- that was
    // never a real "off-chip demand event" to measure, just collateral
    // damage from the same fix. So: run warm_start(broadcast_only=true) on
    // EVERY layer (chained or not) to keep genuine broadcast files free,
    // and additionally run the ORIGINAL full warm_start() only on a true
    // cold start (no residency to load).
    warm_start(/*broadcast_only=*/true);
    // init-dram-mapping-dram-fetching-nested-blum plan: pre-seed TP4
    // replica=1/2 shard-cores (dram_map.yaml's warm_seed_owners, only
    // populated under gen_prefill_dp_adaptive(tp4_only=True)) as ordinary,
    // still-evictable LRU entries -- runs every layer, BEFORE load_residency
    // ()/cold-start warm_start() below, so genuinely-recently-used data ends
    // up more-recent than this speculative pre-seed (see warm_start_extra()'s
    // header comment for why that ordering matters under capacity pressure).
    // A no-op when no path is tagged warm_seed_owners (e.g. tp4_only=False
    // or non-ADP runs), so this is safe to call unconditionally.
    warm_start_extra();
    // ENABLE_DYNAMIC_DISPATCH: residency chaining (Phase G) disabled for
    // the student handout -- always cold-start via warm_start(). See
    // Front A4.
#ifdef ENABLE_DYNAMIC_DISPATCH
    if (!cfg.dram_residency_in_path.empty()) {
        load_residency(cfg.dram_residency_in_path);
    } else {
        warm_start();
    }
#else
    warm_start();
#endif // ENABLE_DYNAMIC_DISPATCH
}

// ---------------------------------------------------------------------------
// build_group_members -- Phase C: precompute
// each core's dram_group.bin neighborhood once, up front. address_mapper_.
// num_cores() is 0 if load_groups() was never called (no dram_group_path
// given), in which case group_members_ stays empty and
// find_resident_group_member() naturally treats every core as having no
// group (its own is_group() would also return false for every pair in that
// case, so this is exactly equivalent, just precomputed).
// ---------------------------------------------------------------------------
template <typename T>
void DataManager<T>::build_group_members() {
    const int n = address_mapper_.num_cores();
    for (int i = 0; i < n; ++i) {
        std::vector<int> members;
        for (int j = 0; j < n; ++j) {
            if (j != i && address_mapper_.is_group(i, j)) members.push_back(j);
        }
        if (!members.empty()) group_members_[i] = std::move(members);
    }
}

// ---------------------------------------------------------------------------
// Private: per-file mutex registry
// ---------------------------------------------------------------------------
template <typename T>
std::mutex& DataManager<T>::get_file_mutex(const std::string& path) {
    std::lock_guard<std::mutex> map_lk(map_mutex_);
    auto it = file_mutexes_.find(path);
    if (it == file_mutexes_.end()) {
        auto [ins, _] = file_mutexes_.emplace(path, std::make_unique<std::mutex>());
        return *ins->second;
    }
    return *it->second;
}

// ---------------------------------------------------------------------------
// read_data
//   Converts DRAMAddr_t row/col indices into byte offsets within the binary
//   file (row-major layout, col_end == total columns == row stride) and reads
//   the submatrix [row_start:row_end, col_start:col_end] row by row.
//   Returns a flat 1D vector in row-major order.
// ---------------------------------------------------------------------------
template <typename T>
std::vector<T> DataManager<T>::read_data(const LLM_SIM_DRAM::DRAMAddrPtr_t& addr_ptr,
                                          int core_id) {
    const uint64_t n_rows = (addr_ptr->row_end > addr_ptr->row_start)
        ? static_cast<uint64_t>(addr_ptr->row_end - addr_ptr->row_start) : 0;
    const uint64_t n_cols = (addr_ptr->col_end > addr_ptr->col_start)
        ? static_cast<uint64_t>(addr_ptr->col_end - addr_ptr->col_start) : 0;
    if (n_rows == 0 || n_cols == 0) return {};

    // Phase 7a: touch LRU state (if core_id has capacity tracking enabled)
    // BEFORE the miss check below, matching normal LRU semantics (a read
    // that finds the entry resident still counts as "just used"). Actual
    // eviction only fires when this touch causes used_bytes to exceed
    // capacity_bytes (see touch_lru/evict_if_needed).
    if (core_id >= 0 && capacity_for(core_id) > 0) {
        touch_lru(core_id, addr_ptr->file_path, logical_bytes(addr_ptr->file_path, n_rows * n_cols));
    }

    // Transaction-only mode: skip file I/O and return zeros (padded size if requested).
    if (!compute_) {
        const uint64_t out_cols = (addr_ptr->pad_to > 0 && static_cast<uint64_t>(addr_ptr->pad_to) > n_cols)
            ? static_cast<uint64_t>(addr_ptr->pad_to) : n_cols;
        return std::vector<T>(n_rows * out_cols, T{});
    }

    const std::string& path      = addr_ptr->file_path;

    const int64_t      row_start = addr_ptr->row_start;
    const int64_t      row_end   = addr_ptr->row_end;
    const int64_t      col_start = addr_ptr->col_start;
    const int64_t      col_end   = addr_ptr->col_end;   // end column (exclusive) of the access range

    const uint64_t row_stride = addr_ptr->row_stride;  // total cols in file (byte-offset multiplier)

    std::lock_guard<std::mutex> file_lk(get_file_mutex(path));

    std::ifstream fin(path, std::ios::in | std::ios::binary);
    if (!fin.is_open()) {
        std::cerr << "[DataManager] read_data: cannot open file: " << path << std::endl;
        return {};
    }

    std::vector<T> result;
    result.reserve(n_rows * n_cols);

    for (int64_t r = row_start; r < row_end; ++r) {
        if (r < 0 || col_start < 0) {
            // Out-of-range rows/cols are treated as implicit zero padding.
            result.insert(result.end(), n_cols, static_cast<T>(0));
            continue;
        }

        // byte offset of element (r, col_start): row_stride is the actual row width in the file
        const uint64_t byte_offset =
            (static_cast<uint64_t>(r) * row_stride + static_cast<uint64_t>(col_start)) * sizeof(T);

        fin.seekg(static_cast<std::streamoff>(byte_offset), std::ios::beg);
        if (!fin) {
            std::cerr << "[DataManager] read_data: seekg failed: " << path
                      << " row=" << r << " byte_offset=" << byte_offset << std::endl;
            return {};
        }

        std::vector<T> row_buf(n_cols);
        fin.read(reinterpret_cast<char*>(row_buf.data()),
                 static_cast<std::streamsize>(n_cols * sizeof(T)));
        if (!fin) {
            std::cerr << "[DataManager] read_data: read failed: " << path
                      << " row=" << r << " n_cols=" << n_cols << std::endl;
            return {};
        }

        result.insert(result.end(), row_buf.begin(), row_buf.end());

        // Zero-pad this row to pad_to columns if requested.
        if (addr_ptr->pad_to > 0 && addr_ptr->pad_to > n_cols) {
            const uint64_t pad_elems = static_cast<uint64_t>(addr_ptr->pad_to) - n_cols;
            result.insert(result.end(), pad_elems, static_cast<T>(0));
        }
    }

    return result;
}

// ---------------------------------------------------------------------------
// write_data
//   Writes the flat 1D vector (row-major order) to the submatrix
//   [row_start:row_end, col_start:col_end] within the binary file.
//   Creates the file if it does not exist (directory must already exist).
// ---------------------------------------------------------------------------
template <typename T>
void DataManager<T>::write_data(const LLM_SIM_DRAM::DRAMAddrPtr_t& addr_ptr,
                                const std::vector<T>& data, int core_id,
                                int elem_bits) {
    if (data.empty()) return;
    // Transaction-only mode: skip file I/O entirely.
    if (!compute_) return;

    // Phase 7a: touch LRU state (if core_id has capacity tracking enabled).
    // A write always makes this core the most-recent toucher of the entry.
    if (core_id >= 0 && capacity_for(core_id) > 0) {
        touch_lru(core_id, addr_ptr->file_path,
                  logical_bytes(addr_ptr->file_path, data.size(), elem_bits));
    }

    const std::string& path      = addr_ptr->file_path;
    const int64_t      row_start = addr_ptr->row_start;
    const int64_t      row_end   = addr_ptr->row_end;
    const int64_t      col_start = addr_ptr->col_start;
    const int64_t      col_end   = addr_ptr->col_end;   // end column (exclusive) of the access range

    const uint64_t row_stride = addr_ptr->row_stride;  // total cols in file (byte-offset multiplier)
    const uint64_t n_rows = (row_end > row_start) ? static_cast<uint64_t>(row_end - row_start) : 0;
    const uint64_t n_cols = (col_end > col_start) ? static_cast<uint64_t>(col_end - col_start) : 0;

    if (n_rows == 0 || n_cols == 0) return;

    std::lock_guard<std::mutex> file_lk(get_file_mutex(path));

    // Ensure destination directory exists for nested cache paths.
    std::error_code ec;
    std::filesystem::path fs_path(path);
    auto parent = fs_path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent, ec);
    }

    // Open existing file for random read+write; create it first if absent.
    std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
    if (!f.is_open()) {
        { std::ofstream tmp(path, std::ios::out | std::ios::binary); }
        f.open(path, std::ios::in | std::ios::out | std::ios::binary);
    }
    if (!f.is_open()) {
        std::cerr << "[DataManager] write_data: cannot open file: " << path << std::endl;
        return;
    }

    for (uint64_t r = 0; r < n_rows; ++r) {
        const int64_t logical_row = row_start + static_cast<int64_t>(r);

        // Negative row/col means out-of-range padding region; skip file writes.
        if (logical_row < 0 || col_start < 0) {
            continue;
        }

        // Guard against OOB access when the provided data vector is smaller
        // than the full (n_rows × n_cols) region (e.g. due to a send/recv size
        // mismatch where fewer rows were actually transmitted).
        if ((r + 1) * n_cols > data.size()) {
            std::cerr << "[DataManager] write_data: data size mismatch: "
                      << path << " row=" << logical_row
                      << " expected at least " << (r + 1) * n_cols
                      << " elements but got " << data.size() << std::endl;
            continue;
        }

        // byte offset of element (logical_row, col_start): row_stride is the actual row width
        const uint64_t byte_offset =
            (static_cast<uint64_t>(logical_row) * row_stride + static_cast<uint64_t>(col_start)) * sizeof(T);

        f.seekp(static_cast<std::streamoff>(byte_offset), std::ios::beg);
        if (!f) {
            std::cerr << "[DataManager] write_data: seekp failed: " << path
                      << " row=" << logical_row << " byte_offset=" << byte_offset << std::endl;
            return;
        }

        // Row r of the input vector starts at index r * n_cols
        f.write(reinterpret_cast<const char*>(data.data() + r * n_cols),
                static_cast<std::streamsize>(n_cols * sizeof(T)));
        if (!f) {
            std::cerr << "[DataManager] write_data: write failed: " << path
                      << " row=" << logical_row << " n_cols=" << n_cols << std::endl;
            return;
        }
    }

    f.flush();
}

// ---------------------------------------------------------------------------
// init_kv_files (only active for T == float)
//
// For each layer n in [0, NUM_LAYERS), creates K_n.bin and V_n.bin inside
// K_file_path and V_file_path respectively if they do not already exist.
// Each file is created as a sparse file whose size equals
//   BATCH * HEAD_NUM * MAX_SEQ_LEN * HEAD_DIM * sizeof(T) bytes
// so that unwritten regions read back as zero (T(0)).
// ---------------------------------------------------------------------------
template <typename T>
void DataManager<T>::init_kv_files() {
    if constexpr (!std::is_same_v<T, float>) return;  // KV cache is float only
    // Transaction-only mode: skip KV cache file creation.
    if (!compute_) return;

    using namespace LLM_SIM_DRAM;

    const uint64_t file_size =
        static_cast<uint64_t>(BATCH) * HEAD_NUM * MAX_SEQ_LEN * HEAD_DIM * sizeof(T);

    // Create a sparse file of exactly file_size bytes (Linux: gap reads as zeros).
    auto create_sparse = [&](const std::string& path) {
        std::error_code ec;
        std::filesystem::path fs_path(path);
        auto parent = fs_path.parent_path();
        if (!parent.empty()) {
            std::filesystem::create_directories(parent, ec);
        }

        // Skip if the file already exists.
        if (std::ifstream(path, std::ios::binary).good()) return;

        // Create an empty file, then seek to the last byte and write one zero
        // so the filesystem allocates the correct file size sparsely.
        { std::ofstream tmp(path, std::ios::out | std::ios::binary); }
        std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
        if (!f.is_open()) {
            std::cerr << "[DataManager] init_kv_files: cannot create: " << path << "\n";
            return;
        }
        f.seekp(static_cast<std::streamoff>(file_size - 1), std::ios::beg);
        const char zero = '\0';
        f.write(&zero, 1);
        if (!f)
            std::cerr << "[DataManager] init_kv_files: seekp/write failed: " << path << "\n";
    };

    for (int n = 0; n < NUM_LAYERS; ++n) {
        create_sparse(K_file_path + "/K_" + std::to_string(n) + ".bin");
        create_sparse(V_file_path + "/V_" + std::to_string(n) + ".bin");
    }
}

// ---------------------------------------------------------------------------
// Phase 7a/7b/7c: LRU/capacity tracking, partitioned by core_id.
// ---------------------------------------------------------------------------
template <typename T>
uint64_t DataManager<T>::capacity_for(int core_id) const {
    std::lock_guard<std::mutex> lk(lru_mutex_);
    auto it = local_dram_capacity_overrides_.find(core_id);
    if (it != local_dram_capacity_overrides_.end()) return it->second;
    return local_dram_capacity_default_;
}

// Lock-free variant for callers that already hold lru_mutex_ (touch_lru).
// capacity_for() cannot be reused there -- it takes lru_mutex_ itself,
// and std::mutex is not recursive.
template <typename T>
uint64_t DataManager<T>::capacity_for_locked(int core_id) const {
    auto it = local_dram_capacity_overrides_.find(core_id);
    if (it != local_dram_capacity_overrides_.end()) return it->second;
    return local_dram_capacity_default_;
}

template <typename T>
bool DataManager<T>::has_capacity_limit(int core_id) const {
    return capacity_for(core_id) > 0;
}

template <typename T>
bool DataManager<T>::skip_group_remote(int /*core_id*/) const {
    return skip_group_remote_;
}

template <typename T>
int DataManager<T>::elem_bits_for(const std::string& path) const {
    return address_mapper_.elem_bits_of(path);
}

template <typename T>
uint64_t DataManager<T>::logical_bytes(const std::string& path, uint64_t n_elems) const {
    // elem_bits_of()==4 is used as the "is this path tagged mxfp4" signal --
    // deliberately reusing AddressMapper's single source of truth (the
    // format field) rather than a second independent format check, since 4
    // is only ever produced for an mxfp4-tagged path (see elem_bits_of()).
    if (address_mapper_.elem_bits_of(path) == 4) {
        return ((n_elems + 31) / 32) * 17ULL;
    }
    // (2026-08-14, Bug H
    // continuation): F16 has no block/scale encoding like MXFP4 (unlike
    // elem_bits_of()==4's real 4.25 bits/element average, F16 is exactly
    // 2 bytes/element, no approximation gap to correct for) -- straight
    // n_elems*2, mirroring the same "16 is only ever produced for an
    // f16-tagged path" reuse of AddressMapper's single source of truth.
    if (address_mapper_.elem_bits_of(path) == 16) {
        return n_elems * 2ULL;
    }
    return n_elems * 4ULL;
}

template <typename T>
uint64_t DataManager<T>::logical_bytes(const std::string& path, uint64_t n_elems,
                                        int producer_elem_bits) const {
    const int tag_elem_bits = address_mapper_.elem_bits_of(path);
    if (tag_elem_bits != 32) {
        // Path is format-tagged (mxfp4/f16) -- tag remains sole authority,
        // identical to the path-only overload above.
        return logical_bytes(path, n_elems);
    }
    if (producer_elem_bits != 32 && producer_elem_bits > 0) {
        return (n_elems * static_cast<uint64_t>(producer_elem_bits) + 7ULL) / 8ULL;
    }
    return n_elems * 4ULL;
}

template <typename T>
uint64_t DataManager<T>::used_bytes(int core_id) const {
    std::lock_guard<std::mutex> lk(lru_mutex_);
    auto it = lru_states_.find(core_id);
    return it == lru_states_.end() ? 0 : it->second.used_bytes;
}

template <typename T>
bool DataManager<T>::is_resident(int core_id, const std::string& path) const {
    if (!has_capacity_limit(core_id)) return true;  // tracking not enabled -> always resident
    std::lock_guard<std::mutex> lk(lru_mutex_);
    auto st_it = lru_states_.find(core_id);
    if (st_it == lru_states_.end()) return false;  // configured but never touched yet
    return st_it->second.index.find(path) != st_it->second.index.end();
}

// Phase C: same check as is_resident() above, but assumes the caller
// already holds lru_mutex_ (see this method's header comment in
// DataManager.h for why is_resident() itself cannot be reused for a
// multi-core group scan).
template <typename T>
bool DataManager<T>::resident_locked(int core_id, const std::string& path) const {
    if (capacity_for_locked(core_id) == 0) return true;  // tracking not enabled -> always resident
    auto st_it = lru_states_.find(core_id);
    if (st_it == lru_states_.end()) return false;  // configured but never touched yet
    return st_it->second.index.find(path) != st_it->second.index.end();
}

template <typename T>
int DataManager<T>::find_resident_group_member(int core_id, const std::string& path) const {
    auto it = group_members_.find(core_id);
    if (it == group_members_.end()) return LLM_SIM_DRAM::AddressMapper::OWNER_LOCAL;
    std::lock_guard<std::mutex> lk(lru_mutex_);
    for (int j : it->second) {
        if (resident_locked(j, path)) return j;
    }
    return LLM_SIM_DRAM::AddressMapper::OWNER_LOCAL;
}

template <typename T>
void DataManager<T>::touch_lru(int core_id, const std::string& path, uint64_t size_bytes) {
    std::lock_guard<std::mutex> lk(lru_mutex_);
    LruState& state = lru_states_[core_id];
    if (state.capacity_bytes == 0) {
        state.capacity_bytes = capacity_for_locked(core_id);
    }

    auto idx_it = state.index.find(path);
    if (idx_it != state.index.end()) {
        // Hit: move to front (most-recently-used), size unchanged.
        state.order.splice(state.order.begin(), state.order, idx_it->second);
        return;
    }

    // New entry: insert at front, account for its size, then evict from the
    // tail if this pushes used_bytes over capacity_bytes.
    const bool is_kv = address_mapper_.is_initial_kv(path);
    const auto no_evict_owners = address_mapper_.no_evict_owners_of(path);
    const bool no_evict = std::find(no_evict_owners.begin(), no_evict_owners.end(), core_id)
        != no_evict_owners.end();
    state.order.push_front(LruEntry{path, size_bytes, is_kv, no_evict});
    state.index[path] = state.order.begin();
    state.used_bytes += size_bytes;

    evict_if_needed(core_id, state);
}

template <typename T>
void DataManager<T>::evict_if_needed(int core_id, LruState& state) {
    while (state.used_bytes > state.capacity_bytes && !state.order.empty()) {
        // scan from the tail (least-
        // recently-used first, preserving normal LRU victim preference)
        // for the first entry NOT marked no_evict, leaving every skipped
        // protected entry in its existing list position untouched. If the
        // entire list is protected, the outer while condition keeps
        // retrying forever with no progress -- guard against that
        // explicitly rather than looping; staying over capacity in that
        // case is the caller's responsibility (see no_evict's doc comment).
        auto victim_it = state.order.end();
        for (auto it = state.order.rbegin(); it != state.order.rend(); ++it) {
            if (!it->no_evict) {
                victim_it = std::next(it).base();
                break;
            }
        }
        if (victim_it == state.order.end()) break;  // every remaining entry is protected

        LruEntry victim = *victim_it;
        state.order.erase(victim_it);
        state.index.erase(victim.path);
        state.used_bytes -= victim.size_bytes;
        eviction_count_.fetch_add(1, std::memory_order_relaxed);
        (victim.is_kv ? eviction_count_kv_ : eviction_count_weight_)
            .fetch_add(1, std::memory_order_relaxed);

        // opt-in eviction-victim
        // trace for diagnosing exactly what a run's evictions consist of
        // (path/core_id/size), gated behind an env var so it stays silent by
        // default (this fires per-eviction, potentially thousands of times).
        if (std::getenv("DEBUG_EVICTION_TRACE")) {
            std::cerr << "[evict] core=" << core_id << " kv=" << victim.is_kv
                       << " bytes=" << victim.size_bytes << " path=" << victim.path
                       << "\n";
        }

        if (victim.is_kv) {
            // Phase 7c: KV entries need write-back to their off-chip owner
            // before being dropped -- weight entries (is_kv == false) are
            // read-only and simply disappear (already resident off-chip,
            // never modified). Mark (core_id, path) as "eviction in
            // flight" so DRAMSim's group-read path (Phase 3b) knows to
            // requeue rather than service a read against data that's
            // about to move -- keyed by core_id too (not path alone) so
            // evicting a broadcast-owned path on one core never blocks
            // every other core that also owns it (see is_evicting()'s
            // header comment).
            evicting_paths_.insert({core_id, victim.path});

            // Phase A fix: read back the
            // FULL on-disk file, not a region derived from victim.size_bytes.
            // touch_lru's hit branch never grows size_bytes on repeat
            // touches of the same path (see below), so size_bytes only
            // ever reflects whichever single touch happened to create this
            // LRU entry -- a DRAMAddr_t built from it (as the old code did:
            // row 0, cols [0, size_bytes/sizeof(T))) targets the wrong
            // region for any file touched at a nonzero row/col offset (the
            // common case for K/V cache tensor-mode accesses). Querying the
            // real file size and reading it back as one flat row recovers
            // the file's true current contents regardless of which slice
            // was last touched (mirrors how DMA's start_offchip_fetch
            // itself synthesizes a flat row-0 access for files it isn't
            // otherwise slicing -- DMA.cpp).
            std::error_code ec;
            const auto file_bytes = std::filesystem::file_size(victim.path, ec);
            const int64_t n_elems = (!ec && file_bytes > 0)
                ? static_cast<int64_t>(file_bytes / sizeof(T)) : 0;

            std::vector<T> data;
            if (n_elems > 0) {
                auto addr = std::make_shared<LLM_SIM_DRAM::DRAMAddr_t>(
                    victim.path, 0, 1, 0, n_elems);
                // Safe to call here (holding lru_mutex_, not file_mutexes_):
                // read_data() takes its own file mutex internally and does
                // not touch lru_mutex_ itself when core_id is default (-1),
                // so there is no recursive-lock risk.
                data = read_data(addr, /*core_id=*/-1);
            }

            pending_writebacks_[core_id].push_back(
                EvictedKv{core_id, victim.path, std::move(data)});
        }
    }
}

template <typename T>
bool DataManager<T>::is_evicting(int core_id, const std::string& path) const {
    std::lock_guard<std::mutex> lk(lru_mutex_);
    return evicting_paths_.find({core_id, path}) != evicting_paths_.end();
}

template <typename T>
void DataManager<T>::clear_evicting(int core_id, const std::string& path) {
    std::lock_guard<std::mutex> lk(lru_mutex_);
    evicting_paths_.erase({core_id, path});
}

template <typename T>
bool DataManager<T>::pop_pending_writeback(int core_id, EvictedKv& out) {
    std::lock_guard<std::mutex> lk(lru_mutex_);
    auto it = pending_writebacks_.find(core_id);
    if (it == pending_writebacks_.end() || it->second.empty()) return false;
    out = std::move(it->second.front());
    it->second.pop_front();
    return true;
}

// ---------------------------------------------------------------------------
// warm_start -- Phase D: seed each owning
// core's LRU state with the files dram_map.yaml says it owns. Phase E: sized
// by logical_bytes(), not raw on-disk file size -- an mxfp4-tagged file's
// physical bytes (always fp32, sizeof(T)*n) overstate its logical HBM
// footprint by ~4x, which would make capacity accounting inconsistent with
// every other logical_bytes()-based touch_lru call (read_data/write_data).
// file_bytes/sizeof(T) recovers the element count from the physical file
// size (files are always stored as fp32 regardless of format tag -- see
// Decision 5), then logical_bytes() converts
// that count using path's actual format. Files not yet on disk (ec set, or
// a ranges-only entry with no whole-file owners) are skipped, not fatal --
// warm start is best-effort bookkeeping, not a correctness precondition.
// Cores without a configured capacity limit are left untouched entirely
// (touch_lru only creates an LruState for cores that actually have one,
// same as any normal read/write touch).
// ---------------------------------------------------------------------------
template <typename T>
void DataManager<T>::warm_start(bool broadcast_only) {
    // the owner-count histogram
    // for a real ADP prefill layer is bimodal with a huge gap -- {1,4,5,6}
    // owners (TP1/TP2/TP4(+R) tier-ladder positions) vs 64 owners (genuine
    // broadcast: every core legitimately needs its own copy of e.g. the
    // per-layer norm/router weights, independent of any tiering decision).
    // Half of num_cores() is a safe, conservative cut between the two --
    // R_MAX=4 (the largest tier-ladder shard count seen in practice) is
    // nowhere close to num_cores()/2 for any realistic core count.
    //
    // PER-OWNER (not per-entry) TP1-base exception: a single dram_map.yaml
    // entry can have a MIX of always-resident and demand-acquired owners --
    // e.g. ffn_gate_exps_e0_bias.bin's owners=[0,32,1,2,3]: core 0 is
    // expert 0's genuine TP1 base position (hardware-out-of-the-box
    // resident), while 32/1/2/3 are TP2/TP4 shard cores sharing the SAME
    // r=0 base file path but must NOT be seeded for free (they only exist
    // because ADP's tier-ladder union registered them; whether this owner
    // is ever touched at all depends on this layer's real demand). Checking
    // owners.size() alone (this entry has 5 owners, not 1) would incorrectly
    // exclude core 0 too. dram_map_writer.py's always_resident_owners field
    // (set by gen_core_2_1024.py only for core_id < N_EXPERTS -- the
    // documented TP1 layout, config_real.py: "Expert core: c < N_EXPERTS")
    // carries the correct per-owner answer through from generation time.
    const int broadcast_threshold = std::max(1, address_mapper_.num_cores() / 2);
    for (const auto& [path, owners] : address_mapper_.all_entries()) {
        if (owners.empty()) continue;  // ranges-only entry, nothing to seed
        const bool is_broadcast_entry =
            static_cast<int>(owners.size()) >= broadcast_threshold;
        const std::vector<int> always_resident =
            broadcast_only && !is_broadcast_entry
                ? address_mapper_.always_resident_owners_of(path)
                : std::vector<int>{};
        std::error_code ec;
        const auto file_bytes = std::filesystem::file_size(path, ec);
        if (ec || file_bytes == 0) continue;
        const uint64_t n_elems = static_cast<uint64_t>(file_bytes) / sizeof(T);
        const uint64_t bytes = logical_bytes(path, n_elems);
        for (int owner : owners) {
            if (owner < 0 || capacity_for(owner) == 0) continue;
            if (broadcast_only && !is_broadcast_entry &&
                std::find(always_resident.begin(), always_resident.end(), owner)
                    == always_resident.end()) continue;
            touch_lru(owner, path, bytes);
        }
    }
}

template <typename T>
void DataManager<T>::warm_start_extra() {
    // init-dram-mapping-dram-fetching-nested-blum plan: same all_entries()
    // walk as warm_start(), but gated on warm_seed_owners_of() instead of
    // always_resident_owners_of(), and calling plain touch_lru() with no
    // eviction-exemption bookkeeping -- these entries are ordinary LRU
    // residents from this call onward, indistinguishable from one seeded by
    // a real kernel read. Unconditional (no broadcast_only split): broadcast
    // entries are never tagged with warm_seed_owners (gen_prefill_dp_
    // adaptive only tags TP4 replica shard-cores), so there is no
    // broadcast/tier-ladder ambiguity to resolve here like warm_start() has.
    for (const auto& [path, owners] : address_mapper_.all_entries()) {
        if (owners.empty()) continue;  // ranges-only entry, nothing to seed
        const std::vector<int> warm_seed = address_mapper_.warm_seed_owners_of(path);
        if (warm_seed.empty()) continue;
        std::error_code ec;
        const auto file_bytes = std::filesystem::file_size(path, ec);
        if (ec || file_bytes == 0) continue;
        const uint64_t n_elems = static_cast<uint64_t>(file_bytes) / sizeof(T);
        const uint64_t bytes = logical_bytes(path, n_elems);
        for (int owner : warm_seed) {
            if (owner < 0 || capacity_for(owner) == 0) continue;
            touch_lru(owner, path, bytes);
        }
    }
}

template <typename T>
uint64_t DataManager<T>::eviction_count() const {
    return eviction_count_.load(std::memory_order_relaxed);
}

template <typename T>
uint64_t DataManager<T>::remote_read_count() const {
    return remote_read_count_.load(std::memory_order_relaxed);
}

template <typename T>
void DataManager<T>::note_remote_read() {
    remote_read_count_.fetch_add(1, std::memory_order_relaxed);
}

template <typename T>
uint64_t DataManager<T>::offchip_fill_count() const {
    return offchip_fill_count_.load(std::memory_order_relaxed);
}

template <typename T>
uint64_t DataManager<T>::eviction_count_kv() const {
    return eviction_count_kv_.load(std::memory_order_relaxed);
}

template <typename T>
uint64_t DataManager<T>::eviction_count_weight() const {
    return eviction_count_weight_.load(std::memory_order_relaxed);
}

template <typename T>
void DataManager<T>::install_resident(int core_id, const std::string& path, uint64_t n_elems) {
    // install_resident() has TWO
    // call sites, both firing exactly once per completed real off-chip wire
    // transfer (DMA::start_offchip_fetch's NoC round trip actually landing):
    // DRAMSim::fill_thread (a core's OWN local-miss fill) and
    // DMA::tick_group_cache_install (a remote group-read that was already a
    // known/discovered miss, so it went straight to off-chip via
    // start_group_miss_fetch -> start_offchip_fetch instead of round-
    // tripping a doomed remote read first). warm_start() never calls this
    // (it uses touch_lru() directly for the zero-cost initial seed, not a
    // real fetch) so counting every install_resident() call here is exactly
    // "how many times did this simulation actually pull data across the
    // NoC from off-chip DRAM" -- not eviction_count() (counts local
    // capacity kick-outs, not refetches) or remote_read_count() (counts
    // remote read DISPATCH attempts, before hit/miss is even known).
    const uint64_t seq = offchip_fill_count_.fetch_add(1, std::memory_order_relaxed) + 1;
    std::cout << "[offchip_fetch] #" << seq << " core=" << core_id
              << " bytes=" << logical_bytes(path, n_elems) << " path=" << path
              << "\n";
    touch_lru(core_id, path, logical_bytes(path, n_elems));
}

template <typename T>
bool DataManager<T>::is_fill_inflight(int core_id, const std::string& path) const {
    std::lock_guard<std::mutex> lk(lru_mutex_);
    return fill_inflight_.find({core_id, path}) != fill_inflight_.end();
}

template <typename T>
void DataManager<T>::mark_fill_inflight(int core_id, const std::string& path) {
    std::lock_guard<std::mutex> lk(lru_mutex_);
    fill_inflight_.insert({core_id, path});
}

template <typename T>
void DataManager<T>::clear_fill_inflight(int core_id, const std::string& path) {
    std::lock_guard<std::mutex> lk(lru_mutex_);
    fill_inflight_.erase({core_id, path});
}

// ---------------------------------------------------------------------------
// dump_residency / load_residency -- Phase G:
// cross-process residency chaining. See both methods' header comments in
// DataManager.h for the schema and the load-before-warm_start ordering
// rationale.
// ---------------------------------------------------------------------------
template <typename T>
void DataManager<T>::dump_residency(const std::string& out_path) const {
    std::lock_guard<std::mutex> lk(lru_mutex_);

    YAML::Emitter out;
    out << YAML::BeginMap;
    out << YAML::Key << "version" << YAML::Value << 1;
    out << YAML::Key << "cores" << YAML::Value << YAML::BeginSeq;
    for (const auto& [core_id, state] : lru_states_) {
        out << YAML::BeginMap;
        out << YAML::Key << "core_id" << YAML::Value << core_id;
        out << YAML::Key << "entries" << YAML::Value << YAML::BeginSeq;
        for (const auto& entry : state.order) {  // front-to-back == MRU-to-LRU
            out << YAML::BeginMap;
            out << YAML::Key << "path" << YAML::Value << entry.path;
            out << YAML::Key << "size_bytes" << YAML::Value << entry.size_bytes;
            out << YAML::EndMap;
        }
        out << YAML::EndSeq;
        out << YAML::EndMap;
    }
    out << YAML::EndSeq;
    out << YAML::EndMap;

    std::filesystem::create_directories(
        std::filesystem::path(out_path).parent_path());
    std::ofstream f(out_path);
    f << out.c_str() << "\n";
}

template <typename T>
void DataManager<T>::load_residency(const std::string& in_path) {
    YAML::Node root = YAML::LoadFile(in_path);  // throws YAML::BadFile if missing
    if (!root["cores"]) return;

    std::lock_guard<std::mutex> lk(lru_mutex_);
    for (const auto& core_node : root["cores"]) {
        const int core_id = core_node["core_id"].as<int>();
        LruState& state = lru_states_[core_id];
        if (state.capacity_bytes == 0) {
            state.capacity_bytes = capacity_for_locked(core_id);
        }
        if (!core_node["entries"]) continue;
        for (const auto& entry_node : core_node["entries"]) {
            const std::string path = entry_node["path"].as<std::string>();
            const uint64_t size_bytes = entry_node["size_bytes"].as<uint64_t>();
            if (state.index.find(path) != state.index.end()) continue;  // shouldn't happen, defensive
            // push_back (not touch_lru's push_front): preserves the dump's
            // front-to-back MRU-to-LRU order exactly, since this is a fresh
            // LruState with nothing in it yet -- see this method's header
            // comment in DataManager.h. is_kv=false unconditionally (see
            // same comment for why a stale KV entry from a finished layer
            // needs no write-back semantics). no_evict is re-derived from
            // AddressMapper (not persisted in the residency dump itself)
            // so protection survives a checkpoint reload exactly like
            // touch_lru's normal path -- see no_evict_owners_of()'s doc
            // comment.
            const auto no_evict_owners = address_mapper_.no_evict_owners_of(path);
            const bool no_evict = std::find(no_evict_owners.begin(), no_evict_owners.end(), core_id)
                != no_evict_owners.end();
            state.order.push_back(LruEntry{path, size_bytes, /*is_kv=*/false, no_evict});
            auto it = state.order.end();
            --it;
            state.index[path] = it;
            state.used_bytes += size_bytes;
        }
        // Defensive: if the loaded residency alone already exceeds this
        // core's capacity (e.g. a smaller budget than a previous run used),
        // evict from the back (oldest) immediately rather than leaving
        // used_bytes > capacity_bytes until the first future touch_lru call.
        evict_if_needed(core_id, state);
    }
}

template class DataManager<int>;
template class DataManager<float>;
