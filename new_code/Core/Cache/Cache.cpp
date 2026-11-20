#include "Core/Cache/Cache.h"

template <typename T>
Cache<T>::Cache(sc_module_name name, const int num_ports) : sc_module(name), num_ports_(num_ports) {
    data_out.init(num_ports);
    data_req.init(num_ports);
    data_in.init(num_ports);
    wreq_in.init(num_ports);
    pending_reqs.resize(num_ports);
    pending_reqs_valid.resize(num_ports);
    pending_wreq.resize(num_ports);
    pending_wreq_valid.resize(num_ports);
    active_wreq.resize(num_ports);
    active_wreq_valid.resize(num_ports);
    wreq_ready_cycle.resize(num_ports);
    SC_THREAD(run);
    sensitive << clk.pos();
}

// ── data_bits ─────────────────────────────────────────────────────────────────
// Total bits described by a write request: product_of_size * elem_bits,
// computed in uint64 throughout (a 32-bit running product would overflow
// well inside today's largest tensors). size must be non-empty with every
// dimension > 0 -- a producer that forgets to tag a real shape (the
// Array/Vector/TMU transaction-only bug this request type exists to catch)
// aborts here instead of silently being treated as "1 element".
template <typename T>
uint64_t Cache<T>::data_bits(const LLM_SIM_INST::CacheWriteReqPtr_t& req) const {
    if (req->size.empty()) {
        std::cout << "[CACHE-WREQ-ERROR] @" << sc_time_stamp() << " mod=" << name()
                  << " id=" << req->id
                  << " write request has empty size -- producer must tag a real shape\n"
                  << std::flush;
        std::abort();
    }
    uint64_t sz = 1;
    for (int d : req->size) {
        if (d <= 0) {
            std::cout << "[CACHE-WREQ-ERROR] @" << sc_time_stamp() << " mod=" << name()
                      << " id=" << req->id << " non-positive dim=" << d
                      << " in write request shape\n" << std::flush;
            std::abort();
        }
        sz *= static_cast<uint64_t>(d);
    }
    return sz * static_cast<uint64_t>(req->elem_bits);
}

// Resident-entry variant: a Data_t already accepted into cache_data always
// carries the shape its write request was validated against (validate_write
// below), so this can reuse the exact same product-of-size * elem_bits
// computation without re-deriving a request object.
template <typename T>
uint64_t Cache<T>::data_bits(const LLM_SIM_DATA::DataPtr_t<T>& data) const {
    uint64_t sz = 1;
    for (int d : data->size_) sz *= static_cast<uint64_t>(d);
    return sz * static_cast<uint64_t>(data->elem_bits);
}

// ── validate_write ───────────────────────────────────────────────────────────
// Commit-time contract check: the Data_t actually read off data_in[i] must
// describe exactly what its admitted write request claimed (same id, same
// shape, same elem_bits, shape's product == value.size()). A mismatch means
// the request and data FIFOs went out of sync (producer wrote them out of
// order, or to the wrong ports) or a declared shape (e.g. Router's
// data_size, DMA's dst_size) disagreed with the real payload -- both are
// gen-time bugs, never a runtime race, so this aborts rather than silently
// trusting whichever side is wrong.
template <typename T>
void Cache<T>::validate_write(int port, const LLM_SIM_INST::CacheWriteReqPtr_t& req,
                               const LLM_SIM_DATA::DataPtr_t<T>& data) const {
    bool ok = (data->id == req->id) && (data->size_ == req->size) &&
              (data->elem_bits == req->elem_bits);
    uint64_t prod = 1;
    if (ok) {
        for (int d : data->size_) prod *= static_cast<uint64_t>(d);
        ok = (prod == static_cast<uint64_t>(data->value.size()));
    }
    if (ok) return;
    std::cout << "[CACHE-WREQ-MISMATCH] @" << sc_time_stamp() << " mod=" << name()
              << " port=" << port
              << " req_id=" << req->id << " data_id=" << data->id
              << " req_size=[";
    for (int s : req->size) std::cout << s << ",";
    std::cout << "] data_size_=[";
    for (int s : data->size_) std::cout << s << ",";
    std::cout << "] req_elem_bits=" << req->elem_bits
              << " data_elem_bits=" << data->elem_bits
              << " shape_prod=" << prod
              << " value.size()=" << data->value.size()
              << " -- write request and data disagree\n" << std::flush;
    std::abort();
}

// ── log_oom ───────────────────────────────────────────────────────────────────
// Permanent (not env-gated) capacity-exhaustion notice: prints once per
// (module, id) the first time a write candidate is denied admission by the
// capacity check, then at most once every OOM_LOG_PERIOD occurrences after
// that, so it stays informative under a sustained pend-and-retry loop
// without flooding stdout every cycle. Cache has no eviction (pend-and-retry
// only -- see Bug H/Round 6-21), so
// this is the single choke point that would otherwise require a dedicated
// DBG_RACE_TRACE run to notice a core is stuck on capacity at all.
template <typename T>
void Cache<T>::log_oom(uint64_t new_bits, uint64_t old_bits) const {
    static thread_local std::unordered_map<uint64_t, uint64_t> streak;
    constexpr uint64_t OOM_LOG_PERIOD = 500000;
    if (++streak[new_bits] % OOM_LOG_PERIOD != 1) return;
    const double cur_mb   = static_cast<double>(current_usage_bits_) / 8.0 / (1024.0 * 1024.0);
    const double write_mb = static_cast<double>(new_bits) / 8.0 / (1024.0 * 1024.0);
    std::cout << "[CACHE-OOM] @" << sc_time_stamp() << " mod=" << name()
              << " Cache OOM: current capacity: " << cur_mb << " MB, write capacity: "
              << write_mb << " MB\n" << std::flush;
}

// ── log_pending_ports ─────────────────────────────────────────────────────────
// Permanent (not env-gated), fixed-cycle-interval snapshot of every port's
// pending write/read data_id -- unlike log_oom (which only fires at the
// moment admission is denied), this prints on a wall-clock cadence
// regardless of whether anything is currently stuck, so a trend (a port's
// pending id changing vs. staying wedged on the same id cycle after cycle)
// is visible without needing DBG_RACE_TRACE.
template <typename T>
void Cache<T>::log_pending_ports(uint64_t cycle_count) const {
    constexpr uint64_t PENDING_LOG_PERIOD = 500000;
    if (cycle_count % PENDING_LOG_PERIOD != 0) return;
    std::cout << "[CACHE-PENDING] @" << sc_time_stamp() << " mod=" << name()
              << " pending_read_ids=[";
    for (int p = 0; p < num_ports_; ++p) {
        if (p > 0) std::cout << " ";
        std::cout << p << ":" << (pending_reqs_valid[p] && pending_reqs[p]
                                   ? std::to_string(pending_reqs[p]->getAddr()) : "-");
    }
    std::cout << "] pending_wreq_ids=[";
    for (int p = 0; p < num_ports_; ++p) {
        if (p > 0) std::cout << " ";
        std::cout << p << ":" << (pending_wreq_valid[p] && pending_wreq[p]
                                   ? std::to_string(pending_wreq[p]->id) : "-");
    }
    std::cout << "] active_wreq_ids=[";
    for (int p = 0; p < num_ports_; ++p) {
        if (p > 0) std::cout << " ";
        std::cout << p << ":" << (active_wreq_valid[p] && active_wreq[p]
                                   ? std::to_string(active_wreq[p]->id) : "-");
    }
    std::cout << "] resident=[";
    bool first_resident = true;
    for (const auto& [id, data] : cache_data) {
        if (!first_resident) std::cout << " ";
        first_resident = false;
        std::cout << id << ":" << data_bits(data);
    }
    std::cout << "]\n" << std::flush;
}

// ── rename_or_erase ───────────────────────────────────────────────────────────
// Rename-on-terminal-read (data-operation-inst-array-vector-tmu plan): when a
// terminal (reserved) read also carries a nonzero new_data_id, re-key the
// SAME physical Data_t from addr to new_id instead of erasing it -- zero
// duplication, current_usage_bits_ unchanged (a re-key, not a resize). The
// new id doesn't exist in cache_data until this fires, so any consumer of it
// naturally blocks via the existing miss/retry path until this read
// completes -- that's the whole mechanism. Both fatal checks below are
// gen-time contract violations (never a runtime race: Cache has no
// eviction, so a collision can only be a static id-planning bug), logged
// permanently like log_oom rather than silently accepted.
template <typename T>
void Cache<T>::rename_or_erase(uint64_t addr, uint64_t new_id, bool is_reserved,
                                const LLM_SIM_DATA::DataPtr_t<T>& data) {
    if (new_id != 0) {
        if (!is_reserved) {
            std::cout << "[CACHE-RENAME-ERROR] @" << sc_time_stamp() << " mod=" << name()
                      << " new_data_id=" << new_id << " requested on addr=" << addr
                      << " without is_reserved -- gen-time contract violation\n" << std::flush;
            std::abort();
        }
        if (cache_data.count(new_id) != 0) {
            std::cout << "[CACHE-RENAME-ERROR] @" << sc_time_stamp() << " mod=" << name()
                      << " new_data_id=" << new_id << " already resident (id collision)"
                      << " -- gen-time id-planning bug\n" << std::flush;
            std::abort();
        }
        cache_data.erase(addr);
        data->id = new_id;
        cache_data[new_id] = data;
    } else if (is_reserved) {
        current_usage_bits_ -= data_bits(data);
        cache_data.erase(addr);
    }
}

// ── latency_cal ───────────────────────────────────────────────────────────────
// Latency = floor(data_bits(...) / CACHE_BUS_WIDTH_BITS), minimum 1 cycle.
// Shares the same total-size quantity data_bits() uses for capacity accounting.
template <typename T>
uint64_t Cache<T>::latency_cal(const LLM_SIM_INST::CacheWriteReqPtr_t& req) const {
    const uint64_t bits   = data_bits(req);
    const uint64_t cycles = bits / LLM_SIM_CACHE::CACHE_BUS_WIDTH_BITS;
    return cycles > 0 ? cycles : 1;
}
template <typename T>
uint64_t Cache<T>::latency_cal(const LLM_SIM_DATA::DataPtr_t<T>& data) const {
    const uint64_t bits   = data_bits(data);
    const uint64_t cycles = bits / LLM_SIM_CACHE::CACHE_BUS_WIDTH_BITS;
    return cycles > 0 ? cycles : 1;
}

// ── run ───────────────────────────────────────────────────────────────────────
// Scheduling policy (single-port: one operation per cycle; dual-port
// (LLM_SIM_CACHE::DUAL_PORT): one write AND one read per cycle):
//   Phase 1 – Write priority: scan ports 0..N-1; serve the FIRST port whose
//              write REQUEST fits under CACHE_MAX_CAPACITY_BITS. Wait for
//              the request's own latency (computed from the request alone,
//              before the payload is read), THEN read data_in[i] and store
//              in cache_data; skip Phase 2 this cycle unless DUAL_PORT is set.
//              Over capacity → keep/store as pending (pending_wreq) and
//              SKIP to next port, same priority-inversion protection as
//              read misses below.
//   Phase 2 – Read: scan ports 0..N-1 in priority order; find the first port
//              whose request HITS in cache_data and serve it.
//              Miss → keep/store as pending and SKIP to next port
//              (priority-inversion protection: a high-priority miss never
//               starves a lower-priority port that needs to write first).
// Round 16/17: precise single-(core,
// id) pin diagnostic for the S512 QK-output-never-delivered deadlock --
// DBG_RACE_TRACE's global-scope tracing changed the deadlock's own timing
// enough that it stopped reproducing faithfully. This is near-zero overhead
// (two getenv+atoll calls, cached in statics, per Cache::run() invocation)
// since it only ever compares against ONE fixed (core substring, id) pair
// instead of touching every id/every port.
static bool dbg_pin_matches(const char* mod_name, uint64_t id) {
    static const char* pin_id_env = std::getenv("DBG_PIN_ID");
    static const char* pin_core_env = std::getenv("DBG_PIN_CORE");
    if (!pin_id_env) return false;
    static const uint64_t pin_id = static_cast<uint64_t>(std::atoll(pin_id_env));
    if (id != pin_id) return false;
    if (!pin_core_env) return true;
    return std::string(mod_name).find(pin_core_env) != std::string::npos;
}

template <typename T>
void Cache<T>::run() {
    for (int i = 0; i < num_ports_; ++i) {
        pending_reqs_valid[i] = false;
        pending_reqs[i] = nullptr;
        pending_wreq_valid[i] = false;
        pending_wreq[i] = nullptr;
        active_wreq_valid[i] = false;
        active_wreq[i] = nullptr;
        wreq_ready_cycle[i] = 0;
    }
    std::vector<uint64_t> miss_streak(num_ports_, 0);
    std::vector<uint64_t> miss_addr(num_ports_, 0);
    // Consecutive cycles a port has had a write candidate but wasn't served
    // (reset to 0 the cycle it's served or has none). Used only to break a
    // genuine indefinite-starvation deadlock -- see WRITE_STARVE_THRESHOLD
    // below; does not otherwise change the fixed-priority scan order, so
    // normal-case timing is bit-for-bit identical to the original behavior.
    // (An earlier decode B=2/4/8/16 "regression" seen while testing this was
    // traced to stale KV-cache fixtures left over from unrelated real-prompt
    // testing, NOT this fix,
    // 2026-07-16.)
    std::vector<uint64_t> write_wait_age(num_ports_, 0);
    // Same anti-starvation mechanism for Phase 2 (read): TMU gathers its OWN
    // instruction's inputs via read requests on its own port (5), so if
    // reads are starved the same way writes were, TMU can never even finish
    // collecting inputs to attempt a write in the first place -- a write-only
    // fix does not cover this (found 2026-07-16: prefill still deadlocked,
    // just later in the program, after a write-only fix).
    std::vector<uint64_t> read_wait_age(num_ports_, 0);
    // Comfortably above any legitimate multi-port contention (a handful of
    // cycles in practice) but far below the hundreds-of-thousands-of-cycles
    // duration the real prefill deadlock exhibited (port 5 / TMU output
    // starved by ports 0-4,
    // 2026-07-15), so it only ever fires to prevent that deadlock.
    constexpr uint64_t WRITE_STARVE_THRESHOLD = 64;
    constexpr uint64_t READ_STARVE_THRESHOLD = 64;

    // DBG_WATCH_ID_TRACE gates CACHE-WRITE/CACHE-HIT/CACHE-MISS below --
    // previously unconditional, which made every run pay this id-range
    // check (and the resulting print volume made deadlock reruns take
    // hours of wall-clock to reach a stall point). Set the env var to
    // re-enable this trace for a specific investigation.
    static const bool watch_id_trace_on = std::getenv("DBG_WATCH_ID_TRACE") != nullptr;
    auto is_watch_id = [](uint64_t id) {
        if (!watch_id_trace_on) return false;
        // Expert kernel WIDs (9002-9013, 19002-19013, ...) + activation slot 21 + DMA prefetch
        if (id == 9010) return true;
        if (id == 21) return true;
        // Result slots and Phase 14 TMU outputs for core 14 leader
        if (id >= 3484 && id <= 3494) return true;
        // Any kernel-remapped WID: original 9002-9013, stride 10000
        uint64_t base = id % 10000;
        if (base >= 9002 && base <= 9013) return true;
        // (2026-08-12): the id==21
        // literal above only ever matched call_n==0 (unremapped) --
        // WID_DISPATCH_ACT (21) remapped by call_n*10000+21 for call_n>0
        // (repeated self-dispatch on the same core) was invisible to this
        // watch list, which is why the TMUOp::cancat/split "value.size()=1"
        // corruption at seq_len 256/512/1024 (chunk large enough to push
        // call_n well above 0) was never caught by CACHE-WRITE/CACHE-HIT
        // trace before now.
        return base == 21;
    };

    // ── PARALLEL_ARBITER-only state ──────────────────────────────────────────
    // Every port gets its own independent write and read pipe (no priority
    // scan, no starvation), so instead of a single blocking wait() modelling
    // bandwidth latency (which would serialize all ports through one thread),
    // each port tracks the cycle it becomes free again after a commit/hit.
    // Write latency/busy tracking lives in active_wreq_valid[i] /
    // wreq_ready_cycle[i] (member state, so it's accessible to
    // validate_write etc.) instead of a local busy_until_write here.
    // Reads additionally need a separate delayed-delivery slot: the hit is
    // resolved immediately (so a later cycle doesn't re-process the same
    // request), but data_out/erase-on-reserved only fire once that port's
    // latency has elapsed, mirroring the wait-then-deliver order the default
    // path uses.
    std::vector<uint64_t> busy_until_read(num_ports_, 0);
    std::vector<bool> delivery_valid(num_ports_, false);
    std::vector<uint64_t> delivery_cycle(num_ports_, 0);
    std::vector<LLM_SIM_DATA::DataPtr_t<T>> delivery_data(num_ports_, nullptr);
    std::vector<bool> delivery_reserved(num_ports_, false);
    // Rename-on-terminal-read target, captured at hit-resolution time
    // (alongside delivery_reserved) since the originating CacheReq_t is
    // gone (pending_reqs[i] cleared) by the time delivery actually flushes.
    std::vector<uint64_t> delivery_new_id(num_ports_, 0);
    uint64_t cycle_count = 0;

    // DBG_CACHE_OCCUPANCY: periodic snapshot of this Cache's resident-entry
    // count and bit usage vs. CACHE_MAX_CAPACITY_BITS, printed every
    // DBG_CACHE_OCCUPANCY cycles (value = period). Answers "is this stuck
    // core's Cache actually full (space problem) or sitting well under cap
    // with a lost/never-arriving request (logic problem)?" without needing
    // to instrument any specific id -- distinguishes the two hypotheses from
    // one flat, always-comparable number.
    const char* occ_env = std::getenv("DBG_CACHE_OCCUPANCY");
    const uint64_t occ_period = occ_env ? static_cast<uint64_t>(std::atoll(occ_env)) : 0;
    uint64_t occ_next_print = occ_period;

    while (true) {
        wait(); // wait for the next clock edge
        ++cycle_count; // now incremented unconditionally (was PARALLEL_ARBITER-only),
                        // so both the occupancy snapshot below and log_pending_ports
                        // fire correctly regardless of arbiter_mode.

        if (occ_period > 0 && cycle_count >= occ_next_print) {
            occ_next_print += occ_period;
            std::cout << "[CACHE-OCCUPANCY] @" << sc_time_stamp()
                      << " mod=" << name()
                      << " n_resident=" << cache_data.size()
                      << " usage_bits=" << current_usage_bits_
                      << " cap_bits=" << LLM_SIM_CACHE::CACHE_MAX_CAPACITY_BITS
                      << " pct=" << (100.0 * current_usage_bits_ / LLM_SIM_CACHE::CACHE_MAX_CAPACITY_BITS)
                      << "\n" << std::flush;
        }

        log_pending_ports(cycle_count);

        if (LLM_SIM_CACHE::PARALLEL_ARBITER) {

            // ── Delivery: flush any read hit whose latency has elapsed ──────
            for (int i = 0; i < num_ports_; ++i) {
                if (!delivery_valid[i] || cycle_count < delivery_cycle[i]) continue;
                auto data = delivery_data[i];
                // Capture the id this delivery was requested/looked-up under
                // BEFORE rename_or_erase can mutate data->id -- logging below
                // (and the requester's own view of "what did I just read")
                // must see the pre-rename id, not the post-rename one.
                const uint64_t orig_id = data->id;
                const uint64_t new_id = delivery_new_id[i];
                data_out[i].write(data);
                rename_or_erase(orig_id, new_id, delivery_reserved[i], data);
                delivery_valid[i] = false;
                delivery_data[i] = nullptr;
                delivery_new_id[i] = 0;
                if (dbg_pin_matches(name(), orig_id)) {
                    std::cout << "[PIN-DELIVER] @" << sc_time_stamp()
                              << " mod=" << name() << " id=" << orig_id
                              << " port=" << i << " erased=" << delivery_reserved[i]
                              << " renamed_to=" << new_id
                              << " cycle_count=" << cycle_count
                              << "\n" << std::flush;
                }
                if (is_watch_id(orig_id)) {
                    float norm = 0.f, v1701 = 0.f;
                    if (!data->value.empty()) {
                        for (auto v : data->value) norm += v * v;
                        norm = std::sqrt(norm);
                        if (data->value.size() > 1701) v1701 = data->value[1701];
                    }
                    std::cout << "[CACHE-HIT] @" << sc_time_stamp()
                              << " mod=" << name()
                              << " id=" << orig_id
                              << " port=" << i
                              << " norm=" << norm
                              << " v[1701]=" << v1701
                              << (delivery_reserved[i] ? " [ERASED]" : "")
                              << "\n";
                    miss_streak[i] = 0;
                    miss_addr[i] = 0;
                }
            }

            // ── Write, step 1: commit any active request whose latency has
            // elapsed and whose payload has arrived on data_in[i] ─────────
            for (int i = 0; i < num_ports_; ++i) {
                if (!active_wreq_valid[i] || cycle_count < wreq_ready_cycle[i]) continue;
                if (data_in[i].num_available() == 0) continue; // latency elapsed, payload not here yet

                auto req = active_wreq[i];
                auto data = data_in[i].read();
                validate_write(i, req, data);

                const uint64_t new_bits = data_bits(req);
                auto existing = cache_data.find(data->id);
                const bool overwrite = (existing != cache_data.end());
                const uint64_t old_bits = overwrite ? data_bits(existing->second) : 0;

                cache_data[data->id] = data;
                current_usage_bits_ = current_usage_bits_ - old_bits + new_bits;
                reserved_bits_ -= new_bits;
                active_wreq_valid[i] = false;
                active_wreq[i] = nullptr;
                if (dbg_pin_matches(name(), data->id)) {
                    std::cout << "[PIN-WRITE] @" << sc_time_stamp()
                              << " mod=" << name() << " id=" << data->id
                              << " port=" << i << " overwrite=" << overwrite
                              << " value.size()=" << data->value.size()
                              << " cycle_count=" << cycle_count
                              << "\n" << std::flush;
                }
                if (is_watch_id(data->id)) {
                    float norm = 0.f, v1701 = 0.f;
                    if (!data->value.empty()) {
                        for (auto v : data->value) norm += v * v;
                        norm = std::sqrt(norm);
                        if (data->value.size() > 1701) v1701 = data->value[1701];
                    }
                    std::cout << "[CACHE-WRITE" << (overwrite ? "-OVERWRITE" : "") << "] @"
                              << sc_time_stamp()
                              << " mod=" << name()
                              << " id=" << data->id
                              << " port=" << i
                              << " norm=" << norm
                              << " v[1701]=" << v1701
                              << " value.size()=" << data->value.size()
                              << " size_=[";
                    for (int s : data->size_) std::cout << s << ",";
                    std::cout << "]\n";
                }
            }

            // ── Write, step 2: admit a new request into the now-empty active
            // slot, reserving its bits and computing latency before the
            // payload is ever read ──────────────────────────────────────────
            for (int i = 0; i < num_ports_; ++i) {
                if (active_wreq_valid[i]) continue; // still waiting on a prior commit

                LLM_SIM_INST::CacheWriteReqPtr_t req = nullptr;
                if (pending_wreq_valid[i]) {
                    req = pending_wreq[i];
                } else if (wreq_in[i].num_available() > 0) {
                    req = wreq_in[i].read();
                }
                if (!req) continue;

                if (dbg_pin_matches(name(), req->id)) {
                    std::cout << "[PIN-WRITE-SEEN] @" << sc_time_stamp()
                              << " mod=" << name() << " id=" << req->id
                              << " port=" << i
                              << " from_pending=" << pending_wreq_valid[i]
                              << "\n" << std::flush;
                }

                const uint64_t new_bits = data_bits(req);
                auto existing = cache_data.find(req->id);
                const uint64_t old_bits = (existing != cache_data.end()) ? data_bits(existing->second) : 0;

                if (current_usage_bits_ + reserved_bits_ - old_bits + new_bits > LLM_SIM_CACHE::CACHE_MAX_CAPACITY_BITS) {
                    // Over capacity: pend and retry next cycle (checked in the
                    // same fixed port 0..N-1 order every cycle -- lower-index
                    // ports still get first claim on freed-up capacity).
                    log_oom(new_bits, old_bits);
                    if (std::getenv("DBG_RACE_TRACE")) {
                        static thread_local std::unordered_map<uint64_t, uint64_t> dbg_pend_streak;
                        if (++dbg_pend_streak[req->id] % 500000 == 1) {
                            std::cout << "[CACHE-CAPACITY-PEND] @" << sc_time_stamp()
                                      << " mod=" << name()
                                      << " id=" << req->id << " port=" << i
                                      << " new_bits=" << new_bits
                                      << " current_usage_bits=" << current_usage_bits_
                                      << " reserved_bits=" << reserved_bits_
                                      << " cap=" << LLM_SIM_CACHE::CACHE_MAX_CAPACITY_BITS
                                      << " n_resident=" << cache_data.size()
                                      << "\n" << std::flush;
                        }
                    }
                    pending_wreq_valid[i] = true;
                    pending_wreq[i] = req;
                    continue;
                }

                reserved_bits_ += new_bits;
                pending_wreq_valid[i] = false;
                pending_wreq[i] = nullptr;
                active_wreq_valid[i] = true;
                active_wreq[i] = req;
                wreq_ready_cycle[i] = cycle_count + latency_cal(req);
            }

            // ── Read: every non-busy port attempts its own hit/miss ─────────
            for (int i = 0; i < num_ports_; ++i) {
                if (cycle_count < busy_until_read[i]) continue; // still delivering a prior hit

                LLM_SIM_INST::CacheReqPtr_t req = nullptr;
                if (pending_reqs_valid[i]) {
                    req = pending_reqs[i];
                } else if (data_req[i].num_available() > 0) {
                    data_req[i].read(req);
                }
                if (!req) continue;

                uint64_t addr = req->getAddr();
                const bool dbg_pin = dbg_pin_matches(name(), addr);
                auto it = cache_data.find(addr);
                if (it == cache_data.end()) {
                    // Miss costs no latency, same as default mode -- retried every cycle.
                    pending_reqs_valid[i] = true;
                    pending_reqs[i] = req;
                    if (dbg_pin) {
                        static thread_local uint64_t dbg_pin_miss_streak = 0;
                        ++dbg_pin_miss_streak;
                        if (dbg_pin_miss_streak == 1 || dbg_pin_miss_streak % 200000 == 0) {
                            std::cout << "[PIN-MISS] @" << sc_time_stamp()
                                      << " mod=" << name() << " id=" << addr
                                      << " port=" << i
                                      << " streak=" << dbg_pin_miss_streak
                                      << " cycle_count=" << cycle_count
                                      << " n_resident=" << cache_data.size()
                                      << "\n" << std::flush;
                        }
                    }
                    if (is_watch_id(addr)) {
                        if (miss_addr[i] == addr) {
                            miss_streak[i]++;
                            if (miss_streak[i] == 1 || miss_streak[i] % 1000000 == 0) {
                                std::cout << "[CACHE-MISS] @" << sc_time_stamp()
                                          << " mod=" << name()
                                          << " id=" << addr << " port=" << i
                                          << " streak=" << miss_streak[i] << "\n";
                            }
                        } else {
                            miss_addr[i] = addr;
                            miss_streak[i] = 1;
                            std::cout << "[CACHE-MISS-NEW] @" << sc_time_stamp()
                                      << " mod=" << name()
                                      << " id=" << addr << " port=" << i << "\n";
                        }
                    }
                    continue;
                }

                // Hit: resolve now (so this request isn't re-processed), but
                // defer delivery/erase-on-reserved to the port's own latency.
                auto data = it->second;
                pending_reqs_valid[i] = false;
                pending_reqs[i] = nullptr;
                const uint64_t lat = latency_cal(data);
                busy_until_read[i] = cycle_count + lat;
                delivery_valid[i] = true;
                delivery_cycle[i] = cycle_count + lat;
                delivery_data[i] = data;
                delivery_reserved[i] = req->getIsReserved();
                delivery_new_id[i] = req->getNewDataId();
                if (dbg_pin) {
                    std::cout << "[PIN-HIT] @" << sc_time_stamp()
                              << " mod=" << name() << " id=" << addr
                              << " port=" << i << " reserved=" << delivery_reserved[i]
                              << " delivery_cycle=" << delivery_cycle[i]
                              << " cycle_count=" << cycle_count
                              << "\n" << std::flush;
                }
            }

            continue;
        }

        // ── Phase 1: Write (highest priority) ────────────────────────────────
        // Candidate resolution mirrors Phase 2's pending_reqs handling: a
        // write blocked last cycle by the capacity check is retried from
        // pending_wreq[i] before any new wreq_in[i] entry is looked at.
        // A candidate that still doesn't fit is kept pending and SKIPPED
        // (not blocking lower-priority ports), same priority-inversion
        // protection as the read-miss path below.
        //
        // Scan order is 0..N-1 (unchanged) UNLESS some port has been starved
        // past WRITE_STARVE_THRESHOLD, in which case that port is moved to
        // the front just for this cycle only -- see write_wait_age comment
        // above for why this is deliberately not a full round-robin.
        int starved_port = -1;
        for (int i = 0; i < num_ports_; ++i) {
            const bool has_candidate = pending_wreq_valid[i] || (wreq_in[i].num_available() > 0);
            if (!has_candidate) { write_wait_age[i] = 0; continue; }
            ++write_wait_age[i];
            if (write_wait_age[i] >= WRITE_STARVE_THRESHOLD &&
                (starved_port == -1 || write_wait_age[i] > write_wait_age[starved_port])) {
                starved_port = i;
            }
        }

        std::vector<int> scan_order;
        scan_order.reserve(num_ports_);
        if (starved_port != -1) scan_order.push_back(starved_port);
        for (int i = 0; i < num_ports_; ++i) if (i != starved_port) scan_order.push_back(i);

        bool did_write = false;
        for (int i : scan_order) {
            LLM_SIM_INST::CacheWriteReqPtr_t req = nullptr;
            if (pending_wreq_valid[i]) {
                req = pending_wreq[i]; // already dequeued, waiting for space
            } else if (wreq_in[i].num_available() > 0) {
                req = wreq_in[i].read(); // dequeue new write request
            }
            if (!req) continue; // no write candidate for this port

            const uint64_t new_bits = data_bits(req);
            auto existing = cache_data.find(req->id);
            const bool overwrite = (existing != cache_data.end());
            const uint64_t old_bits = overwrite ? data_bits(existing->second) : 0;

            if (current_usage_bits_ - old_bits + new_bits > LLM_SIM_CACHE::CACHE_MAX_CAPACITY_BITS) {
                // Over capacity: pend and try a lower-priority port instead.
                log_oom(new_bits, old_bits);
                pending_wreq_valid[i] = true;
                pending_wreq[i] = req;
                continue;
            }

            pending_wreq_valid[i] = false;
            pending_wreq[i] = nullptr;
            write_wait_age[i] = 0;
            // Latency known up front from the request alone -- wait for it
            // BEFORE the payload is ever read off data_in[i] (the whole
            // point of this mechanism). The producer writes wreq_in[i] then
            // data_in[i] in the same cycle, so by the time this wait()
            // elapses the payload is guaranteed to already be sitting in
            // data_in[i].
            // CONSTRAINT: this read blocks the whole Cache thread. If a
            // producer ever sends the request and the data separately (any
            // wait() between the two writes, e.g. sending the request ahead
            // of a computation), every port stalls here until that data
            // arrives. Such a producer needs this changed to a non-blocking
            // retry (as the PARALLEL_ARBITER path already does).
            wait(latency_cal(req) * sc_time(1, SC_NS)); // model write bandwidth
            auto data = data_in[i].read();
            validate_write(i, req, data);

            cache_data[data->id] = data;
            current_usage_bits_ = current_usage_bits_ - old_bits + new_bits;
            if (is_watch_id(data->id)) {
                float norm = 0.f, v1701 = 0.f;
                if (!data->value.empty()) {
                    for (auto v : data->value) norm += v * v;
                    norm = std::sqrt(norm);
                    if (data->value.size() > 1701) v1701 = data->value[1701];
                }
                std::cout << "[CACHE-WRITE" << (overwrite ? "-OVERWRITE" : "") << "] @"
                          << sc_time_stamp()
                          << " mod=" << name()
                          << " id=" << data->id
                          << " port=" << i
                          << " norm=" << norm
                          << " v[1701]=" << v1701
                          << " value.size()=" << data->value.size()
                          << " size_=[";
                for (int s : data->size_) std::cout << s << ",";
                std::cout << "]\n";
            }
            did_write = true;
            break; // one write commit per cycle
        }
        // Single-port: a write this cycle excludes a read this cycle.
        // Dual-port: fall through into Phase 2 regardless of did_write, so
        // one read and one write can both complete in the same cycle.
        if (did_write && !LLM_SIM_CACHE::DUAL_PORT) continue;

        // ── Phase 2: Read (write-exhausted; first hit wins) ──────────────────
        // Same starved-port-to-front override as Phase 1 -- see read_wait_age
        // comment above.
        int starved_read_port = -1;
        for (int i = 0; i < num_ports_; ++i) {
            const bool has_candidate = pending_reqs_valid[i] || (data_req[i].num_available() > 0);
            if (!has_candidate) { read_wait_age[i] = 0; continue; }
            ++read_wait_age[i];
            if (read_wait_age[i] >= READ_STARVE_THRESHOLD &&
                (starved_read_port == -1 || read_wait_age[i] > read_wait_age[starved_read_port])) {
                starved_read_port = i;
            }
        }

        std::vector<int> read_scan_order;
        read_scan_order.reserve(num_ports_);
        if (starved_read_port != -1) read_scan_order.push_back(starved_read_port);
        for (int i = 0; i < num_ports_; ++i) if (i != starved_read_port) read_scan_order.push_back(i);

        for (int i : read_scan_order) {
            // Determine the active request for this port.
            LLM_SIM_INST::CacheReqPtr_t req = nullptr;
            if (pending_reqs_valid[i]) {
                req = pending_reqs[i]; // already waiting
            } else if (data_req[i].num_available() > 0) {
                data_req[i].read(req); // dequeue new request
            }
            if (!req) continue; // no request for this port

            uint64_t addr = req->getAddr();
            auto it = cache_data.find(addr);
            if (it == cache_data.end()) {
                // Miss: store as pending and skip — do NOT block other ports.
                pending_reqs_valid[i] = true;
                pending_reqs[i] = req;
                if (is_watch_id(addr)) {
                    if (miss_addr[i] == addr) {
                        miss_streak[i]++;
                        if (miss_streak[i] == 1 || miss_streak[i] % 1000000 == 0) {
                            std::cout << "[CACHE-MISS] @" << sc_time_stamp()
                                      << " mod=" << name()
                                      << " id=" << addr << " port=" << i
                                      << " streak=" << miss_streak[i] << "\n";
                        }
                    } else {
                        miss_addr[i] = addr;
                        miss_streak[i] = 1;
                        std::cout << "[CACHE-MISS-NEW] @" << sc_time_stamp()
                                  << " mod=" << name()
                                  << " id=" << addr << " port=" << i << "\n";
                    }
                } else if (std::getenv("DBG_RACE_TRACE")) {
                    // (2026-08-13,
                    // seq_len=1024 full-attention chain-forward deadlock):
                    // is_watch_id() only covers expert-kernel scratch ids,
                    // never Phase 4 K/V-exchange data_ids (e.g. the k_cache_ws
                    // prefetch reads DMA.cpp's CACHE_READ_WAIT state stalls
                    // on) -- a permanent miss on one of THOSE ids was
                    // invisible to CACHE-MISS/CACHE-MISS-NEW before now. Log
                    // any non-watched id whose miss streak crosses 500k
                    // (a value that would never legitimately keep missing --
                    // either the write that should have populated it never
                    // happened, or happened on the wrong core/data_id).
                    static thread_local std::unordered_map<uint64_t, uint64_t> dbg_miss_streak;
                    if (++dbg_miss_streak[addr] == 500000) {
                        std::cout << "[CACHE-MISS-STUCK] @" << sc_time_stamp()
                                  << " mod=" << name()
                                  << " id=" << addr << " port=" << i
                                  << " streak=500000\n" << std::flush;
                    }
                }
                continue; // try lower-priority port
            }

            // Hit: model read bandwidth, send data, clear pending.
            auto data = it->second;
            wait(latency_cal(data) * sc_time(1, SC_NS));
            data_out[i].write(data);
            rename_or_erase(addr, req->getNewDataId(), req->getIsReserved(), data);
            pending_reqs_valid[i] = false;
            pending_reqs[i] = nullptr;
            read_wait_age[i] = 0;
            if (is_watch_id(addr)) {
                float norm = 0.f, v1701 = 0.f;
                if (!data->value.empty()) {
                    for (auto v : data->value) norm += v * v;
                    norm = std::sqrt(norm);
                    if (data->value.size() > 1701) v1701 = data->value[1701];
                }
                std::cout << "[CACHE-HIT] @" << sc_time_stamp()
                          << " mod=" << name()
                          << " id=" << addr
                          << " port=" << i
                          << " norm=" << norm
                          << " v[1701]=" << v1701
                          << (req->getIsReserved() ? " [ERASED]" : "")
                          << "\n";
                miss_streak[i] = 0;
                miss_addr[i] = 0;
            }
            break; // one operation per cycle
        }
    }
}

template class Cache<int>;
template class Cache<float>;