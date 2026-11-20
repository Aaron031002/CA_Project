#ifndef LLM_SIM_CACHE_H
#define LLM_SIM_CACHE_H

#include <algorithm>
#include <cstdlib>
#include <unordered_map>
#include <string>
#include <systemc.h>

#include "utils/Data.h"
#include "utils/BaseInst.h"
#include "Core/Cache/cache_params.h"
/*
    This code is used for simulate a shared memory in a single core
*/


template <typename T>
class Cache: public sc_module {
    public:
        sc_in_clk clk;
        sc_vector<sc_fifo_out<LLM_SIM_DATA::DataPtr_t<T>>> data_out;
        sc_vector<sc_fifo_in<LLM_SIM_INST::CacheReqPtr_t>> data_req;
        sc_vector<sc_fifo_in<LLM_SIM_DATA::DataPtr_t<T>>> data_in;
        // Write-request metadata FIFO, one per write port, consumed strictly
        // ahead of the matching data_in[i] entry so Cache can compute
        // latency/capacity before it ever reads the payload. See the
        // write-request-before-data plan for the full design.
        sc_vector<sc_fifo_in<LLM_SIM_INST::CacheWriteReqPtr_t>> wreq_in;
        Cache(sc_module_name name, const int num_ports);

        // Synchronous read used by MainCtrl::handleRPCDispatch to access
        // topk_idx / routing_table without going through the DMA pipeline.
        LLM_SIM_DATA::DataPtr_t<T> lookupData(uint64_t data_id) const {
            auto it = cache_data.find(data_id);
            return (it != cache_data.end()) ? it->second : nullptr;
        }

        // Synchronous write used by MainCtrl for self-dispatch aliasing:
        // copies activation data into a fixed slot (WID_DISPATCH_ACT) so the
        // registered expert kernel (which reads from WID_DISPATCH_ACT=21)
        // can find it without a NoC round-trip.
        void insertData(uint64_t data_id, LLM_SIM_DATA::DataPtr_t<T> data) {
            auto copy = std::make_shared<LLM_SIM_DATA::Data_t<T>>(*data);
            copy->id = data_id;
            auto it = cache_data.find(data_id);
            if (it != cache_data.end()) current_usage_bits_ -= data_bits(it->second);
            current_usage_bits_ += data_bits(copy);
            cache_data[data_id] = copy;
        }

        // Remove an entry so the next read generates a cache miss rather than
        // returning stale data from a prior kernel run.
        void eraseData(uint64_t data_id) {
            auto it = cache_data.find(data_id);
            if (it != cache_data.end()) {
                current_usage_bits_ -= data_bits(it->second);
                cache_data.erase(it);
            }
        }
    private:
        const int num_ports_;
        std::unordered_map<uint64_t, LLM_SIM_DATA::DataPtr_t<T>> cache_data;
        std::vector<LLM_SIM_INST::CacheReqPtr_t> pending_reqs;
        std::vector<bool> pending_reqs_valid;
        // Write-request state machine, per port:
        //   pending_wreq[i]   -- dequeued from wreq_in[i] but denied admission
        //                        by the capacity check; retried each cycle.
        //   active_wreq[i]    -- admitted (space reserved in reserved_bits_),
        //                        waiting for wreq_ready_cycle[i] to elapse and
        //                        for data_in[i] to actually have the payload.
        std::vector<LLM_SIM_INST::CacheWriteReqPtr_t> pending_wreq;
        std::vector<bool> pending_wreq_valid;
        std::vector<LLM_SIM_INST::CacheWriteReqPtr_t> active_wreq;
        std::vector<bool> active_wreq_valid;
        std::vector<uint64_t> wreq_ready_cycle;
        // Bits reserved by admitted-but-not-yet-committed active_wreq
        // entries, so a second port's capacity check sees space already
        // claimed by the first port's in-flight write.
        uint64_t reserved_bits_ = 0;
        // Running total of bits resident in cache_data, checked against
        // LLM_SIM_CACHE::CACHE_MAX_CAPACITY_BITS on every write commit.
        uint64_t current_usage_bits_ = 0;
        // Total bits described by a write request / resident Data_t:
        // product_of_size * elem_bits. Shared by latency_cal (bandwidth
        // modelling) and capacity accounting. Aborts if size is empty or
        // any dimension is non-positive -- a producer must always tag a
        // real shape (see utils/Data.h's make_cache_write_req()).
        uint64_t data_bits(const LLM_SIM_INST::CacheWriteReqPtr_t& req) const;
        uint64_t data_bits(const LLM_SIM_DATA::DataPtr_t<T>& data) const;
        // latency: floor(data_bits(...) / CACHE_BUS_WIDTH_BITS), minimum 1.
        // Request overload drives the write path (latency known before the
        // payload is read); DataPtr overload still drives the read path
        // (a read has no separate request-ahead-of-data mechanism).
        uint64_t latency_cal(const LLM_SIM_INST::CacheWriteReqPtr_t& req) const;
        uint64_t latency_cal(const LLM_SIM_DATA::DataPtr_t<T>& data) const;
        // Commit-time check that the data actually read off data_in[i]
        // matches the write request that was admitted for it (same id,
        // same shape, same elem_bits, shape's product == value.size()).
        // Aborts on mismatch -- a gen-time contract violation (request and
        // data must come from the same producer call, in order), never a
        // runtime race.
        void validate_write(int port, const LLM_SIM_INST::CacheWriteReqPtr_t& req,
                             const LLM_SIM_DATA::DataPtr_t<T>& data) const;
        // Permanent (non-env-gated), rate-limited "Cache OOM" notice, called
        // at every capacity-check-fail site (both the default and
        // PARALLEL_ARBITER write paths). See Cache.cpp for details.
        void log_oom(uint64_t new_bits, uint64_t old_bits) const;
        // Permanent (non-env-gated), fixed-cycle-interval snapshot of every
        // port's pending write/read data_id. See Cache.cpp for details.
        void log_pending_ports(uint64_t cycle_count) const;
        // Rename-on-terminal-read (new_data_id != 0): re-key the same
        // physical `data` entry from `addr` to `new_id` instead of erasing
        // it, when `is_reserved` is set (today's plain erase-on-reserved
        // behavior when new_id == 0). Fatal (permanent stdout notice, like
        // log_oom) if new_id is set without is_reserved, or if new_id is
        // already resident -- both are gen-time contract violations, never
        // a runtime race (Cache has no eviction). See Cache.cpp.
        void rename_or_erase(uint64_t addr, uint64_t new_id, bool is_reserved,
                              const LLM_SIM_DATA::DataPtr_t<T>& data);
        // thread function
        void run();
};

#endif