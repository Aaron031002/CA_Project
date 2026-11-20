#ifndef LLM_SIM_CACHE_PARAMS_H
#define LLM_SIM_CACHE_PARAMS_H

// AUTO-GENERATED from config/hardware.yaml by tools/gen_config_headers.py.
// Do not edit by hand -- edit the YAML and re-run the generator instead.

#include <cstdint>

namespace LLM_SIM_CACHE {
    // true = 1 read + 1 write may complete in the same trigger cycle;
    // false = mutually exclusive (a write excludes a read that cycle).
    // Moot when PARALLEL_ARBITER is true (every port already gets both).
    static constexpr bool DUAL_PORT = true;

    // Hard ceiling on total bits resident in the cache (sum over cache_data
    // of elem_count * Data_t<T>::elem_bits). Writes that would exceed this
    // are pended and retried, mirroring the existing read-miss pending
    // mechanism -- no eviction. Defaults to 2 MiB when unset in YAML.
    static constexpr uint64_t CACHE_MAX_CAPACITY_BYTES = 2097152ULL;
    static constexpr uint64_t CACHE_MAX_CAPACITY_BITS = CACHE_MAX_CAPACITY_BYTES * 8ULL;

    // Cache bus width in bits/cycle. latency_cal() = floor(data_bits(data) /
    // CACHE_BUS_WIDTH_BITS), minimum 1 cycle.
    static constexpr uint64_t CACHE_BUS_WIDTH_BITS = 256ULL;

    // false ("default") = fixed-priority scan over ports 0..N-1 with
    // age-based anti-starvation (at most one write + one read, two if
    // DUAL_PORT, committed per cycle).
    // true ("parallel") = every port completes its own read and write every
    // cycle it has a ready (non-busy) candidate; no starvation bookkeeping
    // needed since no port is ever skipped for another's benefit.
    static constexpr bool PARALLEL_ARBITER = true;
} // namespace LLM_SIM_CACHE

#endif // LLM_SIM_CACHE_PARAMS_H
