#ifndef LLM_SIM_OFFCHIP_DRAM_PARAMS_H
#define LLM_SIM_OFFCHIP_DRAM_PARAMS_H

// Phase 6b: off-chip DRAM's own hardware parameters, kept in
// a file physically separate from new_code/DRAM_sim/dram_params.h (local
// DRAM's constants) per the explicit design decision to keep the two config
// surfaces distinct -- they model different physical hardware and are
// expected to diverge (off-chip is farther away and slower).

namespace LLM_SIM_DRAM {
    // Fixed per-transfer overhead cycles for an off-chip DRAM access,
    // analogous to DRAMSim::get_dram_latency()'s hardcoded local-DRAM value
    // (5) but larger, reflecting that off-chip memory is physically slower
    // to service a request once it arrives. Placeholder value -- adjust
    // once real off-chip latency numbers are available.
    constexpr int OFFCHIP_DRAM_OVERHEAD = 15;

    // Elements transferred per clock cycle for an off-chip DRAM access.
    // Narrower than local DRAM's DRAM_BUS_BANDWIDTH (64) -- placeholder,
    // adjust once real off-chip bandwidth numbers are available.
    constexpr int OFFCHIP_DRAM_BANDWIDTH = 32;
}

#endif // LLM_SIM_OFFCHIP_DRAM_PARAMS_H
