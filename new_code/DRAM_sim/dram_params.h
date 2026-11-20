#ifndef LLM_SIM_DRAM_PARAMS_H
#define LLM_SIM_DRAM_PARAMS_H

// AUTO-GENERATED from config/{hardware,model,workload}.yaml by
// tools/gen_config_headers.py. Do not edit by hand -- edit the YAML and
// re-run the generator instead.

#include <cstdint>
#include <iostream>
#include <string>

// DRAM_DATA_ROOT is injected at compile time (see DRAM_sim/CMakeLists.txt).
// It must point to the directory that contains Kcache/ and Vcache/.
#ifndef DRAM_DATA_ROOT
#define DRAM_DATA_ROOT "."
#endif

namespace LLM_SIM_DRAM {
    // ports declaration
    const int NUM_SRC = 2;

    // parameters for KV cache initialization
    const int BATCH       = 64;   // batch size (number of concurrent requests)
    const int MAX_SEQ_LEN = 1024; // maximum sequence length per request
    const int HEAD_DIM    = 64;   // head dimension (elements per seq-position per head)
    const int HEAD_NUM    = 8;    // number of attention heads
    const int NUM_LAYERS  = 1;   // number of transformer layers

    // KV cache file directories (absolute paths injected via DRAM_DATA_ROOT)
    const std::string K_file_path = std::string(DRAM_DATA_ROOT) + "/Kcache";
    const std::string V_file_path = std::string(DRAM_DATA_ROOT) + "/Vcache";

    // KV cache binary layout (per file K_n.bin / V_n.bin):
    //   total rows    = BATCH * HEAD_NUM
    //   row stride    = MAX_SEQ_LEN * HEAD_DIM   (elements per row)
    //   row index     = batch_id * HEAD_NUM + head_id
    //   col range for seq position s: [s * HEAD_DIM, (s+1) * HEAD_DIM)

    // DRAM bandwidth (used for latency estimation in DRAMSim)
    const int DRAM_BUS_WIDTH      = 64; // bus width in bits
    const int DRAM_BUS_BANDWIDTH  = 64; // elements transferred per clock cycle

    // Phase 3c: cross-core DRAM "rail" bandwidth. Applied by
    // DRAMRouter on the return path, AFTER the owner's own DRAMSim latency
    // (which is unaffected -- a remote read occupies the owner's real DRAM
    // bandwidth exactly like a local read). Modeled as one independent rail
    // resource per unordered mesh edge (min(core_a,core_b), max(core_a,core_b)),
    // so unrelated edges never contend with each other.
    const int DRAM_RAIL_OVERHEAD    = 5;  // fixed per-transfer overhead cycles (mirrors get_dram_latency())
    const int DRAM_RAIL_BANDWIDTH   = 64; // elements transferred per clock cycle over one rail

    // Phase D: per-core local DRAM capacity
    // limit, in bytes. NOT applied automatically anywhere -- testbenches
    // must opt in explicitly by passing this to Acc8x8's
    // local_dram_capacity_bytes constructor arg (or DataManagerConfig::
    // local_dram_capacity_bytes directly). Single source of truth so every
    // opted-in testbench and doc reference the same number.
    const uint64_t LOCAL_DRAM_CAPACITY_BYTES = 536870912ULL;

    // DRAM timing parameters (reserved for future accurate latency model)
    /*
    const int nCCDL = 4;
    const int nCCDS = 4;
    const int nRFC  = 110;
    const int nRCD  = 8;
    const int nCL   = 7;
    const int nRP   = 4;
    const int nRC   = 14;
    const int nRAS  = 11;
    */
}

#endif