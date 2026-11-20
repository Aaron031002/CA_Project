#ifndef LLM_SIM_ARRAY_PARAMS_H
#define LLM_SIM_ARRAY_PARAMS_H

#include <cstdint>
#include <array>

namespace LLM_SIM_ARRAY {
    constexpr int TILE_SIZE = 8;
    constexpr int DATA_SIZE = 16;
    constexpr int WEIGHT_BUF_OVERHEAD = 5;
    constexpr int WEIGHT_BUF_BANDWIDTH = 1024 / DATA_SIZE;
    constexpr int HEAD_DIM = 16; // per-head dimension used by QK and SV operations

    // MAC throughput: multiply-accumulate operations per cycle.
    // For NORMAL [M×K] × [K×N]: MACs = M * K * N.
    // Compute latency = ceil(MACs / ARRAY_MACS).
    // Final Array latency = max(DRAM_weight_lat, compute_lat) + WEIGHT_BUF_OVERHEAD.
    // Set to match hardware design; 64 = baseline (memory-bound for decode M=1,
    // compute-bound for prefill M=16).
    constexpr uint64_t ARRAY_MACS = 256;

    // Systolic array geometry (§A). ARRAY_ROWS *
    // ARRAY_COLS == ARRAY_MACS so the tile-based compute_lat formulas
    // (ceil(K/ROWS)*ceil(N/COLS)*M for weight-stationary, ceil(M/ROWS)*
    // ceil(N/COLS)*K for output-stationary) reduce to the old M*K*N/ARRAY_MACS
    // whenever dimensions are exact multiples of the tile size, and only
    // diverge (correctly) when a tile is not fully packed.
    constexpr int ARRAY_ROWS = 16;
    constexpr int ARRAY_COLS = 16;
    static_assert(ARRAY_ROWS * ARRAY_COLS == static_cast<int>(ARRAY_MACS),
                  "ARRAY_ROWS * ARRAY_COLS must equal ARRAY_MACS");
    // Pipeline fill/drain depth for the systolic array.
    constexpr int ARRAY_FILL_DRAIN = ARRAY_ROWS + ARRAY_COLS;

    // ── Mixed-precision MAC throughput ──
    // The 16x16 16-bit-integer MAC array can be symmetrically split into
    // narrower sub-multipliers: a P-bit mode packs (16/P)^2 P-bit MACs into
    // the area/time of one 16-bit MAC (square-law throughput scaling).
    // Supported modes are the native width (16) plus the narrower split
    // widths the hardware actually implements (4, 8), extended with 32 so
    // the same formula also covers an operand *wider* than the native MAC
    // (a 32-bit MAC consumes (32/16)^2 = 4 native MACs' worth of area/time,
    // i.e. it is proportionally slower, not merely "unaccelerated") --
    // this is the common case today since Data_t<T>::elem_bits defaults to
    // sizeof(T)*8 (T=float -> 32) for any value no quantization stage has
    // tagged with a narrower width.
    constexpr std::array<int, 4> MAC_P_MODES = {4, 8, 16, 32};

    // P_mode = smallest supported mode >= the operand's effective bit
    // width (§2's "向上取" rule).
    // eff_bits is the *nominal* elem_bits tag already carried on the
    // operand's Data_t (chunk->elem_bits for the weight, in_ptr->elem_bits
    // for the activation) -- NOT the true integerized effective width some
    // formats need (e.g. MXFP4's E2M1 payload needs signed 5 bits, so by
    // the advice's own §3 it should resolve to P_mode=8, not 4; this
    // known gap is intentionally not closed here -- elem_bits==4 for
    // MXFP4 currently resolves to P_mode=4). Anything above the largest
    // supported mode clamps to that largest mode (currently a no-op since
    // 32 is itself the largest supported mode).
    constexpr int resolve_p_mode(int eff_bits) {
        for (int mode : MAC_P_MODES) {
            if (eff_bits <= mode) return mode;
        }
        return MAC_P_MODES.back();
    }

    // Square-law MAC throughput multiplier for a given P_mode, relative to
    // the native 16-bit MAC: (16/P_mode)^2. P_mode=16 -> 1x (unchanged
    // baseline behavior); P_mode=4 -> 16x; P_mode=32 -> 0.25x (slower, see
    // resolve_p_mode's comment on wider-than-native operands).
    constexpr double mac_speedup(int p_mode) {
        return (16.0 / static_cast<double>(p_mode)) * (16.0 / static_cast<double>(p_mode));
    }
}

#endif // LLM_SIM_ARRAY_PARAMS_H