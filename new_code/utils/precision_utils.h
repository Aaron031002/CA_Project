#ifndef LLM_SIM_PRECISION_UTILS_H
#define LLM_SIM_PRECISION_UTILS_H

#include <stdint.h>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>

// Shared transfer-latency helper for DMA / NoC / Cache.
// Converts "elements * elem_bits" total information into a cycle count
// given a bandwidth constant expressed in "elements per cycle assuming
// 32-bit (float) elements" -- i.e. the EXISTING bandwidth constants
// (RBW/WBW/LANES/CACHE_BANDWIDTH) keep their current numeric values and
// physical meaning unchanged. When elem_bits == 32 this reduces exactly to
// the original ceil(elements / bandwidth) formula (verified: bits_per_cycle
// = bandwidth*32, total_bits = elements*32, ratio = elements/bandwidth),
// so all existing elem_bits==32 (default) call sites are unaffected.
namespace LLM_SIM_PRECISION {

// std::max/std::min give no guarantee when either argument is NaN (the
// typical libstdc++ implementation returns whichever argument is literally
// first whenever a NaN is involved, regardless of which operand carries it) --
// so plain std::max/std::min silently swallow an upstream NaN into a finite
// number depending on argument order, which can mask a real bug (e.g. an
// uninitialized read or an overflow earlier in the pipeline) behind an
// innocuous-looking finite value. These wrappers always propagate NaN
// instead, including through a running-max reduction loop (once the
// accumulator becomes NaN, it stays NaN for the rest of the loop).
inline float nan_safe_max(float a, float b) {
    return (std::isnan(a) || std::isnan(b)) ? std::numeric_limits<float>::quiet_NaN() : std::max(a, b);
}
inline float nan_safe_min(float a, float b) {
    return (std::isnan(a) || std::isnan(b)) ? std::numeric_limits<float>::quiet_NaN() : std::min(a, b);
}

inline uint64_t transfer_cycles(uint64_t elements, int elem_bits,
                                 uint64_t ref_bandwidth_elems_per_cycle) {
    if (elements == 0) return 0;
    const uint64_t total_bits     = elements * static_cast<uint64_t>(elem_bits);
    const uint64_t bits_per_cycle = ref_bandwidth_elems_per_cycle * 32ULL;
    return (total_bits + bits_per_cycle - 1) / bits_per_cycle;
}

// ── F16 round-trip (§3.4) ──────────────────────────
// Bit-level IEEE 754 binary16 round-trip, no compiler-specific _Float16/__fp16
// dependency (project targets C++17, no std::float16_t). Ported VERBATIM from
// third_party/llama.cpp/ggml/src/ggml-impl.h's
// ggml_compute_fp32_to_fp16/ggml_compute_fp16_to_fp32 (the actual reference
// implementation gpt-oss-20b/ggml uses) -- NOT re-derived from the
// §7.1 prose description, which (when re-derived
// independently) turned out to omit the scale_to_inf/scale_to_zero step and
// get the mantissa mask wrong. This port was verified against the doc's own
// worked example (x=1/3 -> f16 0x3555 -> 0.333251953125) after the fix.
inline uint32_t f32_to_bits(float f) {
    uint32_t w;
    std::memcpy(&w, &f, sizeof(w));
    return w;
}
inline float fp32_from_bits(uint32_t w) {
    float f;
    std::memcpy(&f, &w, sizeof(f));
    return f;
}

inline uint16_t f32_to_f16_bits(float f) {
    const float scale_to_inf  = fp32_from_bits(0x77800000u);
    const float scale_to_zero = fp32_from_bits(0x08800000u);
    float base = (std::fabs(f) * scale_to_inf) * scale_to_zero;

    const uint32_t w = f32_to_bits(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & 0x80000000u;
    uint32_t bias = shl1_w & 0xFF000000u;
    if (bias < 0x71000000u) bias = 0x71000000u;

    base = fp32_from_bits((bias >> 1) + 0x07800000u) + base;
    const uint32_t bits = f32_to_bits(base);
    const uint32_t exp_bits = (bits >> 13) & 0x00007C00u;
    const uint32_t mantissa_bits = bits & 0x00000FFFu;
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return static_cast<uint16_t>(
        (sign >> 16) | (shl1_w > 0xFF000000u ? 0x7E00u : nonsign));
}

inline float f16_bits_to_f32(uint16_t h) {
    const uint32_t w = static_cast<uint32_t>(h) << 16;
    const uint32_t sign = w & 0x80000000u;
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = 0xE0u << 23;
    const float exp_scale = fp32_from_bits(0x7800000u);
    const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    const uint32_t magic_mask = 126u << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    const uint32_t denormalized_cutoff = 1u << 27;
    const uint32_t result = sign |
        (two_w < denormalized_cutoff ? f32_to_bits(denormalized_value) : f32_to_bits(normalized_value));
    return fp32_from_bits(result);
}

inline float f32_to_f16_to_f32(float f) {
    return f16_bits_to_f32(f32_to_f16_bits(f));
}

// ── Q8_0 quantize/dequantize round-trip (§3) ───────
// Per-32-element-block symmetric int8 quantization, scale stored as F16.
// Ported from the verified Python reference
// tools/run_gpt_oss_transformer_block.py::quantize_dequantize_q8_0()
// (itself verified against ggml's quantize_row_q8_0_ref): d = amax/127,
// qs = round(x/d), d itself rounded to F16 before being used to dequantize.
inline void quantize_dequantize_q8_0_inplace(std::vector<float>& x) {
    constexpr int BLOCK = 32;
    const int n = static_cast<int>(x.size());
    for (int b = 0; b + BLOCK <= n; b += BLOCK) {
        float amax = 0.0f;
        for (int i = 0; i < BLOCK; ++i) amax = nan_safe_max(amax, std::fabs(x[b + i]));
        float d = f32_to_f16_to_f32(amax / 127.0f);
        float inv_d = (d > 0.0f) ? (1.0f / d) : 0.0f;
        for (int i = 0; i < BLOCK; ++i) {
            float scaled = x[b + i] * inv_d;
            scaled = nan_safe_max(-127.0f, nan_safe_min(127.0f, scaled));
            int8_t q = static_cast<int8_t>(std::lround(scaled));
            x[b + i] = static_cast<float>(q) * d;
        }
    }
}

} // namespace LLM_SIM_PRECISION

#endif
