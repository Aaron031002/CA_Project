#include "Core/VectorUnit/VectorOp.h"
#include "utils/precision_utils.h"
#include <algorithm>
#include <numeric>
#include <tuple>
#include <limits>
#include <iostream>
#include <cstdlib>
#include <cmath>

using namespace LLM_SIM_VEC;

template <typename T>
VectorOp<T>::VectorOp() {}

// ─── dispatch ────────────────────────────────────────────────────────────────

template <typename T>
std::pair<std::vector<LLM_SIM_DATA::DataPtr_t<T>>, uint64_t> VectorOp<T>::execute(
    const LLM_SIM_INST::VectorInst_t& inst,
    const LLM_SIM_DATA::DataPtr_t<T>& input,
    const LLM_SIM_DATA::DataPtr_t<T>& weight)
{
    const auto& in_vec = input->value;

    // ── Special case: ROPE — no weight needed, cos/sin computed on-the-fly ──
    using VM_early = LLM_SIM_INST::VectorInst_t::VectorMode_t;
    if (inst.mode == VM_early::ROPE) {
        auto [result, latency] = execute_rope(in_vec, inst.dim, inst.rope_token_pos, inst.rope_theta);
        auto output = std::make_shared<LLM_SIM_DATA::Data_t<T>>();
        output->id    = inst.out_id;
        output->value = std::move(result);
        if (!input->size_.empty()) {
            int prod = 1;
            for (int d : input->size_) prod *= d;
            output->size_ = (prod == inst.dim) ? input->size_ : std::vector<int>{inst.dim};
        } else {
            output->size_ = {inst.dim};
        }
        return {{output}, latency};
    }

    // ── Special case: TOP_K returns two DataPtr_t (indices + values) ─────────
    if (inst.mode == VM_early::TOP_K) {
        // w_id encodes n_experts (row size); 0 = legacy global top_k
        const int n_experts = (inst.w_id > 0) ? static_cast<int>(inst.w_id) : 0;
        auto [idx_vec, val_vec, latency] = execute_top_k_dual(in_vec, inst.dim, n_experts);

        auto idx_out = std::make_shared<LLM_SIM_DATA::Data_t<T>>();
        idx_out->id    = inst.out_id;
        idx_out->value = std::move(idx_vec);
        idx_out->size_ = {(int)idx_out->value.size()};

        auto val_out = std::make_shared<LLM_SIM_DATA::Data_t<T>>();
        val_out->id    = inst.out_id2;
        val_out->value = std::move(val_vec);
        val_out->size_ = {(int)val_out->value.size()};

        return {{idx_out, val_out}, latency};
    }

    std::vector<T> result_vec;
    uint64_t latency = 0;

    // Most remaining modes need the weight vector, but VectorUnit.cpp's
    // needs_weight check skips fetching one for no-weight modes (RECIP,
    // INV_SQRT, RELU, SIG, SILU, ADD_REDUCE, TOP_K, DIV_CONSTANT,
    // QUANTIZE_DEQUANT_Q8_0, IDLE) -- weight is nullptr there, so guard the
    // dereference instead of unconditionally reading weight->value.
    static const std::vector<T> kEmptyWeight;
    const auto& w_vec = weight ? weight->value : kEmptyWeight;

    using VM = LLM_SIM_INST::VectorInst_t::VectorMode_t;
    switch (inst.mode) {
        // ── Element-wise arithmetic ──────────────────────────────────────────
        case VM::ADD:
            std::tie(result_vec, latency) = execute_add(in_vec, w_vec, inst.dim);
            break;
        case VM::SUB:
            std::tie(result_vec, latency) = execute_sub(in_vec, w_vec, inst.dim);
            break;
        case VM::MUL:
            std::tie(result_vec, latency) = execute_mul(in_vec, w_vec, inst.dim);
            break;
        case VM::RECIP:
            std::tie(result_vec, latency) = execute_recip(in_vec, inst.dim);
            break;
        case VM::INV_SQRT:
            std::tie(result_vec, latency) = execute_inv_sqrt(in_vec, inst.dim);
            break;
        // ── Activation functions ─────────────────────────────────────────────
        case VM::RELU:
            std::tie(result_vec, latency) = execute_relu(in_vec, inst.dim);
            break;
        case VM::SIG:
            std::tie(result_vec, latency) = execute_sig(in_vec, inst.dim);
            break;
        case VM::SILU:
            std::tie(result_vec, latency) = execute_silu(in_vec, inst.dim);
            break;
        case VM::SWIGLU:
            std::tie(result_vec, latency) = execute_swiglu(in_vec, w_vec, inst.dim);
            break;
        // ── Normalisation / transform ────────────────────────────────────────
        case VM::RMS:
            std::tie(result_vec, latency) = execute_rms(in_vec, w_vec, inst.dim);
            break;
        case VM::SOFTMAX:
            // w_id > 0 (or w_reserved) signals an attn-sink tensor of size
            // num_batches (one scalar per softmax row, e.g. per Q-head).
            // w_id == 0 with no reservation: legacy/router softmax, no sink.
            if (inst.w_id > 0 || inst.w_reserved)
                std::tie(result_vec, latency) = execute_softmax(in_vec, inst.dim, w_vec);
            else
                std::tie(result_vec, latency) = execute_softmax(in_vec, inst.dim);
            break;
        // ROPE is handled above (early return) — no weight needed.
        // ── Reduction ────────────────────────────────────────────────────────
        case VM::ADD_REDUCE:
            std::tie(result_vec, latency) = execute_add_reduce(in_vec, input->size_, inst.all, inst.dim);
            break;
        case VM::DOT_PRODUCT:
            std::tie(result_vec, latency) = execute_dot_product(in_vec, w_vec, input->size_, inst.all, inst.dim);
            break;
        case VM::TOP_K:
            std::tie(result_vec, latency) = execute_top_k(in_vec, inst.dim);
            break;
        case VM::DIV_CONSTANT:
            std::tie(result_vec, latency) = execute_div_constant(in_vec, inst.dim);
            break;
        case VM::BIAS_ADD:
            std::tie(result_vec, latency) = execute_bias_add(in_vec, w_vec, inst.dim);
            break;
        case VM::SPARSE_GATE: {
            const int n_experts = inst.dim;
            const int top_k     = static_cast<int>(inst.out_id2);
            std::tie(result_vec, latency) = execute_sparse_gate(in_vec, w_vec, n_experts, top_k);
            break;
        }
        case VM::QUANTIZE_DEQUANT_Q8_0:
            std::tie(result_vec, latency) = execute_quantize_dequant_q8_0(in_vec, inst.dim);
            break;
        case VM::REDUCE_MAX_ABS:
            std::tie(result_vec, latency) = execute_reduce_max_abs(in_vec, inst.dim);
            break;
        case VM::QUANTIZE_DEQUANT_Q8_0_EXT_SCALE:
            std::tie(result_vec, latency) = execute_quantize_dequant_q8_0_ext_scale(in_vec, w_vec, inst.dim);
            break;
        case VM::QUANTIZE_DEQUANT_Q8_0_BOUNDARY_FIX:
            std::tie(result_vec, latency) = execute_quantize_dequant_q8_0_boundary_fix(in_vec, w_vec, inst.dim);
            break;
        case VM::IDLE:
            result_vec = in_vec;
            latency = 0;
            break;
        default:
            throw std::invalid_argument("Unsupported vector operation");
    }

    auto output = std::make_shared<LLM_SIM_DATA::Data_t<T>>();
    output->id    = inst.out_id;
    output->value = std::move(result_vec);

    // step 1: Q8_0 round-trip output is still
    // physically stored as T=float (see Data_t::elem_bits's doc comment),
    // but real gpt-oss-20b transfers this activation as Q8_0 (8-bit) --
    // tag it here so downstream Cache/NoC latency accounting (Cache::
    // latency_cal, Router::data2packet, both already elem_bits-aware) sees
    // the true wire size. All other modes keep the sizeof(T)*8 default.
    if (inst.mode == VM::QUANTIZE_DEQUANT_Q8_0 ||
        inst.mode == VM::QUANTIZE_DEQUANT_Q8_0_EXT_SCALE ||
        inst.mode == VM::QUANTIZE_DEQUANT_Q8_0_BOUNDARY_FIX) {
        output->elem_bits = 8;
    }

    if (std::getenv("DEBUG_DUMP_VEC_STATS") && std::getenv("DEBUG_DUMP_VEC_STATS_HI") &&
        output->id >= static_cast<uint64_t>(std::atoll(std::getenv("DEBUG_DUMP_VEC_STATS"))) &&
        output->id <  static_cast<uint64_t>(std::atoll(std::getenv("DEBUG_DUMP_VEC_STATS_HI")))) {
        double vmin = std::numeric_limits<double>::infinity();
        double vmax = -std::numeric_limits<double>::infinity();
        int n_nan = 0, n_inf = 0;
        for (T x : output->value) {
            double d = static_cast<double>(x);
            if (std::isnan(d)) { ++n_nan; continue; }
            if (std::isinf(d)) { ++n_inf; continue; }
            if (d < vmin) vmin = d;
            if (d > vmax) vmax = d;
        }
        std::cerr << "[VEC-STATS] mode=" << static_cast<int>(inst.mode)
                   << " out_id=" << output->id << " size=" << output->value.size()
                   << " min=" << vmin << " max=" << vmax
                   << " n_nan=" << n_nan << " n_inf=" << n_inf << "\n" << std::flush;
    }

    assign_output_shape(inst, input, output);

    return {{output}, latency};
}

// ─── assign_output_shape ──────────────────────────────────────────────────────
// Sets output->size_ for every single-output mode. Shared by execute() (the
// compute path, called just above) and VectorUnit.cpp's transaction-only
// zero-fill path, so both paths always agree on the shape they tag --
// without this, a transaction-only run's Data_t could carry a different
// size_ than a compute run's for the exact same instruction, which the
// Cache write-request mechanism (data_bits/latency_cal) would then price
// differently depending on which mode happened to run.
//
// Rules:
//   IDLE        – full copy of input; size_ always matches input.
//   ADD_REDUCE,
//   DOT_PRODUCT – reduce to scalar.
//   TOP_K       – handled early above (dual-output); never reaches this code.
//   all others  – shape-preserving; output has exactly inst.dim elements.
//                 Reuse input->size_ when its product equals inst.dim so
//                 multi-dimensional layout is preserved; otherwise fall back
//                 to {inst.dim} to keep size_ consistent with value.size().
template <typename T>
void VectorOp<T>::assign_output_shape(
    const LLM_SIM_INST::VectorInst_t& inst,
    const LLM_SIM_DATA::DataPtr_t<T>& input,
    const LLM_SIM_DATA::DataPtr_t<T>& output)
{
    using VM2 = LLM_SIM_INST::VectorInst_t::VectorMode_t;

    // Helper: product of a size_ vector
    auto size_product = [](const std::vector<int>& sz) -> int {
        int p = 1;
        for (int d : sz) p *= d;
        return p;
    };

    switch (inst.mode) {
        case VM2::IDLE:
        case VM2::BIAS_ADD:
            // Full copy / broadcast-add – preserve full input shape.
            output->size_ = input->size_;
            break;
        case VM2::ADD_REDUCE:
        case VM2::DOT_PRODUCT:
            if (inst.all || input->size_.size() <= 1) {
                output->size_ = {1};
            } else {
                // Remove axis inst.dim from input->size_
                std::vector<int> out_shape;
                for (int i = 0; i < (int)input->size_.size(); ++i)
                    if (i != inst.dim) out_shape.push_back(input->size_[i]);
                output->size_ = out_shape.empty() ? std::vector<int>{1} : out_shape;
            }
            break;
        case VM2::TOP_K:
            // Should never reach here: TOP_K is handled in the early-return block above.
            output->size_ = {inst.dim};
            break;
        case VM2::SPARSE_GATE: {
            // Output is [chunk, n_experts] where n_experts = inst.dim.
            const int n_exp  = inst.dim;
            const int chunk  = static_cast<int>(output->value.size()) / n_exp;
            output->size_ = {chunk, n_exp};
            break;
        }
        default:
            // Shape-preserving: preserve multi-dimensional layout when input
            // total is a multiple of inst.dim (batched processing).
            if (!input->size_.empty() && size_product(input->size_) % inst.dim == 0
                    && static_cast<int>(output->value.size()) == size_product(input->size_)) {
                output->size_ = input->size_;
            } else {
                output->size_ = {static_cast<int>(output->value.size())};
            }
            break;
    }

    // Safety net: if the shape assigned above doesn't actually describe
    // value.size() (e.g. a future mode added to the switch's default case
    // without a matching execute_xxx output size), fall back to a flat
    // shape rather than let Cache's data_bits() silently mis-price this
    // write using a shape that disagrees with the real payload.
    int prod = 1;
    for (int d : output->size_) prod *= d;
    if (output->size_.empty() || prod != static_cast<int>(output->value.size())) {
        output->size_ = {static_cast<int>(output->value.size())};
    }
}

// ─── output_elem_count ────────────────────────────────────────────────────────
// The element count execute() would produce for this instruction, mirroring
// each execute_xxx()'s own output vector allocation exactly (see the
// "output(...)" size comment at each execute_xxx's allocation site above).
// Used by the transaction-only path to zero-fill the correctly-sized output
// without running the real arithmetic.
template <typename T>
size_t VectorOp<T>::output_elem_count(
    const LLM_SIM_INST::VectorInst_t& inst,
    const LLM_SIM_DATA::DataPtr_t<T>& input)
{
    using VM = LLM_SIM_INST::VectorInst_t::VectorMode_t;
    const size_t total = input->value.size();
    const size_t dim   = static_cast<size_t>(inst.dim);

    switch (inst.mode) {
        // execute_add/sub/mul/rms/softmax/swiglu/bias_add all allocate
        // output(total) or output(input.size()).
        case VM::ADD:
        case VM::SUB:
        case VM::MUL:
        case VM::RMS:
        case VM::SOFTMAX:
        case VM::SWIGLU:
        case VM::BIAS_ADD:
        case VM::IDLE:
            return total;
        // execute_recip/inv_sqrt/relu/sig/silu/quantize_dequant_q8_0(_ext_
        // scale/_boundary_fix)/div_constant all allocate output(dim).
        case VM::RECIP:
        case VM::INV_SQRT:
        case VM::RELU:
        case VM::SIG:
        case VM::SILU:
        case VM::QUANTIZE_DEQUANT_Q8_0:
        case VM::QUANTIZE_DEQUANT_Q8_0_EXT_SCALE:
        case VM::QUANTIZE_DEQUANT_Q8_0_BOUNDARY_FIX:
        case VM::DIV_CONSTANT:
            return dim;
        // execute_reduce_max_abs: single scalar {amax}.
        case VM::REDUCE_MAX_ABS:
            return 1;
        // execute_add_reduce/dot_product: scalar unless N-D reduce along an
        // axis (removes axis's extent from input->size_'s product).
        case VM::ADD_REDUCE:
        case VM::DOT_PRODUCT: {
            if (inst.all || input->size_.size() <= 1) return 1;
            size_t out_size = 1;
            for (size_t i = 0; i < input->size_.size(); ++i)
                if (static_cast<int>(i) != inst.dim) out_size *= static_cast<size_t>(input->size_[i]);
            return out_size;
        }
        // execute_sparse_gate: gate(chunk * n_experts), chunk = idx_in.size()/top_k.
        case VM::SPARSE_GATE: {
            const int n_experts = inst.dim;
            const int top_k     = static_cast<int>(inst.out_id2);
            const size_t chunk = (top_k > 0) ? (total / static_cast<size_t>(top_k)) : 0;
            return chunk * static_cast<size_t>(n_experts);
        }
        default:
            return dim;
    }
}

// ─── Element-wise arithmetic ──────────────────────────────────────────────────

template <typename T>
std::pair<std::vector<T>, uint64_t> VectorOp<T>::execute_add(
    const std::vector<T>& input, const std::vector<T>& weight, const int dim)
{
    const int total = static_cast<int>(input.size());
    assert(dim > 0 && total % dim == 0 && (int)weight.size() >= total);
    std::vector<T> output(total);
    for (int i = 0; i < total; ++i)
        output[i] = input[i] + weight[i];
    uint64_t latency = (uint64_t)(std::ceil((total / static_cast<float>(PROCESS_CHUNK)) - 1))
                       * ADD_LATENCY + ELEMENTWISE_OVERHEAD;
    return {output, latency};
}

template <typename T>
std::pair<std::vector<T>, uint64_t> VectorOp<T>::execute_sub(
    const std::vector<T>& input, const std::vector<T>& weight, const int dim)
{
    const int total = static_cast<int>(input.size());
    assert(dim > 0 && total % dim == 0 && (int)weight.size() >= total);
    std::vector<T> output(total);
    for (int i = 0; i < total; ++i)
        output[i] = input[i] - weight[i];
    uint64_t latency = (uint64_t)(std::ceil((total / static_cast<float>(PROCESS_CHUNK)) - 1))
                       * SUB_LATENCY + ELEMENTWISE_OVERHEAD;
    return {output, latency};
}

template <typename T>
std::pair<std::vector<T>, uint64_t> VectorOp<T>::execute_mul(
    const std::vector<T>& input, const std::vector<T>& weight, const int dim)
{
    const int total = static_cast<int>(input.size());
    assert(dim > 0 && total % dim == 0 && (int)weight.size() >= total);
    std::vector<T> output(total);
    for (int i = 0; i < total; ++i)
        output[i] = input[i] * weight[i];
    uint64_t latency = (uint64_t)(std::ceil((total / static_cast<float>(PROCESS_CHUNK)) - 1))
                       * MUL_LATENCY + ELEMENTWISE_OVERHEAD;
    return {output, latency};
}

template <typename T>
std::pair<std::vector<T>, uint64_t> VectorOp<T>::execute_recip(
    const std::vector<T>& input, const int dim)
{
    assert(dim > 0 && dim <= (int)input.size());
    std::vector<T> output(dim);
    for (int i = 0; i < dim; ++i) {
        assert(input[i] != T(0) && "RECIP: division by zero");
        output[i] = T(1) / input[i];
    }
    uint64_t latency = (uint64_t)(std::ceil((dim / static_cast<float>(PROCESS_CHUNK)) - 1))
                       * RECIP_LATENCY + ELEMENTWISE_OVERHEAD;
    return {output, latency};
}

template <typename T>
std::pair<std::vector<T>, uint64_t> VectorOp<T>::execute_inv_sqrt(
    const std::vector<T>& input, const int dim)
{
    assert(dim > 0 && dim <= (int)input.size());
    std::vector<T> output(dim);
    for (int i = 0; i < dim; ++i) {
        assert(input[i] > T(0) && "INV_SQRT: non-positive input");
        output[i] = T(1) / static_cast<T>(std::sqrt(static_cast<float>(input[i])));
    }
    uint64_t latency = (uint64_t)(std::ceil((dim / static_cast<float>(PROCESS_CHUNK)) - 1))
                       * INV_SQRT_LATENCY + ELEMENTWISE_OVERHEAD;
    return {output, latency};
}

// ─── Activation functions ─────────────────────────────────────────────────────

template <typename T>
std::pair<std::vector<T>, uint64_t> VectorOp<T>::execute_relu(
    const std::vector<T>& input, const int dim)
{
    assert(dim > 0 && dim <= (int)input.size());
    std::vector<T> output(dim);
    for (int i = 0; i < dim; ++i)
        output[i] = input[i] > T(0) ? input[i] : T(0);
    uint64_t latency = (uint64_t)(std::ceil((dim / static_cast<float>(PROCESS_CHUNK)) - 1))
                       * RELU_LATENCY + ELEMENTWISE_OVERHEAD;
    return {output, latency};
}

template <typename T>
std::pair<std::vector<T>, uint64_t> VectorOp<T>::execute_sig(
    const std::vector<T>& input, const int dim)
{
    using std::exp;
    assert(dim > 0 && dim <= (int)input.size());
    std::vector<T> output(dim);
    for (int i = 0; i < dim; ++i) {
        float x = static_cast<float>(input[i]);
        float s = (x >= 0.f) ? (1.f / (1.f + exp(-x)))
                              : (exp(x) / (1.f + exp(x)));
        output[i] = T(s);
    }
    uint64_t latency = (uint64_t)(std::ceil((dim / static_cast<float>(PROCESS_CHUNK)) - 1))
                       * SIG_LATENCY + ELEMENTWISE_OVERHEAD;
    return {output, latency};
}

template <typename T>
std::pair<std::vector<T>, uint64_t> VectorOp<T>::execute_silu(
    const std::vector<T>& input, const int dim)
{
    using std::exp;
    assert(dim > 0 && dim <= (int)input.size());
    std::vector<T> output(dim);
    for (int i = 0; i < dim; ++i) {
        float x = static_cast<float>(input[i]);
        float s = (x >= 0.f) ? (1.f / (1.f + exp(-x)))
                              : (exp(x) / (1.f + exp(x)));
        output[i] = T(x * s);
    }
    uint64_t latency = (uint64_t)(std::ceil((dim / static_cast<float>(PROCESS_CHUNK)) - 1))
                       * SILU_LATENCY + SILU_OVERHEAD;
    return {output, latency};
}

template <typename T>
std::pair<std::vector<T>, uint64_t> VectorOp<T>::execute_quantize_dequant_q8_0(
    const std::vector<T>& input, const int dim)
{
    // dim may exceed input.size() when zero-padding is requested (e.g. TP-4
    // shard_dim=720 padded to 736 for aligned Q8_0 blocks).  Extra elements
    // are treated as zero so they don't distort the per-block scale.
    // dim may also be less than input.size() (normal sub-slice quantization).
    assert(dim > 0);
    std::vector<float> buf(dim, 0.0f);
    const int copy_n = std::min(dim, (int)input.size());
    for (int i = 0; i < copy_n; ++i) buf[i] = static_cast<float>(input[i]);
    LLM_SIM_PRECISION::quantize_dequantize_q8_0_inplace(buf);
    std::vector<T> output(dim);
    for (int i = 0; i < dim; ++i) output[i] = T(buf[i]);
    uint64_t latency = (uint64_t)(std::ceil((dim / static_cast<float>(PROCESS_CHUNK)) - 1))
                       * QUANTIZE_DEQUANT_Q8_0_LATENCY + QUANTIZE_DEQUANT_Q8_0_OVERHEAD;
    return {output, latency};
}

template <typename T>
std::pair<std::vector<T>, uint64_t> VectorOp<T>::execute_reduce_max_abs(
    const std::vector<T>& input, const int dim)
{
    assert(dim > 0 && dim <= (int)input.size());
    float amax = 0.0f;
    for (int i = 0; i < dim; ++i)
        amax = LLM_SIM_PRECISION::nan_safe_max(amax, std::fabs(static_cast<float>(input[i])));
    std::vector<T> output = {T(amax)};
    // Same cost as a single ADD_REDUCE pass
    uint64_t latency = (uint64_t)(std::ceil((dim / static_cast<float>(PROCESS_CHUNK)) - 1))
                       * ADD_REDUCE_LATENCY + ADD_REDUCE_OVERHEAD;
    return {output, latency};
}

template <typename T>
std::pair<std::vector<T>, uint64_t> VectorOp<T>::execute_quantize_dequant_q8_0_ext_scale(
    const std::vector<T>& input, const std::vector<T>& ext_scale_vec, const int dim)
{
    // ext_scale_vec[0] = global max_abs.  All 32-element blocks use this single scale.
    assert(dim > 0 && !ext_scale_vec.empty());
    const float global_max = static_cast<float>(ext_scale_vec[0]);
    float d = LLM_SIM_PRECISION::f32_to_f16_to_f32(global_max / 127.0f);
    float inv_d = (d > 0.0f) ? (1.0f / d) : 0.0f;

    std::vector<float> buf(dim, 0.0f);
    const int copy_n = std::min(dim, (int)input.size());
    for (int i = 0; i < copy_n; ++i) buf[i] = static_cast<float>(input[i]);

    for (int i = 0; i < dim; ++i) {
        float scaled = buf[i] * inv_d;
        scaled = LLM_SIM_PRECISION::nan_safe_max(-127.0f, LLM_SIM_PRECISION::nan_safe_min(127.0f, scaled));
        int8_t q = static_cast<int8_t>(std::lround(scaled));
        buf[i] = static_cast<float>(q) * d;
    }
    std::vector<T> output(dim);
    for (int i = 0; i < dim; ++i) output[i] = T(buf[i]);
    uint64_t latency = (uint64_t)(std::ceil((dim / static_cast<float>(PROCESS_CHUNK)) - 1))
                       * QUANTIZE_DEQUANT_Q8_0_LATENCY + QUANTIZE_DEQUANT_Q8_0_OVERHEAD;
    return {output, latency};
}

template <typename T>
std::pair<std::vector<T>, uint64_t> VectorOp<T>::execute_quantize_dequant_q8_0_boundary_fix(
    const std::vector<T>& input, const std::vector<T>& ext_scale_vec, const int dim)
{
    // Standard per-block Q8_0, but the last block's max_abs is clamped to
    // max(local_last_block_max, ext_scale_vec[0]).
    // This corrects the cross-shard boundary block when TP-N splits FFN_DIM
    // at a non-32-multiple boundary (e.g., shard_dim=720, padded to 736).
    assert(dim > 0 && dim % 32 == 0 && !ext_scale_vec.empty());
    const float ext_max = static_cast<float>(ext_scale_vec[0]);
    const int n_blocks = dim / 32;

    std::vector<float> buf(dim, 0.0f);
    const int copy_n = std::min(dim, (int)input.size());
    for (int i = 0; i < copy_n; ++i) buf[i] = static_cast<float>(input[i]);

    std::vector<T> output(dim);
    for (int b = 0; b < n_blocks; ++b) {
        const int start = b * 32;
        float block_max = 0.0f;
        for (int i = start; i < start + 32; ++i) block_max = LLM_SIM_PRECISION::nan_safe_max(block_max, std::fabs(buf[i]));
        if (b == n_blocks - 1) block_max = LLM_SIM_PRECISION::nan_safe_max(block_max, ext_max);  // boundary fix
        float d = LLM_SIM_PRECISION::f32_to_f16_to_f32(block_max / 127.0f);
        float inv_d = (d > 0.0f) ? (1.0f / d) : 0.0f;
        for (int i = start; i < start + 32; ++i) {
            float scaled = buf[i] * inv_d;
            scaled = LLM_SIM_PRECISION::nan_safe_max(-127.0f, LLM_SIM_PRECISION::nan_safe_min(127.0f, scaled));
            int8_t q = static_cast<int8_t>(std::lround(scaled));
            output[i] = T(static_cast<float>(q) * d);
        }
    }
    uint64_t latency = (uint64_t)(std::ceil((dim / static_cast<float>(PROCESS_CHUNK)) - 1))
                       * QUANTIZE_DEQUANT_Q8_0_LATENCY + QUANTIZE_DEQUANT_Q8_0_OVERHEAD;
    return {output, latency};
}

template <typename T>
std::pair<std::vector<T>, uint64_t> VectorOp<T>::execute_swiglu(
    const std::vector<T>& input, const std::vector<T>& weight, const int dim)
{
    // swiglu_oai (gpt-oss-20b's MoE activation). Verified against
    // ggml_compute_forward_swiglu_oai_f32 via
    // tools/run_gpt_oss_transformer_block.py::swiglu_oai. NOT plain silu(gate)*up:
    // gate is clamped from above only, up is clamped both ways and gets a +1
    // offset before the elementwise product, and the sigmoid slope is
    // alpha=1.702 (not 1.0).
    //   in = x_id = gate (caller convention: emit.py vec_swiglu(dim, gate_id, up_id, ...))
    //   weight = w_id = up
    const int total = static_cast<int>(input.size());
    assert(dim > 0 && total % dim == 0 && (int)weight.size() >= total);
    using std::exp;
    const float alpha = LLM_SIM_VEC::SWIGLU_OAI_ALPHA;
    const float limit  = LLM_SIM_VEC::SWIGLU_OAI_LIMIT;
    std::vector<T> output(total);
    for (int i = 0; i < total; ++i) {
        float gate = std::min(static_cast<float>(input[i]), limit);
        float up   = LLM_SIM_PRECISION::nan_safe_max(-limit, LLM_SIM_PRECISION::nan_safe_min(static_cast<float>(weight[i]), limit));
        float s    = (alpha * gate >= 0.f) ? (1.f / (1.f + exp(-alpha * gate)))
                                            : (exp(alpha * gate) / (1.f + exp(alpha * gate)));
        float out_glu = gate * s;
        output[i] = T(out_glu * (up + 1.0f));
    }
    uint64_t latency = (uint64_t)(std::ceil((dim / static_cast<float>(PROCESS_CHUNK)) - 1))
                       * SWIGLU_LATENCY + SWIGLU_OVERHEAD;
    return {output, latency};
}

// ─── Normalisation / transform ────────────────────────────────────────────────

template <typename T>
std::pair<std::vector<T>, uint64_t> VectorOp<T>::execute_rms(
    const std::vector<T>& input, const std::vector<T>& weight, const int dim)
{
    using std::sqrt;
    const int total = static_cast<int>(input.size());
    assert(dim > 0 && total % dim == 0 && dim <= (int)weight.size());
    const int num_batches = total / dim;
    const float eps = 1e-6f;
    std::vector<T> output(total);
    for (int b = 0; b < num_batches; ++b) {
        const int base = b * dim;
        float sum_sq = 0.0f;
        for (int i = 0; i < dim; ++i)
            sum_sq += static_cast<float>(input[base + i]) * static_cast<float>(input[base + i]);
        float rms = sqrt(sum_sq / dim + eps);
        for (int i = 0; i < dim; ++i) {
            float in_val = static_cast<float>(input[base + i]);
            float w_val  = static_cast<float>(weight[i]);
            output[base + i] = T((in_val * w_val) / rms);
        }
    }
    const uint64_t per_batch_lat = (uint64_t)(std::ceil((dim / static_cast<float>(PROCESS_CHUNK)) - 1))
                                  * RMS_LATENCY + RMS_OVERHEAD;
    return {output, (uint64_t)num_batches * per_batch_lat};
}

template <typename T>
std::pair<std::vector<T>, uint64_t> VectorOp<T>::execute_softmax(
    const std::vector<T>& input, const int dim, const std::vector<T>& sink)
{
    using std::exp;
    const int total = static_cast<int>(input.size());
    assert(dim > 0 && total % dim == 0);
    const int num_batches = total / dim;
    const bool has_sink = !sink.empty();
    const int sink_n = static_cast<int>(sink.size());
    // sink broadcasts cyclically over batches: row b uses sink[b % sink_n].
    // This lets one per-Q-head sink vector (len = n_heads) cover all softmax
    // rows when num_batches = n_tokens * n_heads (prefill: rows ordered
    // [token][head]) without inst-gen having to pre-tile the sink tensor.
    if (has_sink)
        assert(num_batches % sink_n == 0);
    std::vector<T> output(total);
    for (int b = 0; b < num_batches; ++b) {
        const int base = b * dim;
        float x_max = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < dim; ++i) {
            float in = static_cast<float>(input[base + i]);
            if (in > x_max) x_max = in;
        }
        const float sk = has_sink ? static_cast<float>(sink[b % sink_n]) : 0.0f;
        if (has_sink && sk > x_max) x_max = sk;
        std::vector<float> exps(dim);
        float denom = 0.0f;
        for (int i = 0; i < dim; ++i) {
            exps[i] = exp(static_cast<float>(input[base + i]) - x_max);
            denom += exps[i];
        }
        if (has_sink)
            denom += exp(sk - x_max);
        if (denom == 0.0f) denom = 1.0f;
        for (int i = 0; i < dim; ++i)
            output[base + i] = T(exps[i] / denom);
    }
    uint64_t latency = (uint64_t)(std::ceil(total / static_cast<float>(PROCESS_CHUNK)))
                       * SOFTMAX_LATENCY + SOFTMAX_OVERHEAD;

    // TEMP DEBUG (0/24-numerically-match-oracle
    // investigation, 2026-08-10): dump softmax's output row(s) + the sink
    // bias actually applied, gated on total matching the expected dim*heads
    // shape for the layer/token under investigation (identified indirectly
    // via `dim`, since VectorOp has no core_id -- caller passes DEBUG_DUMP_
    // SOFTMAX=<dim> to only fire for the QK-score-sized softmax calls, not
    // every other softmax in the pipeline, e.g. the MoE router's own).
    if (std::getenv("DEBUG_DUMP_SOFTMAX") && dim == std::atoi(std::getenv("DEBUG_DUMP_SOFTMAX"))) {
        std::cerr << "[SOFTMAX-DEBUG] dim=" << dim << " num_batches=" << num_batches
                  << " has_sink=" << has_sink << " sink_n=" << sink_n;
        if (has_sink) {
            std::cerr << " sink_vals:";
            for (int i = 0; i < sink_n; ++i) std::cerr << " " << static_cast<float>(sink[i]);
        }
        std::cerr << " out_row0:";
        for (int i = 0; i < dim; ++i) std::cerr << " " << static_cast<float>(output[i]);
        std::cerr << "\n" << std::flush;
    }

    return {output, latency};
}

template <typename T>
std::pair<std::vector<T>, uint64_t> VectorOp<T>::execute_rope(
    const std::vector<T>& input, const int dim, const int token_pos, const float theta_base)
{
    // NEOX-style pairing (front-half/back-half, x[k]/x[k+half]) + YaRN scaling.
    // Verified against tools/run_gpt_oss_transformer_block.py::apply_rope_yarn_neox
    // (itself verified against ggml's rope_yarn / ggml_rope_yarn_corr_dims /
    // rotate_pairs<T>(n_dims, n_dims/2, ..., scale=2)). Real gpt-oss-20b RoPE is
    // NOT the plain interleaved x[2k]/x[2k+1] theta-base rotation this used to be.
    assert(dim > 0 && dim <= VEC_SIZE);
    std::vector<T> output(dim);
    const int half = dim / 2;

    const float freq_scale  = ROPE_YARN_FREQ_SCALE;
    const int   n_ctx_orig   = ROPE_YARN_N_CTX_ORIG;
    const float beta_fast    = ROPE_YARN_BETA_FAST;
    const float beta_slow    = ROPE_YARN_BETA_SLOW;
    const float ext_factor   = ROPE_YARN_EXT_FACTOR;
    const float attn_factor  = ROPE_YARN_ATTN_FACTOR;

    // YaRN correction-dimension ramp bounds (corr_dim(n_rot), ggml_rope_yarn_corr_dims).
    constexpr float kPi = 3.14159265358979323846f;
    auto corr_dim = [&](float n_rot) -> float {
        return dim * std::log(static_cast<float>(n_ctx_orig) / (n_rot * 2.0f * kPi))
               / (2.0f * std::log(theta_base));
    };
    const float low  = std::max(0.0f, std::floor(corr_dim(beta_fast)));
    const float high = std::min(static_cast<float>(dim) - 1.0f, std::ceil(corr_dim(beta_slow)));

    // Position-independent magnitude factor (mscale), applies uniformly incl. pos 0.
    const float mscale = (ext_factor != 0.0f)
        ? attn_factor * (1.0f + 0.1f * std::log(1.0f / freq_scale))
        : attn_factor;

    for (int k = 0; k < half; ++k) {
        const float a = static_cast<float>(input[k]);
        const float b = static_cast<float>(input[k + half]);

        const float theta_extrap = static_cast<float>(token_pos) * std::pow(theta_base, -(2.0f * k) / dim);
        const float theta_interp = freq_scale * theta_extrap;
        float theta;
        if (ext_factor != 0.0f) {
            float ramp = (static_cast<float>(k) - low) / std::max(0.001f, high - low);
            ramp = 1.0f - std::min(1.0f, std::max(0.0f, ramp));
            const float ramp_mix = ramp * ext_factor;
            theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        } else {
            theta = theta_interp;
        }

        const float cos_v = std::cos(theta) * mscale;
        const float sin_v = std::sin(theta) * mscale;
        output[k]        = T((a * cos_v) - (b * sin_v));
        output[k + half] = T((a * sin_v) + (b * cos_v));
    }
    uint64_t latency = (uint64_t)(std::ceil((dim / static_cast<float>(PROCESS_CHUNK)) - 1))
                       * ROPE_LATENCY + ROPE_OVERHEAD;
    return {output, latency};
}

// ─── Reduction ────────────────────────────────────────────────────────────────

// ── reduction helpers ────────────────────────────────────────────────────────
// Compute row-major strides for a shape vector.
static std::vector<int> make_strides(const std::vector<int>& shape) {
    int n = (int)shape.size();
    std::vector<int> strides(n);
    if (n == 0) return strides;
    strides[n-1] = 1;
    for (int i = n-2; i >= 0; --i)
        strides[i] = strides[i+1] * shape[i+1];
    return strides;
}

template <typename T>
std::pair<std::vector<T>, uint64_t> VectorOp<T>::execute_add_reduce(
    const std::vector<T>& input, const std::vector<int>& shape, bool all, int axis)
{
    const int total = static_cast<int>(input.size());

    if (all || shape.size() <= 1) {
        // Reduce every element → scalar
        float sum = 0.0f;
        for (const auto& v : input) sum += static_cast<float>(v);
        uint64_t lat = (uint64_t)std::ceil(total / float(PROCESS_CHUNK))
                       * ADD_REDUCE_LATENCY + ADD_REDUCE_OVERHEAD;
        return {{T(sum)}, lat};
    }

    // N-D reduce along axis
    const int ndim = (int)shape.size();
    assert(axis >= 0 && axis < ndim);
    const auto strides = make_strides(shape);

    std::vector<int> out_shape;
    for (int i = 0; i < ndim; ++i)
        if (i != axis) out_shape.push_back(shape[i]);
    int out_size = 1;
    for (int d : out_shape) out_size *= d;
    const auto out_strides = make_strides(out_shape);

    std::vector<T> output(out_size, T(0));
    for (int flat = 0; flat < total; ++flat) {
        int rem = flat, out_flat = 0, oi = 0;
        for (int i = 0; i < ndim; ++i) {
            int coord = rem / strides[i];
            rem      %= strides[i];
            if (i != axis) out_flat += coord * out_strides[oi++];
        }
        output[out_flat] = T(static_cast<float>(output[out_flat]) + static_cast<float>(input[flat]));
    }
    uint64_t lat = (uint64_t)std::ceil(total / float(PROCESS_CHUNK))
                   * ADD_REDUCE_LATENCY + ADD_REDUCE_OVERHEAD;
    return {output, lat};
}

template <typename T>
std::pair<std::vector<T>, uint64_t> VectorOp<T>::execute_dot_product(
    const std::vector<T>& input, const std::vector<T>& weight,
    const std::vector<int>& shape, bool all, int axis)
{
    const int total = static_cast<int>(input.size());
    assert(total == (int)weight.size());

    if (all || shape.size() <= 1) {
        // Dot product of every element → scalar
        float dot = 0.0f;
        for (int i = 0; i < total; ++i)
            dot += static_cast<float>(input[i]) * static_cast<float>(weight[i]);
        uint64_t lat = (uint64_t)std::ceil(total / float(PROCESS_CHUNK))
                       * DOT_PRODUCT_LATENCY + DOT_PRODUCT_OVERHEAD;
        return {{T(dot)}, lat};
    }

    // N-D dot along axis
    const int ndim = (int)shape.size();
    assert(axis >= 0 && axis < ndim);
    const auto strides = make_strides(shape);

    std::vector<int> out_shape;
    for (int i = 0; i < ndim; ++i)
        if (i != axis) out_shape.push_back(shape[i]);
    int out_size = 1;
    for (int d : out_shape) out_size *= d;
    const auto out_strides = make_strides(out_shape);

    std::vector<T> output(out_size, T(0));
    for (int flat = 0; flat < total; ++flat) {
        int rem = flat, out_flat = 0, oi = 0;
        for (int i = 0; i < ndim; ++i) {
            int coord = rem / strides[i];
            rem      %= strides[i];
            if (i != axis) out_flat += coord * out_strides[oi++];
        }
        output[out_flat] = T(static_cast<float>(output[out_flat])
                            + static_cast<float>(input[flat]) * static_cast<float>(weight[flat]));
    }
    uint64_t lat = (uint64_t)std::ceil(total / float(PROCESS_CHUNK))
                   * DOT_PRODUCT_LATENCY + DOT_PRODUCT_OVERHEAD;
    return {output, lat};
}

template <typename T>
std::pair<std::vector<T>, uint64_t> VectorOp<T>::execute_top_k(
    const std::vector<T>& input, const int dim)
{
    const int N = static_cast<int>(input.size());
    assert(dim > 0 && dim <= N);
    // Build full index set [0..N-1], sort descending by value, keep top dim
    std::vector<int> indices(N);
    std::iota(indices.begin(), indices.end(), 0);
    std::stable_sort(indices.begin(), indices.end(),
        [&](int a, int b) {
            return static_cast<float>(input[a]) > static_cast<float>(input[b]);
        });
    std::vector<T> output(dim);
    for (int i = 0; i < dim; ++i)
        output[i] = T(indices[i]);
    uint64_t latency = (uint64_t)(std::ceil(dim / static_cast<float>(PROCESS_CHUNK)))
                       * TOP_K_LATENCY + TOP_K_OVERHEAD;
    return {output, latency};
}

// ─── execute_top_k_dual ────────────────────────────────────────────────────────────────
// Dual-output TOP_K:
//   n_experts > 0: per-row top_k over rows of size n_experts, with renormalised values
//   n_experts == 0: global top_k (legacy fallback)
// Returns (sorted_indices_as_T, renorm_values, latency).

template <typename T>
std::tuple<std::vector<T>, std::vector<T>, uint64_t>
VectorOp<T>::execute_top_k_dual(const std::vector<T>& input, const int dim, const int n_experts)
{
    const int N = static_cast<int>(input.size());
    assert(dim > 0);

    if (n_experts > 0) {
        // Per-row top-k with renormalised weights.
        assert(n_experts >= dim && N % n_experts == 0);
        const int chunk = N / n_experts;
        const int top_k = dim;  // dim = k

        std::vector<T> idx_out(chunk * top_k);
        std::vector<T> val_out(chunk * top_k);

        std::vector<int> row_idx(n_experts);
        for (int t = 0; t < chunk; ++t) {
            const int row_base = t * n_experts;
            std::iota(row_idx.begin(), row_idx.end(), 0);
            std::stable_sort(row_idx.begin(), row_idx.end(),
                [&](int a, int b) {
                    return static_cast<float>(input[row_base + a]) >
                           static_cast<float>(input[row_base + b]);
                });
            // Re-sort the top-k indices by expert index (ascending) so that
            // score_slices[ki] matches selected_experts[ki] in the instruction
            // generator (both use expert-index order, not score-rank order).
            std::sort(row_idx.begin(), row_idx.begin() + top_k);
            // §0b.3: softmax over the selected top-k
            // RAW logits (LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT, norm_w=
            // False) -- NOT linear x/sum renormalisation. Selection order (top-k
            // on raw logits) is unchanged; only the weighting formula differs.
            float max_logit = -std::numeric_limits<float>::infinity();
            for (int k = 0; k < top_k; ++k)
                max_logit = LLM_SIM_PRECISION::nan_safe_max(max_logit, static_cast<float>(input[row_base + row_idx[k]]));
            float sum_exp = 0.f;
            for (int k = 0; k < top_k; ++k)
                sum_exp += std::exp(static_cast<float>(input[row_base + row_idx[k]]) - max_logit);
            for (int k = 0; k < top_k; ++k) {
                idx_out[t * top_k + k] = T(row_idx[k]);
                val_out[t * top_k + k] =
                    T(std::exp(static_cast<float>(input[row_base + row_idx[k]]) - max_logit) / sum_exp);
            }
        }
        uint64_t latency = (uint64_t)(std::ceil(top_k / static_cast<float>(PROCESS_CHUNK)))
                           * TOP_K_LATENCY + TOP_K_OVERHEAD;
        return {idx_out, val_out, latency};
    } else {
        // Legacy: global top_k over all N elements.
        assert(dim <= N);
        std::vector<int> indices(N);
        std::iota(indices.begin(), indices.end(), 0);
        std::stable_sort(indices.begin(), indices.end(),
            [&](int a, int b) {
                return static_cast<float>(input[a]) > static_cast<float>(input[b]);
            });
        std::vector<T> idx_out(dim), val_out(dim);
        for (int i = 0; i < dim; ++i) {
            idx_out[i] = T(indices[i]);
            val_out[i] = input[indices[i]];
        }
        uint64_t latency = (uint64_t)(std::ceil(dim / static_cast<float>(PROCESS_CHUNK)))
                           * TOP_K_LATENCY + TOP_K_OVERHEAD;
        return {idx_out, val_out, latency};
    }
}

template <typename T>
std::pair<std::vector<T>, uint64_t> VectorOp<T>::execute_div_constant(
    const std::vector<T>& input, const int dim)
{
    assert(dim > 0 && dim <= (int)input.size());
    std::vector<T> output(dim);
    for (int i = 0; i < dim; ++i)
        output[i] = T(static_cast<float>(input[i]) / DIV_CONSTANT_VALUE);
    uint64_t latency = (uint64_t)(std::ceil((dim / static_cast<float>(PROCESS_CHUNK)) - 1))
                       * DIV_CONSTANT_LATENCY + ELEMENTWISE_OVERHEAD;
    return {output, latency};
}


// ─── Broadcast bias add ───────────────────────────────────────────────────────
// input : [N_total] elements (row-major, N_total must be a multiple of feature_dim)
// bias  : [feature_dim] elements
// output: same size/shape as input, output[r*feature_dim + d] = input[r*feature_dim + d] + bias[d]
template <typename T>
std::pair<std::vector<T>, uint64_t> VectorOp<T>::execute_bias_add(
    const std::vector<T>& input, const std::vector<T>& bias, const int feature_dim)
{
    const int total = static_cast<int>(input.size());
    assert(feature_dim > 0 && feature_dim <= total);
    assert(feature_dim <= (int)bias.size());
    assert(total % feature_dim == 0 && "BIAS_ADD: total input elements must be divisible by feature_dim");
    std::vector<T> output(total);
    for (int i = 0; i < total; ++i)
        output[i] = T(static_cast<float>(input[i]) + static_cast<float>(bias[i % feature_dim]));
    uint64_t latency = (uint64_t)(std::ceil((total / static_cast<float>(PROCESS_CHUNK)) - 1))
                       * BIAS_ADD_LATENCY + ELEMENTWISE_OVERHEAD;
    return {output, latency};
}

// ─── execute_sparse_gate ─────────────────────────────────────────────────────
// Converts per-token top-k routing (indices + renorm weights) to a dense
// sparse gate tensor: gate[t, e] = renorm_weight if e is in top-k for token t,
// else 0.
//
// idx_in : [chunk * top_k]  — top-k expert indices per token (stored as T/float)
// val_in : [chunk * top_k]  — renorm weights (from execute_top_k_dual per-row)
// n_experts : row width of the output gate (= inst.dim)
// top_k     : number of top-k entries per token (= inst.out_id2)
// Output: gate [chunk * n_experts]

template <typename T>
std::pair<std::vector<T>, uint64_t>
VectorOp<T>::execute_sparse_gate(
    const std::vector<T>& idx_in,
    const std::vector<T>& val_in,
    const int n_experts,
    const int top_k)
{
    const int total = static_cast<int>(idx_in.size());
    assert(top_k > 0 && n_experts >= top_k);
    assert(total % top_k == 0);
    const int chunk = total / top_k;

    std::vector<T> gate(chunk * n_experts, T{0});
    for (int t = 0; t < chunk; ++t) {
        for (int k = 0; k < top_k; ++k) {
            const int e = static_cast<int>(std::round(static_cast<float>(idx_in[t * top_k + k])));
            assert(e >= 0 && e < n_experts);
            gate[t * n_experts + e] = val_in[t * top_k + k];
        }
    }
    uint64_t latency = (uint64_t)(std::ceil(chunk * top_k / static_cast<float>(PROCESS_CHUNK)))
                       * TOP_K_LATENCY + TOP_K_OVERHEAD;
    return {gate, latency};
}

// ─── latency_only ─────────────────────────────────────────────────────────────
// Returns the same latency as execute() but does NO arithmetic — for use in
// compute_=false (transaction-only) mode.  Formulas mirror each execute_xxx().

template <typename T>
uint64_t VectorOp<T>::latency_only(
    const LLM_SIM_INST::VectorInst_t& inst,
    const LLM_SIM_DATA::DataPtr_t<T>& input,
    const LLM_SIM_DATA::DataPtr_t<T>& /*weight*/) const
{
    using VM = LLM_SIM_INST::VectorInst_t::VectorMode_t;
    const int total = static_cast<int>(input->value.size());
    const int dim   = inst.dim;

    switch (inst.mode) {
        case VM::ROPE:
            return (uint64_t)(std::ceil((dim / static_cast<float>(PROCESS_CHUNK)) - 1))
                   * ROPE_LATENCY + ROPE_OVERHEAD;

        case VM::TOP_K:
            return (uint64_t)(std::ceil(dim / static_cast<float>(PROCESS_CHUNK)))
                   * TOP_K_LATENCY + TOP_K_OVERHEAD;

        case VM::ADD:
            return (uint64_t)(std::ceil((total / static_cast<float>(PROCESS_CHUNK)) - 1))
                   * ADD_LATENCY + ELEMENTWISE_OVERHEAD;
        case VM::SUB:
            return (uint64_t)(std::ceil((total / static_cast<float>(PROCESS_CHUNK)) - 1))
                   * SUB_LATENCY + ELEMENTWISE_OVERHEAD;
        case VM::MUL:
            return (uint64_t)(std::ceil((total / static_cast<float>(PROCESS_CHUNK)) - 1))
                   * MUL_LATENCY + ELEMENTWISE_OVERHEAD;
        case VM::BIAS_ADD:
            return (uint64_t)(std::ceil((total / static_cast<float>(PROCESS_CHUNK)) - 1))
                   * BIAS_ADD_LATENCY + ELEMENTWISE_OVERHEAD;

        case VM::RECIP:
            return (uint64_t)(std::ceil((dim / static_cast<float>(PROCESS_CHUNK)) - 1))
                   * RECIP_LATENCY + ELEMENTWISE_OVERHEAD;
        case VM::INV_SQRT:
            return (uint64_t)(std::ceil((dim / static_cast<float>(PROCESS_CHUNK)) - 1))
                   * INV_SQRT_LATENCY + ELEMENTWISE_OVERHEAD;
        case VM::RELU:
            return (uint64_t)(std::ceil((dim / static_cast<float>(PROCESS_CHUNK)) - 1))
                   * RELU_LATENCY + ELEMENTWISE_OVERHEAD;
        case VM::SIG:
            return (uint64_t)(std::ceil((dim / static_cast<float>(PROCESS_CHUNK)) - 1))
                   * SIG_LATENCY + ELEMENTWISE_OVERHEAD;
        case VM::SILU:
            return (uint64_t)(std::ceil((dim / static_cast<float>(PROCESS_CHUNK)) - 1))
                   * SILU_LATENCY + SILU_OVERHEAD;
        case VM::DIV_CONSTANT:
            return (uint64_t)(std::ceil((dim / static_cast<float>(PROCESS_CHUNK)) - 1))
                   * DIV_CONSTANT_LATENCY + ELEMENTWISE_OVERHEAD;

        case VM::SWIGLU:
            return (uint64_t)(std::ceil((dim / static_cast<float>(PROCESS_CHUNK)) - 1))
                   * SWIGLU_LATENCY + SWIGLU_OVERHEAD;

        case VM::RMS: {
            const int num_batches = total / (dim > 0 ? dim : 1);
            const uint64_t per_batch_lat = (uint64_t)(std::ceil((dim / static_cast<float>(PROCESS_CHUNK)) - 1))
                                          * RMS_LATENCY + RMS_OVERHEAD;
            return (uint64_t)num_batches * per_batch_lat;
        }

        case VM::SOFTMAX:
            return (uint64_t)(std::ceil(total / static_cast<float>(PROCESS_CHUNK)))
                   * SOFTMAX_LATENCY + SOFTMAX_OVERHEAD;

        case VM::ADD_REDUCE:
            return (uint64_t)std::ceil(total / static_cast<float>(PROCESS_CHUNK))
                   * ADD_REDUCE_LATENCY + ADD_REDUCE_OVERHEAD;
        case VM::DOT_PRODUCT:
            return (uint64_t)std::ceil(total / static_cast<float>(PROCESS_CHUNK))
                   * DOT_PRODUCT_LATENCY + DOT_PRODUCT_OVERHEAD;

        case VM::SPARSE_GATE:
            return (uint64_t)(std::ceil(total / static_cast<float>(PROCESS_CHUNK)))
                   * TOP_K_LATENCY + TOP_K_OVERHEAD;

        case VM::REDUCE_MAX_ABS:
            return (uint64_t)(std::ceil((dim / static_cast<float>(PROCESS_CHUNK)) - 1))
                   * ADD_REDUCE_LATENCY + ADD_REDUCE_OVERHEAD;

        case VM::QUANTIZE_DEQUANT_Q8_0:
            return (uint64_t)(std::ceil((dim / static_cast<float>(PROCESS_CHUNK)) - 1))
                   * QUANTIZE_DEQUANT_Q8_0_LATENCY + QUANTIZE_DEQUANT_Q8_0_OVERHEAD;

        case VM::QUANTIZE_DEQUANT_Q8_0_EXT_SCALE:
        case VM::QUANTIZE_DEQUANT_Q8_0_BOUNDARY_FIX:
            return (uint64_t)(std::ceil((dim / static_cast<float>(PROCESS_CHUNK)) - 1))
                   * QUANTIZE_DEQUANT_Q8_0_LATENCY + QUANTIZE_DEQUANT_Q8_0_OVERHEAD;

        case VM::IDLE:
            return 0;

        default:
            return ELEMENTWISE_OVERHEAD;
    }
}

template class VectorOp<float>;
