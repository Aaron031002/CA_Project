#ifndef LLM_SIM_VECTOROP_H
#define LLM_SIM_VECTOROP_H

#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <tuple>
#include <utility>

#include "Core/VectorUnit/vector_params.h"
#include "Core/VectorUnit/vector_utils.h"
#include "utils/BaseInst.h"
#include "utils/Data.h"

template <typename T>
class VectorOp {
    public:
        VectorOp();
        // Returns a list of DataPtr_t (1 for most ops; 2 for TOP_K: [indices, values])
        std::pair<std::vector<LLM_SIM_DATA::DataPtr_t<T>>, uint64_t> execute(
            const LLM_SIM_INST::VectorInst_t& inst,
            const LLM_SIM_DATA::DataPtr_t<T>& input,
            const LLM_SIM_DATA::DataPtr_t<T>& weight);
        // Compute latency only (no arithmetic) — for compute_=false transaction mode
        uint64_t latency_only(
            const LLM_SIM_INST::VectorInst_t& inst,
            const LLM_SIM_DATA::DataPtr_t<T>& input,
            const LLM_SIM_DATA::DataPtr_t<T>& weight) const;
        // Single-output element count execute() would produce for this
        // mode, mirroring each execute_xxx()'s own output vector size
        // exactly (see the "output(...)" size at each execute_xxx's
        // allocation). Covers every mode except the dual-output TOP_K
        // and ROPE, which VectorUnit.cpp's
        // transaction-only path computes inline since they need inst
        // fields (w_id/n_experts) this signature doesn't carry. Used by
        // both the compute path's assign_output_shape() (as a final
        // product-mismatch fallback) and the transaction-only path's
        // zero-fill size.
        static size_t output_elem_count(
            const LLM_SIM_INST::VectorInst_t& inst,
            const LLM_SIM_DATA::DataPtr_t<T>& input);
        // Sets output->size_ for every single-output mode, mirroring
        // execute()'s own shape-assignment switch exactly. Shared by
        // execute() (compute path) and VectorUnit.cpp's transaction-only
        // zero-fill path so both always agree on shape.
        static void assign_output_shape(
            const LLM_SIM_INST::VectorInst_t& inst,
            const LLM_SIM_DATA::DataPtr_t<T>& input,
            const LLM_SIM_DATA::DataPtr_t<T>& output);
    private:
        // --- Operation implementations (operate on raw vectors internally) ---
        // Element-wise arithmetic
        std::pair<std::vector<T>, uint64_t> execute_add(const std::vector<T>& input, const std::vector<T>& weight, const int dim);
        std::pair<std::vector<T>, uint64_t> execute_sub(const std::vector<T>& input, const std::vector<T>& weight, const int dim);
        std::pair<std::vector<T>, uint64_t> execute_mul(const std::vector<T>& input, const std::vector<T>& weight, const int dim);
        std::pair<std::vector<T>, uint64_t> execute_recip(const std::vector<T>& input, const int dim);
        std::pair<std::vector<T>, uint64_t> execute_inv_sqrt(const std::vector<T>& input, const int dim);
        // Activation functions
        std::pair<std::vector<T>, uint64_t> execute_relu(const std::vector<T>& input, const int dim);
        std::pair<std::vector<T>, uint64_t> execute_sig(const std::vector<T>& input, const int dim);
        std::pair<std::vector<T>, uint64_t> execute_silu(const std::vector<T>& input, const int dim);
        std::pair<std::vector<T>, uint64_t> execute_swiglu(const std::vector<T>& input, const std::vector<T>& weight, const int dim);
        // Normalisation / transform
        std::pair<std::vector<T>, uint64_t> execute_rms(const std::vector<T>& input, const std::vector<T>& weight, const int dim);
        // sink: optional per-row scalar (one per softmax batch row), added to the
        // softmax denominator as exp(sink-max), per gpt-oss-20b's real attn-sink
        // semantics (ggml_compute_forward_soft_max_f32). Empty = no sink (legacy).
        std::pair<std::vector<T>, uint64_t> execute_softmax(const std::vector<T>& input, const int dim, const std::vector<T>& sink = {});
        std::pair<std::vector<T>, uint64_t> execute_rope(const std::vector<T>& input, const int dim, const int token_pos, const float theta_base = 10000.0f);
        // Reduction
        std::pair<std::vector<T>, uint64_t> execute_add_reduce(const std::vector<T>& input, const std::vector<int>& shape, bool all, int axis);
        std::pair<std::vector<T>, uint64_t> execute_dot_product(const std::vector<T>& input, const std::vector<T>& weight, const std::vector<int>& shape, bool all, int axis);
        std::pair<std::vector<T>, uint64_t> execute_top_k(const std::vector<T>& input, const int dim);
        // Dual-output TOP_K: returns (sorted_indices_as_T, sorted_values_renorm, latency).
        // n_experts > 0: per-row top_k with renormalised weights (each row = n_experts scores)
        // n_experts == 0: global top_k (legacy/fallback behaviour)
        std::tuple<std::vector<T>, std::vector<T>, uint64_t> execute_top_k_dual(const std::vector<T>& input, const int dim, const int n_experts = 0);
        std::pair<std::vector<T>, uint64_t> execute_div_constant(const std::vector<T>& input, const int dim);
        // Broadcast bias add
        std::pair<std::vector<T>, uint64_t> execute_bias_add(const std::vector<T>& input, const std::vector<T>& bias, const int feature_dim);
        // Sparse MoE gate: topk_idx[chunk*top_k] + topk_val[chunk*top_k] -> gate[chunk*n_experts]
        std::pair<std::vector<T>, uint64_t> execute_sparse_gate(const std::vector<T>& idx_in, const std::vector<T>& val_in, const int n_experts, const int top_k);
        // Q8_0 quantize+dequantize round-trip (§3.2):
        // models activation-precision loss before MXFP4-weight expert matmuls.
        std::pair<std::vector<T>, uint64_t> execute_quantize_dequant_q8_0(const std::vector<T>& input, const int dim);
        std::pair<std::vector<T>, uint64_t> execute_reduce_max_abs(const std::vector<T>& input, const int dim);
        std::pair<std::vector<T>, uint64_t> execute_quantize_dequant_q8_0_ext_scale(const std::vector<T>& input, const std::vector<T>& ext_scale_vec, const int dim);
        std::pair<std::vector<T>, uint64_t> execute_quantize_dequant_q8_0_boundary_fix(const std::vector<T>& input, const std::vector<T>& ext_scale_vec, const int dim);
};


#endif