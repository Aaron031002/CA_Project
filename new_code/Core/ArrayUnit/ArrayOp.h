#ifndef LLM_SIM_ARRAY_OP_H
#define LLM_SIM_ARRAY_OP_H

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Core/ArrayUnit/array_params.h"
#include "utils/BaseInst.h"
#include "utils/Data.h"


template <typename T>
class ArrayOp {
    public:
        ArrayOp() = default;
        LLM_SIM_DATA::Data_t<T> execute(const LLM_SIM_INST::ArrayInstPtr_t& inst_ptr, const LLM_SIM_DATA::Data_t<T> &in, const LLM_SIM_DATA::Data_t<T> &weight) const;
    private:
        // NORMAL: (..., M, K) x (..., K, N) -> (..., M, N)
        // row=K (inner dim), col=N, total_M derived from in.value.size()/K
        LLM_SIM_DATA::Data_t<T> NORMAL(int row, int col, const LLM_SIM_DATA::Data_t<T> &in, const LLM_SIM_DATA::Data_t<T> &weight, uint64_t id) const;
        // DOT_PRODUCT: (..., M, K) x (N, K) -> (..., M, N)  [weight is transposed]
        // row=K (inner dim), col=N, weight stored as [N, K]
        LLM_SIM_DATA::Data_t<T> DOT_PRODUCT(int row, int col, const LLM_SIM_DATA::Data_t<T> &in, const LLM_SIM_DATA::Data_t<T> &weight, uint64_t id) const;
        // QK: multi-head Q*K attention scores with optional scale and causal mask
        // Non-GQA (kv_head_num == -1): row=q_head_num*HEAD_DIM, col=seq_len
        //   in(Q): [row], weight(K): [col*row], out: [q_head_num*col]
        // GQA (kv_head_num > 0): row=q_head_num*HEAD_DIM, col=seq_len
        //   in(Q): [row], weight(K): [col*kv_head_num*HEAD_DIM], out: [q_head_num*col]
        // q_seq_s/k_seq_s_real/win_size: causal sliding-window mask (-1 = disabled)
        // has_sink: legacy mode — if true, K row 0 is a sink key, always attended
        // (exempt from causal/window mask), and k_seq_s_real refers to row 1.
        // false (current real-arch convention): no sink row in the K cache at all;
        // attn-sink is instead applied as a softmax-denominator bias (see VectorOp::execute_softmax).
        LLM_SIM_DATA::Data_t<T> QK(int row, int col, int kv_head_num, int q_seq_s, int k_seq_s_real, int win_size, bool has_sink, const LLM_SIM_DATA::Data_t<T> &in, const LLM_SIM_DATA::Data_t<T> &weight, uint64_t id) const;
        // SV: multi-head softmax-score * V context vector
        // Non-GQA: row=q_head_num*HEAD_DIM, col=seq_len
        //   in(S): [q_head_num*col], weight(V): [col*row], out: [row]
        // GQA (kv_head_num > 0):
        //   in(S): [q_head_num*col], weight(V): [col*kv_head_num*HEAD_DIM], out: [row]
        // has_sink: if true, score column 0 is the sink and is skipped (V has seq_len rows)
        LLM_SIM_DATA::Data_t<T> SV(int row, int col, int kv_head_num, bool has_sink, const LLM_SIM_DATA::Data_t<T> &in, const LLM_SIM_DATA::Data_t<T> &weight, uint64_t id, int q_head_num_arg = -1) const;
};





#endif