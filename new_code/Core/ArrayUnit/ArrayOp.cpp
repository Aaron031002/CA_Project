#include "Core/ArrayUnit/ArrayOp.h"
#include <cmath>
#include <limits>
#include <iostream>
#include <cstdlib>

// ─── execute ────────────────────────────────────────────────────────────────

template <typename T>
LLM_SIM_DATA::Data_t<T> ArrayOp<T>::execute(const LLM_SIM_INST::ArrayInstPtr_t& inst_ptr, const LLM_SIM_DATA::Data_t<T> &in, const LLM_SIM_DATA::Data_t<T> &weight) const{
    auto id = inst_ptr->out_id;
    switch (inst_ptr->mode) {
        case LLM_SIM_INST::ArrayInst_t::NORMAL:
            return NORMAL(inst_ptr->row, inst_ptr->col, in, weight, id);
        case LLM_SIM_INST::ArrayInst_t::DOT_PRODUCT:
            return DOT_PRODUCT(inst_ptr->row, inst_ptr->col, in, weight, id);
        case LLM_SIM_INST::ArrayInst_t::QK:
            return QK(inst_ptr->row, inst_ptr->col, inst_ptr->kv_head_num,
                       inst_ptr->q_seq_s, inst_ptr->k_seq_s_real, inst_ptr->win_size,
                       inst_ptr->has_sink, in, weight, id);
        case LLM_SIM_INST::ArrayInst_t::SV:
            return SV(inst_ptr->row, inst_ptr->col, inst_ptr->kv_head_num,
                       inst_ptr->has_sink, in, weight, id, inst_ptr->q_head_num);
        default:
            throw std::runtime_error("Unsupported Array Mode");
    }
}

// ─── NORMAL ─────────────────────────────────────────────────────────────────
// Standard batched GEMM: (..., M, K) * (..., K, N) -> (..., M, N)
// Convention: row = K (inner/shared dimension), col = N (output columns)
//   in     : [total_M * K]  where total_M = batch * M, total_M = in.value.size() / K
//   weight : [K * N]        stored row-major (weight[k][n] = weight[k*N + n])
//   output : [total_M * N]  stored row-major (out[m][n]   = out[m*N + n])

template <typename T>
LLM_SIM_DATA::Data_t<T> ArrayOp<T>::NORMAL(int row, int col, const LLM_SIM_DATA::Data_t<T> &in, const LLM_SIM_DATA::Data_t<T> &weight, uint64_t id) const {
    const int K = row, N = col;
    if (K <= 0 || N <= 0)
        throw std::runtime_error("ArrayOp::NORMAL: row and col must be >= 1");
    // total_M = batch (token) dimension.  in.value.size() may be slightly larger
    // than total_M*K when the input was zero-padded for Q8_0 alignment (e.g.
    // shard_dim=720 padded to 736; total_M=1 so we just need size >= K).
    if (in.value.empty() || static_cast<int>(in.value.size()) < K)
        throw std::runtime_error("ArrayOp::NORMAL: in.value.size() must be >= row");
    if (static_cast<int>(weight.value.size()) != K * N)
        throw std::runtime_error("ArrayOp::NORMAL: weight.value.size() must equal row * col");

    const int total_M = static_cast<int>(in.value.size()) / K;
    std::vector<T> acc(total_M * N, T{});
    for (int m = 0; m < total_M; ++m)
        for (int k = 0; k < K; ++k) {
            const T vik = in.value[m * K + k];
            const int wbase = k * N;
            for (int n = 0; n < N; ++n)
                acc[m * N + n] += vik * weight.value[wbase + n];
        }

    if (std::getenv("DEBUG_DUMP_NORM_STATS") && std::getenv("DEBUG_DUMP_NORM_STATS_HI") &&
        id >= static_cast<uint64_t>(std::atoll(std::getenv("DEBUG_DUMP_NORM_STATS"))) &&
        id <  static_cast<uint64_t>(std::atoll(std::getenv("DEBUG_DUMP_NORM_STATS_HI")))) {
        auto stats = [](const char* label, const std::vector<T>& v) {
            double vmin = std::numeric_limits<double>::infinity();
            double vmax = -std::numeric_limits<double>::infinity();
            int n_nan = 0, n_inf = 0;
            for (T x : v) {
                double d = static_cast<double>(x);
                if (std::isnan(d)) { ++n_nan; continue; }
                if (std::isinf(d)) { ++n_inf; continue; }
                if (d < vmin) vmin = d;
                if (d > vmax) vmax = d;
            }
            std::cerr << "[NORM-STATS] " << label << " size=" << v.size()
                       << " min=" << vmin << " max=" << vmax
                       << " n_nan=" << n_nan << " n_inf=" << n_inf << "\n";
        };
        std::cerr << "[NORM-STATS] out_id=" << id << " row(K)=" << K << " col(N)=" << N << "\n";
        stats("in", in.value);
        stats("weight", weight.value);
        stats("out", acc);
        std::cerr << std::flush;
    }

    LLM_SIM_DATA::Data_t<T> out;
    out.size_ = {total_M, N};
    out.value = std::move(acc);
    out.id = id;
    return out;
}

// ─── DOT_PRODUCT ────────────────────────────────────────────────────────────
// Batched dot-product with (non-batched) transposed weight:
//   (..., M, K) * (N, K) -> (..., M, N)
// Convention: row = K (inner dimension), col = N (number of weight rows)
//   in     : [total_M * K]  stored row-major (in[m][k]   = in[m*K + k])
//   weight : [N * K]        stored row-major (weight[n][k] = weight[n*K + k])
//   output : [total_M * N]  (out[m][n] = Σ_k in[m][k] * weight[n][k])

template <typename T>
LLM_SIM_DATA::Data_t<T> ArrayOp<T>::DOT_PRODUCT(int row, int col, const LLM_SIM_DATA::Data_t<T> &in, const LLM_SIM_DATA::Data_t<T> &weight, uint64_t id) const {
    const int K = row, N = col;
    if (K <= 0 || N <= 0)
        throw std::runtime_error("ArrayOp::DOT_PRODUCT: row and col must be >= 1");
    if (in.value.empty() || static_cast<int>(in.value.size()) % K != 0)
        throw std::runtime_error("ArrayOp::DOT_PRODUCT: in.value.size() must be a non-zero multiple of row");
    if (static_cast<int>(weight.value.size()) != N * K)
        throw std::runtime_error("ArrayOp::DOT_PRODUCT: weight.value.size() must equal col * row");

    const int total_M = static_cast<int>(in.value.size()) / K;
    std::vector<T> acc(total_M * N, T{});
    for (int m = 0; m < total_M; ++m)
        for (int n = 0; n < N; ++n) {
            T sum = T{};
            for (int k = 0; k < K; ++k)
                sum += in.value[m * K + k] * weight.value[n * K + k];
            acc[m * N + n] = sum;
        }

    LLM_SIM_DATA::Data_t<T> out;
    out.size_ = {total_M, N};
    out.value = std::move(acc);
    out.id = id;
    return out;
}

// ─── QK ─────────────────────────────────────────────────────────────────────
// Multi-head attention score: Q * K^T  (scale is NOT applied here).
//
// Instruction tensor-mode convention (matches DRAMAddressDecoder):
//   row  = seq_len          (number of cached K-token positions)
//   col  = kv_heads * HEAD_DIM  (total KV feature width from head range × HEAD_DIM)
//
// Q is taken from `in` whose shape must be {q_tokens, q_total_dim}:
//   q_total_dim = q_heads * head_dim
//   q_heads     = q_total_dim / head_dim
//   head_dim    = col / kv_heads
//
// K weight layout: [seq_len][kv_heads][head_dim]
//   K[t][kh][d] = weight[t * kv_heads * head_dim + kh * head_dim + d]
//
// Output layout: [q_tokens * q_heads][seq_len]
//   score[qt*q_heads + h][t] = Σ_d Q[qt][h][d] * K[t][kh][d]
//   where kh = h / group_size, group_size = q_heads / kv_heads
//
// GQA: kv_head_num_arg > 0 → kv_heads = kv_head_num_arg
//      kv_head_num_arg ≤ 0 → kv_heads = 1  (default / no explicit GQA arg)
// q_seq_s / k_seq_s_real / win_size: causal sliding-window mask
//   has_sink=true  (legacy): position 0 in K file = sink (always attended,
//                   exempt from mask); position t > 0 = global K position
//                   k_seq_s_real + t - 1.
//   has_sink=false (current): no sink row; every position t = global K
//                   position k_seq_s_real + t, mask applies to all rows.
//   attend if: k_global <= q_global AND q_global - k_global <= win_size

template <typename T>
LLM_SIM_DATA::Data_t<T> ArrayOp<T>::QK(int row, int col, int kv_head_num_arg,
    int q_seq_s, int k_seq_s_real, int win_size, bool has_sink,
    const LLM_SIM_DATA::Data_t<T> &in, const LLM_SIM_DATA::Data_t<T> &weight, uint64_t id) const {
    const int seq_len  = row;
    const int kv_heads = (kv_head_num_arg > 0) ? kv_head_num_arg : 1;
    if (seq_len <= 0 || col <= 0 || kv_heads <= 0)
        throw std::runtime_error("ArrayOp::QK: row, col, kv_heads must be >= 1");
    if (col % kv_heads != 0)
        throw std::runtime_error("ArrayOp::QK: col must be divisible by kv_heads");
    const int head_dim = col / kv_heads;

    // Infer Q layout from input shape.
    int q_tokens = 1, q_total_dim = static_cast<int>(in.value.size());
    if (in.size_.size() >= 2) {
        q_tokens    = in.size_[0];
        q_total_dim = in.size_[1];
    }
    if (q_total_dim <= 0 || q_total_dim % head_dim != 0)
        throw std::runtime_error("ArrayOp::QK: Q total_dim must be divisible by head_dim");
    const int q_heads    = q_total_dim / head_dim;
    if (q_heads % kv_heads != 0)
        throw std::runtime_error("ArrayOp::QK: q_heads must be divisible by kv_heads");
    const int group_size = q_heads / kv_heads;

    if (static_cast<int>(in.value.size()) != q_tokens * q_total_dim)
        throw std::runtime_error("ArrayOp::QK: in.value.size() inconsistent with inferred Q shape");
    if (static_cast<int>(weight.value.size()) != seq_len * kv_heads * head_dim)
        throw std::runtime_error("ArrayOp::QK: weight.value.size() must equal seq_len * kv_heads * head_dim");

    LLM_SIM_DATA::Data_t<T> out;
    // Output shape: [q_tokens * q_heads][seq_len] – flattened so that batched
    // SOFTMAX (dim=seq_len) operates on each (q_token, q_head) score row.
    const int n_score_rows = q_tokens * q_heads;
    out.size_ = {n_score_rows, seq_len};
    out.value.resize(n_score_rows * seq_len, T{});

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const bool apply_mask = (win_size > 0 && q_seq_s >= 0 && k_seq_s_real >= 0);
    const float NEG_INF = -std::numeric_limits<float>::max() / 2.0f;

    // TEMP DEBUG (0/24-numerically-match-oracle
    // investigation, 2026-08-10): dump QK's actual seq_len/window params +
    // weight(K) content summary, to verify full-mode's chained NoC K/V
    // accumulation actually delivered the FULL [0, q_seq_s+q_tokens) history
    // by the time this core's QK runs (checking for silent truncation/
    // stale-zero-fill at the far end of the 64-hop chain), not just that
    // gen-time's k_len computation LOOKS right on paper.
    if (std::getenv("DEBUG_DUMP_QK_PARAMS")) {
        float kmin = std::numeric_limits<float>::max(), kmax = -std::numeric_limits<float>::max();
        double ksum = 0.0;
        for (float v : weight.value) {
            float f = static_cast<float>(v);
            kmin = std::min(kmin, f); kmax = std::max(kmax, f); ksum += f;
        }
        std::cerr << "[QK-DEBUG] weight_id=" << weight.id << " seq_len(row)=" << seq_len
                  << " q_seq_s=" << q_seq_s << " k_seq_s_real=" << k_seq_s_real
                  << " win_size=" << win_size << " has_sink=" << has_sink
                  << " weight.value.size()=" << weight.value.size()
                  << " K_min=" << kmin << " K_max=" << kmax
                  << " K_mean=" << (ksum / std::max<size_t>(1, weight.value.size())) << "\n" << std::flush;
    }

    // TEMP DEBUG (parallel-run race-condition
    // investigation, 2026-08-10): dump the RAW Q input (in.value) and the
    // RAW K row 0 (weight.value[0..head_dim)) for a specific q_seq_s, to
    // pin down whether Q or K (or both) already differs from the isolated/
    // oracle-matching run BEFORE any matmul happens -- narrows "QK scores
    // differ under parallel load" down to which of the two inputs is wrong.
    if (std::getenv("DEBUG_DUMP_QK_INPUTS") && q_seq_s == std::atoi(std::getenv("DEBUG_DUMP_QK_INPUTS"))) {
        const int hd = col / kv_heads;
        std::cerr << "[QK-INPUTS] q_seq_s=" << q_seq_s << " Q(head0, first " << hd << "):";
        for (int d = 0; d < hd && d < static_cast<int>(in.value.size()); ++d)
            std::cerr << " " << static_cast<float>(in.value[d]);
        std::cerr << "\n";
        std::cerr << "[QK-INPUTS] q_seq_s=" << q_seq_s << " K(pos0, first " << hd << "):";
        for (int d = 0; d < hd && d < static_cast<int>(weight.value.size()); ++d)
            std::cerr << " " << static_cast<float>(weight.value[d]);
        std::cerr << "\n" << std::flush;
    }

    for (int qt = 0; qt < q_tokens; ++qt) {
        for (int h = 0; h < q_heads; ++h) {
            const int kh     = h / group_size;
            const int q_base = qt * q_total_dim + h * head_dim;
            const int o_base = (qt * q_heads + h) * seq_len;
            for (int t = 0; t < seq_len; ++t) {
                const int k_base = t * kv_heads * head_dim + kh * head_dim;
                float sum = 0.f;
                for (int d = 0; d < head_dim; ++d)
                    sum += static_cast<float>(in.value[q_base + d]) *
                           static_cast<float>(weight.value[k_base + d]);
                sum *= scale;
                // Causal sliding-window mask. has_sink: t==0 is the sink row,
                // always attended (exempt from mask); real K starts at t==1.
                if (apply_mask && (!has_sink || t > 0)) {
                    const int k_global = has_sink ? (k_seq_s_real + t - 1) : (k_seq_s_real + t);
                    const int q_global = q_seq_s + qt;
                    if (k_global > q_global || (q_global - k_global) > win_size)
                        sum = NEG_INF;
                }
                out.value[o_base + t] = static_cast<T>(sum);
            }
        }
    }
    out.id = id;

    // TEMP DEBUG (0/24-numerically-match-oracle
    // investigation, 2026-08-10): dump the raw score row for the LAST
    // query token this core computed (head 0 only), to compare against
    // oracle's own independently-computed QK scores at the same
    // (layer, token, head) -- narrows the divergence to "QK matmul itself"
    // vs "something after QK" (softmax/SV/etc).
    if (std::getenv("DEBUG_DUMP_QK_SCORES") && q_seq_s == std::atoi(std::getenv("DEBUG_DUMP_QK_SCORES"))) {
        const int qt = q_tokens - 1;
        const int h = 0;
        const int o_base = (qt * q_heads + h) * seq_len;
        std::cerr << "[QK-SCORES] q_seq_s=" << q_seq_s << " qt=" << qt << " head=" << h
                  << " seq_len=" << seq_len << " scores:";
        for (int t = 0; t < seq_len; ++t) {
            std::cerr << " " << static_cast<float>(out.value[o_base + t]);
        }
        std::cerr << "\n" << std::flush;
    }

    return out;
}

// ─── SV ─────────────────────────────────────────────────────────────────────
// Multi-head attention context vector: S * V.
//
// Instruction tensor-mode convention (same as QK):
//   row  = seq_len
//   col  = kv_heads * HEAD_DIM
//
// S (scores) input shape: [n_groups][seq_len]
//   n_groups = q_tokens * q_heads  (flattened, matching QK output layout)
//
// V weight layout: [seq_len][kv_heads][head_dim]
//   V[t][kh][d] = weight[t * kv_heads * head_dim + kh * head_dim + d]
//
// Output layout: [n_groups][head_dim]
//   ctx[g][d] = Σ_t S[g][t] * V[t][kh][d]
//   where kh = (g % q_heads) / group_size
//
// kv_head_num_arg ≤ 0 → kv_heads = 1 (default for missing GQA arg)

template <typename T>
LLM_SIM_DATA::Data_t<T> ArrayOp<T>::SV(int row, int col, int kv_head_num_arg,
    bool has_sink,
    const LLM_SIM_DATA::Data_t<T> &in, const LLM_SIM_DATA::Data_t<T> &weight, uint64_t id, int q_head_num_arg) const {
    const int seq_len  = row;  // number of real K tokens (NOT including sink)
    const int kv_heads = (kv_head_num_arg > 0) ? kv_head_num_arg : 1;
    if (seq_len <= 0 || col <= 0 || kv_heads <= 0)
        throw std::runtime_error("ArrayOp::SV: row, col, kv_heads must be >= 1");
    if (col % kv_heads != 0)
        throw std::runtime_error("ArrayOp::SV: col must be divisible by kv_heads");
    const int head_dim = col / kv_heads;

    // Score input has an extra leading column when has_sink=true (the sink column to skip)
    const int score_cols = has_sink ? seq_len + 1 : seq_len;
    const int score_offset = has_sink ? 1 : 0;  // skip column 0 when has_sink

    if (static_cast<int>(in.value.size()) % score_cols != 0)
        throw std::runtime_error("ArrayOp::SV: in.value.size() must be divisible by score_cols");
    const int n_groups = static_cast<int>(in.value.size()) / score_cols;

    if (static_cast<int>(weight.value.size()) != seq_len * kv_heads * head_dim)
        throw std::runtime_error("ArrayOp::SV: weight.value.size() must equal seq_len * kv_heads * head_dim");

    // Determine q_heads from score tensor shape (if available) to map groups to kv heads.
    int q_heads = n_groups;  // fallback: treat each group as its own q-head
    if (in.size_.size() >= 2) {
        // in.size_ = [n_groups, seq_len] – q_heads not directly available here.
        // We infer: q_heads per kv_head = ceil(n_groups / kv_heads)
    }
    // With kv_heads=1, all groups map to kh=0 — no further inference needed.

    LLM_SIM_DATA::Data_t<T> out;
    out.size_ = {n_groups, head_dim};
    out.value.assign(n_groups * head_dim, T{});

    for (int g = 0; g < n_groups; ++g) {
        // Map group to kv head.
        // If q_head_num_arg > 0: multi-token GQA case — group g = qt * q_heads + h,
        //   so kh = (g % q_heads) / (q_heads / kv_heads)
        // Otherwise (legacy / decode): kh = g * kv_heads / n_groups  (correct when q_tokens=1)
        int kh = 0;
        if (kv_heads > 1) {
            if (q_head_num_arg > 0) {
                const int q_heads_per_tok = q_head_num_arg;
                kh = (g % q_heads_per_tok) * kv_heads / q_heads_per_tok;
            } else {
                kh = g * kv_heads / n_groups;
            }
        }
        const int s_base = g * score_cols;
        const int o_base = g * head_dim;
        for (int t = 0; t < seq_len; ++t) {
            const T st       = in.value[s_base + score_offset + t];
            const int v_base = t * kv_heads * head_dim + kh * head_dim;
            for (int d = 0; d < head_dim; ++d)
                out.value[o_base + d] += st * weight.value[v_base + d];
        }
    }
    out.id = id;

    // TEMP DEBUG (0/24-numerically-match-oracle
    // investigation, 2026-08-10): dump SV's output for head 0 (group 0,
    // when q_tokens=1), to compare against oracle's own SV computation at
    // the same (layer, token, head) -- narrows the divergence further down
    // the pipeline (QK/softmax already independently verified correct).
    if (std::getenv("DEBUG_DUMP_SV") && seq_len == std::atoi(std::getenv("DEBUG_DUMP_SV"))) {
        std::cerr << "[SV-DEBUG] seq_len=" << seq_len << " head_dim=" << head_dim
                  << " n_groups=" << n_groups << " group0_out:";
        for (int d = 0; d < head_dim; ++d) std::cerr << " " << static_cast<float>(out.value[d]);
        std::cerr << "\n" << std::flush;
    }

    return out;
}


template class ArrayOp<float>;
// template class ArrayOp<int>;
