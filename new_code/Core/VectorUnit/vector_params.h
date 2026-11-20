#ifndef LLM_SIM_VECTOR_PARAMS_H
#define LLM_SIM_VECTOR_PARAMS_H

namespace LLM_SIM_VEC {

    // VectorUnit parameters
    const int NUM_UNIT = 4;
    const int BUFFER_SIZE = 512;
    constexpr int VEC_SIZE = NUM_UNIT * BUFFER_SIZE; // Vector size (number of elements processed in one instruction)
    constexpr int PROCESS_NUM = 32;
    constexpr int PROCESS_CHUNK = NUM_UNIT * PROCESS_NUM;
    // latency parameters
    // RMS_LATENCY/RMS_OVERHEAD (§B): RMS scans dim
    // elements twice (sum-of-squares, then normalise), so cost must scale with
    // dim like every other op below, not be a flat per-batch constant. The
    // pair (41, 49) is chosen so the previous flat cost (951) is reproduced
    // exactly at dim=D_MODEL=2880 (the only dim RMS ever actually runs at —
    // see ids_real.py WID_ATTN_NORM / WID_POST_ATTN_NORM):
    //   chunks = ceil(2880/PROCESS_CHUNK - 1) = 22;  22*41 + 49 == 951.
    const uint64_t RMS_LATENCY = 41;       // 每 128 elements 的延遲
    const uint64_t RMS_OVERHEAD = 49;      // 固定開銷
    const uint64_t ROPE_LATENCY = 10;      // 每 128 elements 的延遲
    const uint64_t ROPE_OVERHEAD = 14;     // 固定開銷
    const uint64_t SILU_LATENCY = 20;      // 每 128 elements 的延遲
    const uint64_t SILU_OVERHEAD = 23;     // 固定開銷
    const uint64_t SOFTMAX_LATENCY = 69;   // 每 128 elements 的延遲（另有固定開銷）
    const uint64_t SOFTMAX_OVERHEAD = 244; // 固定開銷
    const uint64_t ADD_LATENCY = 11; // 每 128 elements 的延遲（element-wise 加法）
    const uint64_t SUB_LATENCY = 11; // 每 128 elements 的延遲（element-wise 減法）
    const uint64_t MUL_LATENCY = 11; // 每 128 elements 的延遲（element-wise 乘法）
    const uint64_t ELEMENTWISE_OVERHEAD = 14; // element-wise 運算的固定開銷
    const uint64_t RECIP_LATENCY = 15;        // 每 128 elements 的延遲（reciprocal）
    const uint64_t INV_SQRT_LATENCY = 20;     // 每 128 elements 的延遲（inverse sqrt）
    const uint64_t RELU_LATENCY = 11;         // 每 128 elements 的延遲（ReLU）
    const uint64_t SIG_LATENCY = 20;          // 每 128 elements 的延遲（sigmoid）
    const uint64_t SWIGLU_LATENCY = 35;       // 每 128 elements 的延遲（SwiGLU）
    const uint64_t SWIGLU_OVERHEAD = 30;      // SwiGLU 固定開銷
    const uint64_t ADD_REDUCE_LATENCY = 20;   // 每 128 elements 的延遲（add-reduce）
    const uint64_t ADD_REDUCE_OVERHEAD = 50;  // add-reduce 固定開銷
    const uint64_t DOT_PRODUCT_LATENCY = 25;  // 每 128 elements 的延遲（dot product）
    const uint64_t DOT_PRODUCT_OVERHEAD = 50; // dot product 固定開銷
    const uint64_t TOP_K_LATENCY = 50;        // 每 128 elements 的延遲（top-k argsort）
    const uint64_t TOP_K_OVERHEAD = 100;      // top-k 固定開銷
    const uint64_t MAC = 980;                 // 固定開銷（如有需要可拆成
    // DIV_CONSTANT 參數
    constexpr float DIV_CONSTANT_VALUE = 8.0f; // 每個元素除以此常數
    const uint64_t DIV_CONSTANT_LATENCY = 15;  // 每 128 elements 的延遲（element-wise division）
    // BIAS_ADD 參數
    const uint64_t BIAS_ADD_LATENCY = 11;      // 每 128 elements 的延遲（broadcast bias add）

    // ── gpt-oss-20b real-architecture constants (§0b) ──
    // RoPE: YaRN scaling + NEOX-style pairing (verified against
    // tools/run_gpt_oss_transformer_block.py::apply_rope_yarn_neox, itself
    // verified against ggml's rope_yarn / ggml_rope_yarn_corr_dims).
    constexpr float ROPE_YARN_FREQ_SCALE  = 1.0f / 32.0f;
    constexpr int   ROPE_YARN_N_CTX_ORIG  = 4096;
    constexpr float ROPE_YARN_BETA_FAST   = 32.0f;
    constexpr float ROPE_YARN_BETA_SLOW   = 1.0f;
    constexpr float ROPE_YARN_EXT_FACTOR  = 1.0f;
    constexpr float ROPE_YARN_ATTN_FACTOR = 1.0f;

    // SwiGLU (OAI variant): gate clamped above only, up clamped both ways +1
    // offset, sigmoid slope alpha != 1 (verified against
    // ggml_compute_forward_swiglu_oai_f32 via run_gpt_oss_transformer_block.py::swiglu_oai).
    constexpr float SWIGLU_OAI_ALPHA = 1.702f;
    constexpr float SWIGLU_OAI_LIMIT = 7.0f;

    // Q8_0 quantize+dequantize round-trip (§3.2):
    // per-32-element-block scan (amax + requantize), modelled with the same
    // per-PROCESS_CHUNK-elements + fixed-overhead style as the other
    // elementwise/activation ops above.
    const uint64_t QUANTIZE_DEQUANT_Q8_0_LATENCY  = 20; // 每 128 elements 的延遲
    const uint64_t QUANTIZE_DEQUANT_Q8_0_OVERHEAD = 23; // 固定開銷
}


#endif