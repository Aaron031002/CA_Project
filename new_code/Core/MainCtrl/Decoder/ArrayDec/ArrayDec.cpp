#include "Core/MainCtrl/Decoder/ArrayDec/ArrayDec.h"
#include "utils/BaseInst.h"
#include "utils/Req.h"
#include "DRAM_sim/dram_params.h"
#include "Core/MainCtrl/Decoder/DecodeError.h"
#include <random>
#include <sstream>
#include <string>

using namespace LLM_SIM_INST;
using namespace LLM_SIM_DRAM;

// Text formats:
//   Array <mode> <layer_name> index <row_start> <row_end> <col_start> <col_end>
//         <x_id> <out_id> <x_reserved> [kv_head_num] [df WS|OS]
//   Array <mode> <layer_name> tensor <batch> <seq_start> <seq_end>
//         <head_start> <head_end> <x_id> <out_id> <x_reserved> [kv_head_num] [df WS|OS]
//
// mode       : NORMAL | DOT_PRODUCT | QK | SV (anything else throws DecodeError,
//              as does a missing/non-numeric required field or a leftover token)
// index_mode : "index" (flat row/col) or "tensor" (KV-cache batch/seq/head coords)
// x_id       : cache data id for the input activation
// out_id     : cache data id for the output
// x_reserved : 0 or 1 — whether to evict the input from cache after use
// kv_head_num: (optional, default -1) GQA kv-head count; -1 = standard MHA
// df         : (optional) systolic array dataflow override — WS (weight-
//              stationary) or OS (output-stationary); default AUTO infers
//              from mode (see ArrayInst_t::resolved_dataflow())
// wsrc       : (optional, must come after [df ...] if both present)
//              "wsrc cache <w_id> <w_reserved>" — fetch the weight tile from
//              the shared Cache (id=w_id, evict-after-read=w_reserved)
//              instead of DRAM. Omitted → weight_src=DRAM (today's behavior,
//              still using the row/col/layer_name DRAM address above).
// rename     : (optional, must come after [wsrc ...] if present) "rename
//              <new_in_id> <new_w_id>" — rename-on-terminal-read (data-
//              operation-inst-array-vector-tmu plan): 0 = no rename for that
//              slot. Only meaningful together with x_reserved/w_reserved==1
//              on the same slot. Omitted → both default to 0 (today's
//              behavior, unchanged instruction text is unaffected).

static std::mt19937& arrayRng() {
    static std::mt19937 rng(42);
    return rng;
}

BaseInstPtr_t ArrayDec::decode() {
    auto& rng = arrayRng();
    auto mode = static_cast<ArrayInst_t::ArrayMode_t>(
        rng() % static_cast<int>(ArrayInst_t::ArrayMode_t::NUM_MODES));
    uint64_t row_start = rng() % 8;
    uint64_t row_end   = row_start + (rng() % 8) + 1;
    uint64_t col_start = rng() % 8;
    uint64_t col_end   = col_start + (rng() % 8) + 1;
    int  row = static_cast<int>(row_end - row_start);
    int  col = static_cast<int>(col_end - col_start);
    auto dram_addr = std::make_shared<DRAMAddr_t>(
        "layer.bin", row_start, row_end, col_start, col_end);
    auto req = std::make_shared<Req_t>(dram_addr, /*is_read=*/true);
    return std::make_shared<ArrayInst_t>(mode, row, col, req,
                                         /*in_id=*/0UL, /*out_id=*/0UL);
}

BaseInstPtr_t ArrayDec::decode(const std::string& inst_str) {
    std::istringstream iss(inst_str);
    std::string opcode, mode_str, layer_name, index_mode;
    uint64_t x_id = 0, out_id = 0;
    int x_reserved_int = 0;

    iss >> opcode >> mode_str >> layer_name >> index_mode;
    const std::string what = "Array " + mode_str;
    if (mode_str != "NORMAL" && mode_str != "DOT_PRODUCT" && mode_str != "QK" && mode_str != "SV") {
        throw DecodeError("unknown Array mode '" + mode_str +
                          "' (expected NORMAL, DOT_PRODUCT, QK or SV)");
    }
    if (index_mode != "index" && index_mode != "tensor") {
        throw DecodeError(what + ": expected <layer_name> followed by 'index' or 'tensor', got '" +
                          layer_name + " " + index_mode + "'");
    }

    int64_t row_start = 0, row_end = 1, col_start = 0, col_end = 1;
    if (index_mode == "tensor") {
        int batch = 0, seq_s = 0, seq_e = 1, head_s = 0, head_e = 1;
        if (!(iss >> batch >> seq_s >> seq_e >> head_s >> head_e)) {
            throw DecodeError(what + ": expected 5 integers after 'tensor' "
                              "(<batch> <seq_start> <seq_end> <head_start> <head_end>)");
        }
        row_start = static_cast<int64_t>(batch) * MAX_SEQ_LEN + seq_s;
        row_end   = static_cast<int64_t>(batch) * MAX_SEQ_LEN + seq_e;
        col_start = static_cast<int64_t>(head_s) * HEAD_DIM;
        col_end   = static_cast<int64_t>(head_e) * HEAD_DIM;
    } else {
        // "index" mode — direct row/col indices
        if (!(iss >> row_start >> row_end >> col_start >> col_end)) {
            throw DecodeError(what + ": expected 4 integers after 'index' "
                              "(<row_start> <row_end> <col_start> <col_end>)");
        }
    }

    // Optional "stride N" keyword — sets the actual row width of the file
    // (needed when col_end < file row width, e.g. partial-column TP shard reads).
    uint64_t row_stride_hint = 0; // 0 → default to col_end inside DRAMAddr_t
    {
        auto saved_pos = iss.tellg();
        std::string stride_kw;
        if ((iss >> stride_kw) && stride_kw == "stride") {
            if (!(iss >> row_stride_hint)) {
                throw DecodeError(what + ": 'stride' must be followed by an integer");
            }
        } else {
            iss.seekg(saved_pos);
            iss.clear();
        }
    }

    if (!(iss >> x_id >> out_id >> x_reserved_int)) {
        throw DecodeError(what + ": expected <x_id> <out_id> <x_reserved> after the coordinates"
                          " (and the optional 'stride N')");
    }

    // Optional kv_head_num for GQA (QK / SV modes); -1 = standard MHA
    int kv_head_num = -1;
    iss >> kv_head_num;   // no-op if not present (leaves kv_head_num = -1)
    iss.clear();          // reset failbit (e.g. NORMAL has no kv_head_num field —
                           // the next token may be "df", which fails int parsing)

    // Optional causal-mask params for QK: q_seq_s k_seq_s_real win_size
    int q_seq_s = -1, k_seq_s_real = -1, win_size = -1;
    if (mode_str == "QK") {
        int tmp1 = -1, tmp2 = -1, tmp3 = -1;
        iss >> tmp1;
        if (tmp1 >= 0) {
            iss >> tmp2 >> tmp3;
            q_seq_s      = tmp1;
            k_seq_s_real = tmp2;
            win_size     = tmp3;
        }
        iss.clear();       // reset failbit if tmp1 read hit "df" instead of a mask value
    }

    // Optional has_sink flag for SV: 0/1 (whether score col 0 is the sink)
    bool has_sink = false;
    if (mode_str == "SV") {
        int has_sink_int = 0;
        iss >> has_sink_int;
        has_sink = (has_sink_int != 0);
        iss.clear();       // reset failbit if this hit "df" instead of 0/1
    }

    // Optional q_head_num for SV: total Q heads per token for multi-token GQA kh mapping
    // Only present when q_head_num > 0 was written by the instruction generator
    int q_head_num = -1;
    if (mode_str == "SV") {
        int tmp = -1;
        if (iss >> tmp && tmp > 0) q_head_num = tmp;
        iss.clear();       // reset failbit if tmp read hit "df" instead of a number
    }

    // Optional "df WS|OS" keyword — overrides the systolic array dataflow
    // (default AUTO infers WEIGHT_STATIONARY for NORMAL/DOT_PRODUCT,
    // OUTPUT_STATIONARY for QK/SV; see ArrayInst_t::resolved_dataflow()).
    ArrayInst_t::Dataflow_t dataflow = ArrayInst_t::Dataflow_t::AUTO;
    {
        auto saved_pos = iss.tellg();
        std::string df_kw, df_val;
        if ((iss >> df_kw) && df_kw == "df") {
            iss >> df_val;
            if (df_val == "WS")      dataflow = ArrayInst_t::Dataflow_t::WEIGHT_STATIONARY;
            else if (df_val == "OS") dataflow = ArrayInst_t::Dataflow_t::OUTPUT_STATIONARY;
            else throw DecodeError(what + ": 'df' must be followed by WS or OS, got '" + df_val + "'");
        } else {
            iss.seekg(saved_pos);
            iss.clear();
        }
    }

    // Optional "wsrc cache <w_id> <w_reserved>" keyword — routes the weight
    // fetch through Cache instead of DRAM. Must come after the "df" block
    // above (fixed tail order). Omitted → weight_src=DRAM.
    ArrayInst_t::WeightSrc_t weight_src = ArrayInst_t::WeightSrc_t::DRAM;
    uint64_t w_id = 0;
    bool w_reserved = false;
    {
        auto saved_pos = iss.tellg();
        std::string wsrc_kw, wsrc_val;
        if ((iss >> wsrc_kw) && wsrc_kw == "wsrc") {
            iss >> wsrc_val;
            if (wsrc_val != "cache") {
                throw DecodeError(what + ": 'wsrc' must be followed by 'cache', got '" + wsrc_val + "'");
            }
            int w_reserved_int = 0;
            if (!(iss >> w_id >> w_reserved_int)) {
                throw DecodeError(what + ": 'wsrc cache' must be followed by <w_id> <w_reserved>");
            }
            weight_src = ArrayInst_t::WeightSrc_t::CACHE;
            w_reserved = (w_reserved_int != 0);
        } else {
            iss.seekg(saved_pos);
            iss.clear();
        }
    }

    // Optional "rename <new_in_id> <new_w_id>" keyword — rename-on-terminal-
    // read (data-operation-inst-array-vector-tmu plan). Must come after
    // [wsrc ...] (fixed tail order, same idiom as df/wsrc above).
    uint64_t new_in_id = 0, new_w_id = 0;
    {
        auto saved_pos = iss.tellg();
        std::string rename_kw;
        if ((iss >> rename_kw) && rename_kw == "rename") {
            if (!(iss >> new_in_id >> new_w_id)) {
                throw DecodeError(what + ": 'rename' must be followed by <new_in_id> <new_w_id>");
            }
        } else {
            iss.seekg(saved_pos);
            iss.clear();
        }
    }
    expect_no_more_tokens(iss, what);

    bool x_reserved = (x_reserved_int != 0);
    int  row = static_cast<int>((row_end > row_start) ? (row_end - row_start) : 0);
    int  col = static_cast<int>((col_end > col_start) ? (col_end - col_start) : 0);

    ArrayInst_t::ArrayMode_t mode = ArrayInst_t::ArrayMode_t::NORMAL;
    if      (mode_str == "DOT_PRODUCT") mode = ArrayInst_t::ArrayMode_t::DOT_PRODUCT;
    else if (mode_str == "QK")          mode = ArrayInst_t::ArrayMode_t::QK;
    else if (mode_str == "SV")          mode = ArrayInst_t::ArrayMode_t::SV;

    auto dram_addr = std::make_shared<DRAMAddr_t>(
        layer_name, row_start, row_end, col_start, col_end, row_stride_hint);
    auto req = std::make_shared<Req_t>(dram_addr, /*is_read=*/true);

    return std::make_shared<ArrayInst_t>(mode, row, col, req, x_id, out_id, x_reserved,
                                         kv_head_num, q_seq_s, k_seq_s_real, win_size,
                                         has_sink, q_head_num, dataflow,
                                         weight_src, w_id, w_reserved,
                                         new_in_id, new_w_id);
}

