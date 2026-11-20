#include "Core/MainCtrl/Decoder/VectorDec/VectorDec.h"
#include "utils/BaseInst.h"
#include "Core/MainCtrl/Decoder/DecodeError.h"
#include <random>
#include <sstream>
#include <string>

using namespace LLM_SIM_INST;

// Text format (general modes):
//   Vector <mode> <dim> <x_id> <w_id> <out_id> <x_reserved> <w_reserved> [all] [out_id2]
//
// ROPE format (no weight):
//   Vector ROPE <head_dim> <token_pos> <x_id> <out_id> <x_reserved>
//
// mode   : ADD | SUB | MUL | RECIP | INV_SQRT | RELU | SIG | SILU | SWIGLU |
//          RMS | SOFTMAX | ROPE | ADD_REDUCE | DOT_PRODUCT | TOP_K | DIV_CONSTANT | BIAS_ADD |
//          SPARSE_GATE | QUANTIZE_DEQUANT_Q8_0 (+ REDUCE_MAX_ABS and the two
//          QUANTIZE_DEQUANT_Q8_0_* variants). An unknown mode, a missing or
//          non-numeric required field, or a leftover token throws DecodeError.
// all    : optional 0/1 flag for ADD_REDUCE / DOT_PRODUCT (default 0)
//          0 = reduce along axis <dim>; 1 = reduce across all elements
// out_id2: optional uint64_t, TOP_K only (default 0).
//          TOP_K emits two DataPtr_t: out_id=sorted indices, out_id2=sorted values.
// rename : (optional, must come last) "rename <new_x_id> <new_w_id>" for the
//          general format, "rename <new_x_id>" for ROPE (no weight input) --
//          rename-on-terminal-read (data-operation-inst-array-vector-tmu
//          plan). 0 = no rename for that slot; only meaningful together
//          with x_reserved/w_reserved==1 on the same slot. Omitted → both
//          default to 0 (today's behavior, unchanged instruction text is
//          unaffected).

static std::mt19937& vecRng() {
    static std::mt19937 rng(43);
    return rng;
}

BaseInstPtr_t VectorDec::decode() {
    auto& rng = vecRng();
    // Skip IDLE (index 0)
    auto mode = static_cast<VectorInst_t::VectorMode_t>(
        static_cast<int>(rng() % (VectorInst_t::VectorMode_t::NUM_OPS - 1)) + 1);
    int dim = static_cast<int>(rng() % 512) + 64;
    return std::make_shared<VectorInst_t>(mode, dim,
                                          /*in_id=*/0UL, /*w_id=*/0UL, /*out_id=*/0UL);
}

BaseInstPtr_t VectorDec::decode(const std::string& inst_str) {
    std::istringstream iss(inst_str);
    std::string opcode, mode_str;
    iss >> opcode >> mode_str;
    const std::string what = "Vector " + mode_str;

    // ── ROPE: Vector ROPE <head_dim> <token_pos> <x_id> <out_id> <x_reserved> [theta] ──
    if (mode_str == "ROPE") {
        int      head_dim = 64;
        int      token_pos = 0;
        uint64_t x_id = 0, out_id = 0;
        int      x_reserved_int = 0;
        float    theta = 10000.0f;
        if (!(iss >> head_dim >> token_pos >> x_id >> out_id >> x_reserved_int)) {
            throw DecodeError(what + ": expected <head_dim> <token_pos> <x_id> <out_id> <x_reserved>");
        }
        iss >> theta;  // optional; leaves theta=10000.0f if not present

        uint64_t new_x_id = 0;
        {
            iss.clear(); // theta may have left the stream in fail state
            std::string rename_kw;
            if (iss >> rename_kw) {
                if (rename_kw != "rename") {
                    throw DecodeError(what + ": unexpected token '" + rename_kw + "'");
                }
                if (!(iss >> new_x_id)) {
                    throw DecodeError(what + ": 'rename' must be followed by <new_x_id>");
                }
            }
        }
        expect_no_more_tokens(iss, what);
        return std::make_shared<VectorInst_t>(
            VectorInst_t::VectorMode_t::ROPE, head_dim, token_pos,
            x_id, out_id, (x_reserved_int != 0), theta, new_x_id);
    }

    // ── General modes ─────────────────────────────────────────────────────────
    VectorInst_t::VectorMode_t mode = VectorInst_t::VectorMode_t::RMS;
    if      (mode_str == "ADD")                                    mode = VectorInst_t::VectorMode_t::ADD;
    else if (mode_str == "SUB")                                    mode = VectorInst_t::VectorMode_t::SUB;
    else if (mode_str == "MUL" || mode_str == "mul"
                                || mode_str == "mult")             mode = VectorInst_t::VectorMode_t::MUL;
    else if (mode_str == "RECIP")                                  mode = VectorInst_t::VectorMode_t::RECIP;
    else if (mode_str == "INV_SQRT")                               mode = VectorInst_t::VectorMode_t::INV_SQRT;
    else if (mode_str == "RELU")                                   mode = VectorInst_t::VectorMode_t::RELU;
    else if (mode_str == "SIG")                                    mode = VectorInst_t::VectorMode_t::SIG;
    else if (mode_str == "SILU")                                   mode = VectorInst_t::VectorMode_t::SILU;
    else if (mode_str == "SWIGLU")                                 mode = VectorInst_t::VectorMode_t::SWIGLU;
    else if (mode_str == "RMS"     || mode_str == "rmsnorm")       mode = VectorInst_t::VectorMode_t::RMS;
    else if (mode_str == "SOFTMAX" || mode_str == "softmax")       mode = VectorInst_t::VectorMode_t::SOFTMAX;
    // ROPE is handled above (early return) and never reaches here.
    else if (mode_str == "ADD_REDUCE")                             mode = VectorInst_t::VectorMode_t::ADD_REDUCE;
    else if (mode_str == "DOT_PRODUCT")                            mode = VectorInst_t::VectorMode_t::DOT_PRODUCT;
    else if (mode_str == "TOP_K")                                  mode = VectorInst_t::VectorMode_t::TOP_K;
    else if (mode_str == "DIV_CONSTANT")                           mode = VectorInst_t::VectorMode_t::DIV_CONSTANT;
    else if (mode_str == "BIAS_ADD")                               mode = VectorInst_t::VectorMode_t::BIAS_ADD;
    else if (mode_str == "SPARSE_GATE")                            mode = VectorInst_t::VectorMode_t::SPARSE_GATE;
    else if (mode_str == "QUANTIZE_DEQUANT_Q8_0")                  mode = VectorInst_t::VectorMode_t::QUANTIZE_DEQUANT_Q8_0;
    else if (mode_str == "REDUCE_MAX_ABS")                         mode = VectorInst_t::VectorMode_t::REDUCE_MAX_ABS;
    else if (mode_str == "QUANTIZE_DEQUANT_Q8_0_EXT_SCALE")        mode = VectorInst_t::VectorMode_t::QUANTIZE_DEQUANT_Q8_0_EXT_SCALE;
    else if (mode_str == "QUANTIZE_DEQUANT_Q8_0_BOUNDARY_FIX")     mode = VectorInst_t::VectorMode_t::QUANTIZE_DEQUANT_Q8_0_BOUNDARY_FIX;
    else throw DecodeError("unknown Vector mode '" + mode_str + "' (see inst_spec.md for the list)");

    int      dim = 64;
    uint64_t x_id = 0, w_id = 0, out_id = 0;
    int      x_reserved_int = 0, w_reserved_int = 0, all_int = 0;

    if (!(iss >> dim >> x_id >> w_id >> out_id >> x_reserved_int >> w_reserved_int)) {
        throw DecodeError(what + ": expected <dim> <x_id> <w_id> <out_id> <x_reserved> <w_reserved>"
                          " (use 0 for an unused w_id/w_reserved)");
    }
    iss >> all_int;   // optional; leaves all_int=0 if not present

    // Optional out_id2 for TOP_K dual output (indices + values)
    uint64_t out_id2 = 0;
    iss >> out_id2;   // no-op if not present (leaves out_id2=0)

    // Optional "rename <new_x_id> <new_w_id>" — must come last.
    uint64_t new_x_id = 0, new_w_id = 0;
    {
        iss.clear(); // out_id2 (or an earlier field) may have left fail state
        std::string rename_kw;
        if (iss >> rename_kw) {
            if (rename_kw != "rename") {
                throw DecodeError(what + ": unexpected token '" + rename_kw + "'");
            }
            if (!(iss >> new_x_id >> new_w_id)) {
                throw DecodeError(what + ": 'rename' must be followed by <new_x_id> <new_w_id>");
            }
        }
    }
    expect_no_more_tokens(iss, what);

    bool x_reserved = (x_reserved_int != 0);
    bool w_reserved = (w_reserved_int != 0);
    bool all        = (all_int != 0);

    return std::make_shared<VectorInst_t>(mode, dim, x_id, w_id, out_id,
                                          x_reserved, w_reserved, all, out_id2,
                                          new_x_id, new_w_id);
}

