#include "Core/MainCtrl/Decoder/TMUDec/TMUDec.h"
#include "utils/BaseInst.h"
#include "Core/MainCtrl/Decoder/DecodeError.h"
#include <random>
#include <sstream>
#include <string>

using namespace LLM_SIM_INST;

// Text format (per New CBI Inst. SPEC):
//   TMU <mode> in <id1> <reserved1> [<id2> <reserved2> ...] out <out_id1> [...] size <size_list>
//
// mode      : concat | scatter | split | trans | view | pad | gather (anything
//             else, a malformed in/out/size list or a leftover token throws
//             DecodeError)
// in / out  : literal separators
// reserved  : 0 or 1 per input
// size_list :
//   - view mode               : target shape   e.g. "size 4 16"
//   - gather mode             : [dim]           e.g. "size 0"  (single axis; num_parts ignored)
//   - all other modes         : [dim] [num_parts]  e.g. "size 0 3"
//   - trans (no size needed)  : size section may be omitted
//
// Examples:
//   TMU concat   in 1 0 2 0  out 3          size 0 2
//   TMU scatter  in 4 0      out 5 6        size 0 2
//   TMU split    in 4 0      out 5 6 7      size 0 3
//   TMU trans    in 8 0      out 9
//   TMU view     in 10 0     out 11         size 4 16
//   TMU pad      in 12 0     out 13         size 1 4
//   TMU gather   in 10 0 11 0  out 12       size 0
//
// Optional trailing block, after size (rename-on-terminal-read, data-
// operation-inst-array-vector-tmu plan): "newids <input_index> <new_id>
// [<input_index> <new_id> ...]" — sparse, only inputs that rename appear.
// 0 = no rename for that input; only meaningful together with that input's
// own reserved==1. Omitted → no renames at all (today's behavior, unchanged
// instruction text is unaffected). Kept as a separate trailing block rather
// than widening each (id, reserved) pair to a triple so existing TMU lines
// (the overwhelming majority, with no rename) don't shift token position at
// all.
//   TMU concat   in 1 0 2 0  out 3          size 0 2  newids 1 100

static std::mt19937& tmuRng() {
    static std::mt19937 rng(45);
    return rng;
}

BaseInstPtr_t TMUDec::decode() {
    auto& rng = tmuRng();
    const int num_modes = static_cast<int>(TMUInst_t::TMUMode_t::NUM_MODES);
    auto mode = static_cast<TMUInst_t::TMUMode_t>(rng() % num_modes);

    // Decide input/output counts per mode.
    int num_inputs  = static_cast<int>(rng() % 3) + 1;
    int num_outputs;
    switch (mode) {
        case TMUInst_t::TMUMode_t::split:
            num_outputs = num_inputs;   // num_parts == num_inputs
            break;
        case TMUInst_t::TMUMode_t::scatter:
            num_inputs  = 1;
            num_outputs = static_cast<int>(rng() % 3) + 2;  // 2..4 parts
            break;
        case TMUInst_t::TMUMode_t::pad:
        case TMUInst_t::TMUMode_t::trans:
        case TMUInst_t::TMUMode_t::view:
            num_inputs  = 1;
            num_outputs = 1;
            break;
        case TMUInst_t::TMUMode_t::gather:
            num_inputs  = 2;  // source tensor + indices tensor
            num_outputs = 1;
            break;
        default: // concat
            num_outputs = 1;
            break;
    }

    std::vector<uint64_t> input_ids, output_ids;
    std::vector<bool>     input_reserved;

    for (int i = 0; i < num_inputs; ++i) {
        input_ids.push_back(static_cast<uint64_t>(rng() % 64));
        input_reserved.push_back(false);
    }
    for (int i = 0; i < num_outputs; ++i) {
        output_ids.push_back(static_cast<uint64_t>(rng() % 64));
    }

    if (mode == TMUInst_t::TMUMode_t::view) {
        std::vector<int> view_shape = { static_cast<int>(rng() % 8 + 1),
                                        static_cast<int>(rng() % 8 + 1) };
        return std::make_shared<TMUInst_t>(input_ids, input_reserved, output_ids,
                                           mode, view_shape);
    }
    int num_parts = num_outputs;  // for scatter/split; irrelevant for trans
    int dim       = static_cast<int>(rng() % 2);  // dim 0 or 1
    return std::make_shared<TMUInst_t>(input_ids, input_reserved, output_ids,
                                       mode, num_parts, dim);
}

BaseInstPtr_t TMUDec::decode(const std::string& inst_str) {
    std::istringstream iss(inst_str);
    std::string opcode, mode_str, sep;
    iss >> opcode >> mode_str >> sep;   // "TMU" "<mode>" "in"
    const std::string what = "TMU " + mode_str;

    // Map mode string to enum
    TMUInst_t::TMUMode_t mode = TMUInst_t::TMUMode_t::concat;
    if      (mode_str == "concat")  mode = TMUInst_t::TMUMode_t::concat;
    else if (mode_str == "scatter") mode = TMUInst_t::TMUMode_t::scatter;
    else if (mode_str == "split")   mode = TMUInst_t::TMUMode_t::split;
    else if (mode_str == "trans")   mode = TMUInst_t::TMUMode_t::trans;
    else if (mode_str == "view")    mode = TMUInst_t::TMUMode_t::view;
    else if (mode_str == "pad")     mode = TMUInst_t::TMUMode_t::pad;
    else if (mode_str == "gather")  mode = TMUInst_t::TMUMode_t::gather;
    else throw DecodeError("unknown TMU mode '" + mode_str +
                           "' (expected concat, scatter, split, trans, view, pad or gather)");
    if (sep != "in") {
        throw DecodeError(what + ": expected 'in' after the mode, got '" + sep + "'");
    }

    // A token of an id/size list must be a plain non-negative integer.
    auto parse_id = [&what](const std::string& tok, const char* field) -> uint64_t {
        if (tok.empty() || tok.find_first_not_of("0123456789") != std::string::npos) {
            throw DecodeError(what + ": expected an integer " + field + ", got '" + tok + "'");
        }
        return std::stoull(tok);
    };

    // Parse (id, reserved) pairs until the "out" separator token.
    std::vector<uint64_t> input_ids;
    std::vector<bool>     input_reserved;
    std::string token;
    bool saw_out = false;
    while (iss >> token) {
        if (token == "out") { saw_out = true; break; }
        uint64_t id = parse_id(token, "input id");
        int reserved_int = 0;
        if (!(iss >> reserved_int)) {
            throw DecodeError(what + ": input id " + token + " has no <reserved> flag after it");
        }
        input_ids.push_back(id);
        input_reserved.push_back(reserved_int != 0);
    }
    if (!saw_out) {
        throw DecodeError(what + ": missing 'out' separator");
    }
    if (input_ids.empty()) {
        throw DecodeError(what + ": expected at least one <id> <reserved> pair after 'in'");
    }

    // Parse output ids until "size" keyword or end of stream.
    // `kw` is the keyword that ended the list ("size"/"newids"), or "" at the
    // end of the line (`token` itself keeps its last value at end of stream).
    std::vector<uint64_t> output_ids;
    std::string kw;
    while (iss >> token) {
        if (token == "size" || token == "newids") { kw = token; break; }
        output_ids.push_back(parse_id(token, "output id"));
    }
    if (output_ids.empty()) {
        throw DecodeError(what + ": expected at least one output id after 'out'");
    }

    // Parse size section.
    // view  → view_shape (all tokens)
    // others → [dim] [num_parts]
    std::vector<int> size_list;
    if (kw == "size") {
        int s;
        while (iss >> s) size_list.push_back(s);
        iss.clear(); // the size-list loop leaves the stream in fail state on
                     // its first non-numeric token
        kw.clear();
        iss >> kw;
    }

    // Optional "newids <input_index> <new_id> [...]" trailing block -- see
    // format comment above. Stays empty (no renames at all) unless present.
    std::vector<uint64_t> new_input_ids;
    if (kw == "newids") {
        new_input_ids.assign(input_ids.size(), 0);
        uint64_t idx, new_id;
        while (iss >> idx >> new_id) {
            if (idx >= new_input_ids.size()) {
                throw DecodeError(what + ": newids input index " + std::to_string(idx) +
                                  " is out of range (" + std::to_string(input_ids.size()) +
                                  " inputs)");
            }
            new_input_ids[idx] = new_id;
        }
    } else if (!kw.empty()) {
        throw DecodeError(what + ": unexpected token '" + kw + "'");
    }
    expect_no_more_tokens(iss, what);

    if (mode == TMUInst_t::TMUMode_t::view) {
        // For view: size_list is the full target shape.
        return std::make_shared<TMUInst_t>(input_ids, input_reserved, output_ids,
                                           mode, size_list, new_input_ids);
    }

    if (mode == TMUInst_t::TMUMode_t::gather) {
        // gather: size_list has exactly one entry — the axis to gather along.
        // num_parts is unused for gather; pass 0.
        int dim = size_list.size() >= 1 ? size_list[0] : 0;
        return std::make_shared<TMUInst_t>(input_ids, input_reserved, output_ids,
                                           mode, /*num_parts=*/0, dim, new_input_ids);
    }

    // For concat / scatter / split / trans / pad:
    //   size [dim] [num_parts]  — dim defaults to -1, num_parts defaults to output count.
    int dim       = size_list.size() >= 1 ? size_list[0] : -1;
    int num_parts = size_list.size() >= 2 ? size_list[1]
                                          : static_cast<int>(output_ids.size());
    return std::make_shared<TMUInst_t>(input_ids, input_reserved, output_ids,
                                       mode, num_parts, dim, new_input_ids);
}
