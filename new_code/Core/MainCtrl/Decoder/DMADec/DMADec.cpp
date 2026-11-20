#include "Core/MainCtrl/Decoder/DMADec/DMADec.h"
#include "utils/BaseInst.h"
#include "utils/Req.h"
#include "NoC_sim/noc_utils.h"
#include "DRAM_sim/dram_params.h"
#include "Core/MainCtrl/Decoder/DecodeError.h"
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace LLM_SIM_INST;
using namespace LLM_SIM_DRAM;

// Text formats:
//   DMA NoC send <packet_id> <data_id> <dest_id1> [<dest_id2> ...] [truncate] [rename <new_id>]
//   DMA NoC recv <packet_id> <data_id> <size1>    [<size2> ...]
//   DMA DRAM read  <file> index  <row_s> <row_e> <col_s> <col_e> <data_id> [pad <n>]
//   DMA DRAM write <file> index  <row_s> <row_e> <col_s> <col_e> <data_id> <reserved> [rename <new_id>]
//   DMA DRAM read  <file> tensor <batch> <seq_s> <seq_e> <head_s> <head_e> <data_id> [pad <n>]
//   DMA DRAM write <file> tensor <batch> <seq_s> <seq_e> <head_s> <head_e> <data_id> <reserved> [rename <new_id>]
//   DMA Cache copy <src_data_id> <dst_data_id>
//
// Any other unit (Cache/NoC/DRAM), sub-mode, a missing or non-numeric
// required field, or a leftover token throws DecodeError.
//
// rename <new_id>: rename-on-terminal-read (data-operation-inst-array-
// vector-tmu plan) — 0/omitted = no rename (today's behavior, unchanged
// instruction text is unaffected). Only meaningful together with
// truncate==1 (NoC send) / reserved==1 (DRAM write). Not supported on
// Cache copy/mc or NoC recv/DRAM read (those don't consume an existing
// resident id the same way).

static std::mt19937& dmaRng() {
    static std::mt19937 rng(44);
    return rng;
}

BaseInstPtr_t DMADec::decode() {
    auto& rng = dmaRng();
    if (rng() % 2 == 0) {
        uint64_t rs = rng() % 8, re = rs + (rng() % 8) + 1;
        uint64_t cs = rng() % 8, ce = cs + (rng() % 8) + 1;
        auto addr = std::make_shared<DRAMAddr_t>("random.bin", rs, re, cs, ce);
        auto req = std::make_shared<Req_t>(addr, true);
        int length = static_cast<int>((re - rs) * (ce - cs));
        return std::make_shared<DMAInst_t>(req, 0UL, length);
    } else {
        uint64_t packet_id = rng() % 256;
        std::vector<int> dst = {1};
        auto noc = std::make_shared<NoCInst_t>(0, dst, packet_id);
        return std::make_shared<DMAInst_t>(noc, 0UL, 0);
    }
}

BaseInstPtr_t DMADec::decode(const std::string& inst_str) {
    std::istringstream iss(inst_str);
    std::string opcode, sub_type;
    iss >> opcode >> sub_type;

    if (sub_type != "Cache" && sub_type != "NoC" && sub_type != "DRAM") {
        throw DecodeError("unknown DMA unit '" + sub_type + "' (expected DRAM, NoC or Cache)");
    }

    if (sub_type == "Cache") {
        std::string cache_mode;
        iss >> cache_mode;
        const std::string what = "DMA Cache " + cache_mode;
        if (cache_mode == "copy") {
            uint64_t src_id = 0, dst_id = 0;
            if (!(iss >> src_id >> dst_id)) {
                throw DecodeError(what + ": expected <src_data_id> <dst_data_id>");
            }
            expect_no_more_tokens(iss, what);
            return std::make_shared<DMAInst_t>(src_id, dst_id);
        }
        if (cache_mode == "mc") {
            uint64_t data_id = 0;
            if (!(iss >> data_id)) {
                throw DecodeError(what + ": expected <data_id>");
            }
            expect_no_more_tokens(iss, what);
            return std::make_shared<DMAInst_t>(data_id, /*for_main_ctrl_tag=*/true);
        }
        throw DecodeError("unknown DMA Cache mode '" + cache_mode + "' (expected copy or mc)");
    } else if (sub_type == "NoC") {
        std::string noc_mode;
        uint64_t packet_id = 0, data_id = 0;
        iss >> noc_mode;
        const std::string what = "DMA NoC " + noc_mode;
        if (noc_mode != "send" && noc_mode != "recv") {
            throw DecodeError("unknown DMA NoC mode '" + noc_mode + "' (expected send or recv)");
        }
        if (!(iss >> packet_id >> data_id)) {
            throw DecodeError(what + ": expected <packet_id> <data_id>");
        }

        if (noc_mode == "send") {
            // Format: DMA NoC send <packet_id> <data_id> <dest_id1> [<dest_id2> ...] [truncate]
            // The last token is truncate (0 or 1) if it appears alone after dest ids;
            // we collect all remaining ints, and if the very last one is 0 or 1 and
            // there is at least one dest_id before it, treat it as the truncate flag.
            std::vector<int> tokens;
            int t;
            while (iss >> t) tokens.push_back(t);
            if (tokens.empty()) {
                throw DecodeError(what + ": expected at least one <dest_id>");
            }
            bool truncate = false;
            if (tokens.size() >= 2 && (tokens.back() == 0 || tokens.back() == 1)) {
                truncate = (tokens.back() == 1);
                tokens.pop_back();
            }
            std::vector<int> dst_ids = tokens;
            auto cast_type = (dst_ids.size() > 1)
                ? LLM_SIM_NOC::CastType::MULTICAST
                : LLM_SIM_NOC::CastType::UNICAST;
            auto noc = std::make_shared<NoCInst_t>(0, dst_ids, packet_id,
                                                    cast_type);

            // Optional "rename <new_id>" trailing keyword (see format
            // comment above). The tokens loop above leaves the stream in
            // fail state at the first non-numeric token ("rename" itself,
            // not yet consumed) -- iss.clear() before reading it as a
            // string, same idiom used elsewhere in this file.
            uint64_t new_data_id = 0;
            {
                iss.clear();
                std::string kw;
                if (iss >> kw) {
                    if (kw != "rename") {
                        throw DecodeError(what + ": unexpected token '" + kw + "'");
                    }
                    if (!(iss >> new_data_id)) {
                        throw DecodeError(what + ": 'rename' must be followed by <new_id>");
                    }
                }
            }
            expect_no_more_tokens(iss, what);

            return std::make_shared<DMAInst_t>(noc, data_id, 0, 32, truncate, new_data_id);
        } else {
            std::vector<int> sizes;
            int sz;
            while (iss >> sz) sizes.push_back(sz);
            if (sizes.empty()) {
                throw DecodeError(what + ": expected at least one <size>");
            }
            expect_no_more_tokens(iss, what);
            int total = 0;
            for (int s : sizes) total += s;
            auto noc = std::make_shared<NoCInst_t>(packet_id, data_id, sizes);
            return std::make_shared<DMAInst_t>(noc, data_id, total);
        }
    } else {
        // DRAM path
        std::string dram_mode, file_name, index_mode;
        uint64_t data_id = 0;
        int reserved = 0;
        iss >> dram_mode >> file_name >> index_mode;
        const std::string what = "DMA DRAM " + dram_mode;
        if (dram_mode != "read" && dram_mode != "write") {
            throw DecodeError("unknown DMA DRAM mode '" + dram_mode + "' (expected read or write)");
        }
        if (index_mode != "index" && index_mode != "tensor") {
            throw DecodeError(what + ": expected <file_name> followed by 'index' or 'tensor', got '" +
                              file_name + " " + index_mode + "'");
        }

        int64_t rs = 0, re = 1, cs = 0, ce = 1;
        if (index_mode == "tensor") {
            int batch = 0, seq_s = 0, seq_e = 1, head_s = 0, head_e = 1;
            if (!(iss >> batch >> seq_s >> seq_e >> head_s >> head_e)) {
                throw DecodeError(what + ": expected 5 integers after 'tensor' "
                                  "(<batch> <seq_start> <seq_end> <head_start> <head_end>)");
            }
            rs = static_cast<int64_t>(batch) * MAX_SEQ_LEN + seq_s;
            re = static_cast<int64_t>(batch) * MAX_SEQ_LEN + seq_e;
            cs = static_cast<int64_t>(head_s) * HEAD_DIM;
            ce = static_cast<int64_t>(head_e) * HEAD_DIM;
        } else {
            // "index" mode — direct row/col indices
            if (!(iss >> rs >> re >> cs >> ce)) {
                throw DecodeError(what + ": expected 4 integers after 'index' "
                                  "(<row_start> <row_end> <col_start> <col_end>)");
            }
        }

        if (!(iss >> data_id)) {
            throw DecodeError(what + ": expected <data_id> after the coordinates");
        }
        if (dram_mode == "write" && !(iss >> reserved)) {
            throw DecodeError(what + ": expected <data_reserved> (0 or 1) after <data_id>");
        }

        // Optional "pad <n>" suffix for DRAM read: pads output to n cols per row with zeros.
        // Latency is still computed from actual read size (ce - cs), not padded size.
        // Optional "rename <new_id>" suffix for DRAM write (see format
        // comment above), always after "pad" when both appear -- the two
        // are mutually exclusive in practice (pad is read-only, rename is
        // write-only) but parsed uniformly here since they share one
        // trailing-keyword slot and one shared DMAInst_t construction below.
        int64_t pad_to = 0;
        uint64_t new_data_id = 0;
        {
            std::string kw;
            if (iss >> kw) {
                if (kw == "pad") {
                    if (!(iss >> pad_to)) {
                        throw DecodeError(what + ": 'pad' must be followed by an integer");
                    }
                    std::string kw2;
                    if (iss >> kw2) {
                        if (kw2 != "rename") {
                            throw DecodeError(what + ": unexpected token '" + kw2 + "'");
                        }
                        if (!(iss >> new_data_id)) {
                            throw DecodeError(what + ": 'rename' must be followed by <new_id>");
                        }
                    }
                } else if (kw == "rename") {
                    if (!(iss >> new_data_id)) {
                        throw DecodeError(what + ": 'rename' must be followed by <new_id>");
                    }
                } else {
                    throw DecodeError(what + ": unexpected token '" + kw + "'");
                }
            }
        }
        expect_no_more_tokens(iss, what);

        bool is_read  = (dram_mode == "read");
        bool truncate = (dram_mode == "write") && (reserved != 0);
        int  length   = static_cast<int>((re - rs) * (ce - cs));  // actual (unpadded) size for latency
        const uint64_t row_stride = 0;  // default: col_end
        auto addr = std::make_shared<DRAMAddr_t>(file_name, rs, re, cs, ce, row_stride, pad_to);
        auto req  = std::make_shared<Req_t>(addr, is_read);
        return std::make_shared<DMAInst_t>(req, data_id, length, 32, truncate, new_data_id);
    }
}
