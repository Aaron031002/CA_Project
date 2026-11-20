#include "Core/MainCtrl/Decoder/Decoder.h"


Decoder::Decoder() {
    array_dec_  = new ArrayDec();
    vector_dec_ = new VectorDec();
    dma_dec_    = new DMADec();
    tmu_dec_    = new TMUDec();
}

BaseInstPtr_t Decoder::decode() {
    // Random mode: pick uniformly from the four supported unit types.
    const int rand_num = rand() % 4;
    switch (rand_num) {
        case 0: return array_dec_->decode();
        case 1: return vector_dec_->decode();
        case 2: return dma_dec_->decode();
        case 3: return tmu_dec_->decode();
        default: return nullptr;
    }
}

BaseInstPtr_t Decoder::decode(const std::string& inst_str) {
    std::istringstream iss(inst_str);
    std::string opcode;
    iss >> opcode;
    if (opcode == "Array" || opcode == "ARRAY") {
        return array_dec_->decode(inst_str);
    } else if (opcode == "Vector" || opcode == "VECTOR") {
        return vector_dec_->decode(inst_str);
    } else if (opcode == "DMA") {
        return dma_dec_->decode(inst_str);
    } else if (opcode == "TMU") {
        return tmu_dec_->decode(inst_str);
    } else if (opcode == "TCore" || opcode == "TCORE") {
        std::string mode_str;
        iss >> mode_str;
        if (mode_str == "WAIT") {
            expect_no_more_tokens(iss, "TCore WAIT");
            return std::make_shared<TCoreInst_t>(TCoreInst_t::WAIT);
        } else if (mode_str == "LOCAL_SYNC") {
            expect_no_more_tokens(iss, "TCore LOCAL_SYNC");
            return std::make_shared<TCoreInst_t>(TCoreInst_t::LOCAL_SYNC);
        } else if (mode_str == "GLOBAL_SYNC") {
            // Format: TCore GLOBAL_SYNC <barrier_id> <N> <core_id_0> ... <core_id_{N-1}>
            uint64_t bid = 0;
            int n = 0;
            if (!(iss >> bid >> n) || n < 1) {
                throw DecodeError("TCore GLOBAL_SYNC: expected <barrier_id> <N> with N >= 1");
            }
            std::vector<int> participants(n);
            for (int k = 0; k < n; ++k) {
                if (!(iss >> participants[k])) {
                    throw DecodeError("TCore GLOBAL_SYNC: N=" + std::to_string(n) +
                                      " but only " + std::to_string(k) + " core ids given");
                }
            }
            expect_no_more_tokens(iss, "TCore GLOBAL_SYNC");
            return std::make_shared<TCoreInst_t>(TCoreInst_t::GLOBAL_SYNC, bid, participants);
        } else {
            throw DecodeError("unknown TCore mode '" + mode_str +
                              "' (expected WAIT, LOCAL_SYNC or GLOBAL_SYNC)");
        }
    } else {
        throw DecodeError("unknown opcode '" + opcode +
                          "' (expected Array, Vector, DMA, TMU or TCore)");
    }
}
