#ifndef LLM_SIM_DECODER_H
#define LLM_SIM_DECODER_H

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <random>
#include <sstream>
#include <memory>

#include "utils/BaseInst.h"
#include "Core/MainCtrl/Decoder/DecodeError.h"
#include "Core/MainCtrl/Decoder/ArrayDec/ArrayDec.h"
#include "Core/MainCtrl/Decoder/VectorDec/VectorDec.h"
#include "Core/MainCtrl/Decoder/DMADec/DMADec.h"
#include "Core/MainCtrl/Decoder/TMUDec/TMUDec.h"

using namespace LLM_SIM_INST;

class Decoder {
    public:
        Decoder();
        // Returns a random instruction (test mode).
        BaseInstPtr_t decode();
        // Parses a text line and returns the appropriate instruction object.
        // Supported opcodes: Array, Vector, DMA, TMU, TCore. Throws
        // DecodeError if the line doesn't match its opcode's format.
        BaseInstPtr_t decode(const std::string& inst_str);
    private:
        ArrayDec*  array_dec_;
        VectorDec* vector_dec_;
        DMADec*    dma_dec_;
        TMUDec*    tmu_dec_;
};

#endif