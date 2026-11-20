#ifndef LLM_SIM_VECTORDEC_H
#define LLM_SIM_VECTORDEC_H

#include <string>
#include "utils/BaseInst.h"

class VectorDec {
public:
    VectorDec() = default;
    LLM_SIM_INST::BaseInstPtr_t decode();
    LLM_SIM_INST::BaseInstPtr_t decode(const std::string& inst_str);
};

#endif
