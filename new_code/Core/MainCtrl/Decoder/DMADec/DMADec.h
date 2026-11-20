#ifndef LLM_SIM_DMADEC_H
#define LLM_SIM_DMADEC_H

#include <string>
#include "utils/BaseInst.h"

class DMADec {
public:
    DMADec() = default;
    LLM_SIM_INST::BaseInstPtr_t decode();
    LLM_SIM_INST::BaseInstPtr_t decode(const std::string& inst_str);
};

#endif
