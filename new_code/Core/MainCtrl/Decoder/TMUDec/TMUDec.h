#ifndef LLM_SIM_TMUDEC_H
#define LLM_SIM_TMUDEC_H

#include <string>
#include "utils/BaseInst.h"

class TMUDec {
public:
    TMUDec() = default;
    // Returns a random TMUInst_t (test mode).
    LLM_SIM_INST::BaseInstPtr_t decode();
    // Parses a text line starting with "TMU" and returns a TMUInst_t.
    LLM_SIM_INST::BaseInstPtr_t decode(const std::string& inst_str);
};

#endif
