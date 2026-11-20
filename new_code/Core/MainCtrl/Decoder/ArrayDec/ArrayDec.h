#ifndef LLM_SIM_ARRAYDEC_H
#define LLM_SIM_ARRAYDEC_H

#include <string>
#include "utils/BaseInst.h"

class ArrayDec {
public:
    ArrayDec() = default;
    // Returns a random ArrayInst_t (test mode).
    LLM_SIM_INST::BaseInstPtr_t decode();
    // Parses a CSV line starting with "ARRAY" and returns an ArrayInst_t.
    LLM_SIM_INST::BaseInstPtr_t decode(const std::string& inst_str);
};

#endif
