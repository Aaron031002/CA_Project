#ifndef LLM_SIM_DECODE_ERROR_H
#define LLM_SIM_DECODE_ERROR_H

#include <sstream>
#include <stdexcept>
#include <string>

// Thrown by Decoder / ArrayDec / VectorDec / DMADec / TMUDec when an
// instruction line can't be decoded exactly as written: unknown opcode or
// mode, a missing or non-numeric required field, or a leftover token the
// format doesn't allow. Before this existed the decoders silently fell back
// to a default (e.g. an unknown Vector mode became RMS, `DMA DRAM reed`
// became a write) or skipped the line, so a typo ran as a different
// instruction. MainCtrl catches it and stops the simulation with the file,
// line number and message (SC_REPORT_FATAL).
class DecodeError : public std::runtime_error {
    public:
        explicit DecodeError(const std::string& msg) : std::runtime_error(msg) {}
};

// Throws DecodeError if `iss` has any token left. `what` names the
// instruction for the message, e.g. "Vector RELU".
inline void expect_no_more_tokens(std::istringstream& iss, const std::string& what) {
    iss.clear();
    std::string extra;
    if (iss >> extra) {
        throw DecodeError(what + ": unexpected token '" + extra + "'");
    }
}

#endif
