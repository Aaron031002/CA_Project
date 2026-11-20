#ifndef LLM_SIM_PORTBUNDLE_H
#define LLM_SIM_PORTBUNDLE_H

#include <functional>
#include <map>
#include <vector>
#include <stdexcept>

#include "utils/BaseInst.h"

using namespace LLM_SIM_INST;

// A dispatch entry holds two callbacks for one (type, index) FIFO port:
//   try_dispatch : calls nb_write on the bound sc_fifo_out; returns true on success.
//   is_full      : returns true when the FIFO has no free slots.
struct DispatchEntry_t {
    std::function<bool(const LLM_SIM_INST::BaseInstPtr_t&)> try_dispatch;
    std::function<bool()>                                    is_full;
};

// PortBundle: maps InstType_t to an ordered list of DispatchEntry_t objects.
//
// Compared to the previous version the hazard-checking and local-busy
// prediction logic has been removed.  The only stall condition is a full
// FIFO (buffer).  Busy-state ports are still declared on MainCtrl for
// hardware compatibility but are not queried here.
//
// Usage:
//   bundle.addPort(InstType_t::ARRAY,
//       [&](const BaseInstPtr_t& i){ return array_inst_out[0].nb_write(...); },
//       [&]{ return array_inst_out[0].num_free() == 0; });
//
//   bool ok = bundle.dispatch(inst);  // tries first non-full port of inst->type
class PortBundle {
public:
    // Register a dispatch-callback pair for a given instruction type.
    void addPort(InstType_t type,
                 std::function<bool(const LLM_SIM_INST::BaseInstPtr_t&)> dispatch_fn,
                 std::function<bool()>                                    full_fn);

    // Attempt to dispatch `inst` to the first non-full port registered for
    // inst->getType().  Returns true if an instruction was written, false if
    // every port of that type is currently full (caller should retry next cycle).
    bool dispatch(const LLM_SIM_INST::BaseInstPtr_t& inst);

    // Whether any ports have been registered for `type`.
    bool hasType(InstType_t type) const;

    // Number of ports registered for `type`.
    size_t count(InstType_t type) const;

private:
    std::map<InstType_t, std::vector<DispatchEntry_t>> entries_;
};

#endif
