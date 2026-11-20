#include "Core/MainCtrl/PortBundle.h"

void PortBundle::addPort(InstType_t type,
                         std::function<bool(const LLM_SIM_INST::BaseInstPtr_t&)> dispatch_fn,
                         std::function<bool()>                                    full_fn) {
    entries_[type].push_back({std::move(dispatch_fn), std::move(full_fn)});
}

bool PortBundle::dispatch(const LLM_SIM_INST::BaseInstPtr_t& inst) {
    InstType_t type = inst->getType();
    auto it = entries_.find(type);
    if (it == entries_.end()) return false;

    for (auto& entry : it->second) {
        if (!entry.is_full()) {
            return entry.try_dispatch(inst);
        }
    }
    return false; // all ports for this type are currently full
}

bool PortBundle::hasType(InstType_t type) const {
    return entries_.find(type) != entries_.end();
}

size_t PortBundle::count(InstType_t type) const {
    auto it = entries_.find(type);
    return (it != entries_.end()) ? it->second.size() : 0;
}

