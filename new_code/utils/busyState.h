#ifndef LLM_SIM_BUSYSTATE_H
#define LLM_SIM_BUSYSTATE_H
#include <iostream>
#include <vector>
#include <unordered_map>
#include <memory>
#include <string>
#include <systemc.h>

#include "utils/bufferAddr.h"
#include "utils/BaseInst.h"
// #include "NoC_sim/noc_utils.h"

namespace LLM_SIM_BUSY {
    class BaseBusy_t {
        public:
            bool busy;
            virtual ~BaseBusy_t() = default;
            virtual bool isBusy() const = 0;
            // Returns a deep-copied, independent instance of this busy state.
            // Always use clone() when storing a local prediction copy so that
            // in-place mutations by the module never corrupt the stored state.
            virtual std::shared_ptr<BaseBusy_t> clone() const = 0;
            friend inline void sc_trace(sc_core::sc_trace_file* tf, const BaseBusy_t& obj, const std::string& name) {
                sc_core::sc_trace(tf, obj.busy, name + ".busy");
            }
    };
    using BaseBusyPtr_t = std::shared_ptr<BaseBusy_t>;
    // sc_trace no-op for sc_in<BaseBusyPtr_t> / sc_out<BaseBusyPtr_t> ports
    inline void sc_trace(sc_core::sc_trace_file*, const BaseBusyPtr_t&, const std::string&) {}

    class ArrayBusy_t : public BaseBusy_t {
        public:
            explicit ArrayBusy_t(int id) : id_(id) { busy = false; }
            bool isBusy() const override { return busy; }
            std::shared_ptr<BaseBusy_t> clone() const override {
                return std::make_shared<ArrayBusy_t>(*this);
            }
            bool isBusy(LLM_SIM_BUF::ArrayAddrPtr_t addr_ptr) const {
                if(addr_ptr->getId() == id_ && busy) { // for simplicity, we can assume id 0 is for IS buffer, id 1 is for OS buffer, etc. In real design, we can have a more complex mapping
                    return busy; // if the array is busy, then all buffers are busy. In real design, we can track the busy state of each buffer separately.
                }
                return false;

            }
            friend inline void sc_trace(sc_core::sc_trace_file* tf, const ArrayBusy_t& obj, const std::string& name) {
                sc_core::sc_trace(tf, obj.busy, name + ".busy");
                sc_core::sc_trace(tf, obj.id_,  name + ".id");
            }
        private:
            const int id_;
    };
    using ArrayBusyPtr_t = std::shared_ptr<ArrayBusy_t>;
    // sc_trace no-op for sc_buffer<ArrayBusyPtr_t> (ADL finds this via ArrayBusy_t's namespace)
    inline void sc_trace(sc_core::sc_trace_file*, const ArrayBusyPtr_t&, const std::string&) {}

    class VectorBusy_t : public BaseBusy_t {
        public:
            explicit VectorBusy_t(int id) : id_(id) { busy = false; }
            bool isBusy() const override { return busy; }
            std::shared_ptr<BaseBusy_t> clone() const override {
                return std::make_shared<VectorBusy_t>(*this);
            }
            bool isBusy(LLM_SIM_BUF::VectorAddrPtr_t addr_ptr) const {
                if(addr_ptr->getId() == id_ && busy) { // for simplicity, we can assume id 0 is for IN buffer, id 1 is for WEIGHT buffer, etc. In real design, we can have a more complex mapping
                    return busy; // if the vector is busy, then all buffers are busy. In real design, we can track the busy state of each buffer separately.
                }
                return false;

            }
            friend inline void sc_trace(sc_core::sc_trace_file* tf, const VectorBusy_t& obj, const std::string& name) {
                sc_core::sc_trace(tf, obj.busy, name + ".busy");
                sc_core::sc_trace(tf, obj.id_,  name + ".id");
            }
        private:
            const int id_;
    };
    using VectorBusyPtr_t = std::shared_ptr<VectorBusy_t>;
    // sc_trace no-op for sc_buffer<VectorBusyPtr_t>
    inline void sc_trace(sc_core::sc_trace_file*, const VectorBusyPtr_t&, const std::string&) {}

    class DMABusy_t : public BaseBusy_t {
        public:
            explicit DMABusy_t() { busy = false; }
            bool isBusy() const override { return busy; }
            // Deep-copies busy flag and the full buf_busy_map so the clone is
            // entirely independent — mutations to either copy do not affect the other.
            std::shared_ptr<BaseBusy_t> clone() const override {
                return std::make_shared<DMABusy_t>(*this);
            }
            bool isBusy(std::string name, std::string op) const {
                if(name == "noc") {
                    if(op == "send") {
                        return noc_send;
                    } else if(op == "recv") {
                        return noc_recv;
                    } else {
                        std::cerr << "Error: Invalid op for DMABusy_t::isBusy. Use \"send\" or \"recv\".\n";
                        return false;
                    }
                }
                else if(name == "dram") {
                    if(op == "write") {
                        return dram_write;
                    } else if(op == "read") {
                        return dram_read;
                    } else {
                        std::cerr << "Error: Invalid op for DMABusy_t::isBusy. Use \"write\" or \"read\".\n";
                        return false;
                    }
                } else if(name == "cache") {
                    if(op == "read") {
                        return cache_read;
                    } else if(op == "write") {
                        return cache_write;
                    } else {
                        std::cerr << "Error: Invalid op for DMABusy_t::isBusy. Use \"read\" or \"write\".\n";
                        return false;
                    }
                } else {
                    std::cerr << "Error: Invalid name for DMABusy_t::isBusy. Use \"noc\", \"dram\", or \"cache\".\n";
                    return false;
                }
            }
            bool noc_send  = false, noc_recv  = false; // can be simultaneous
            bool dram_write = false, dram_read = false; // exclusive
            bool cache_read = false, cache_write = false; // cache bandwidth
            friend inline void sc_trace(sc_core::sc_trace_file* tf, const DMABusy_t& obj, const std::string& name) {
                sc_core::sc_trace(tf, obj.busy, name + ".busy");
            }
            
    };
    using DMABusyPtr_t = std::shared_ptr<DMABusy_t>;
    // sc_trace no-op for sc_buffer<DMABusyPtr_t>
    inline void sc_trace(sc_core::sc_trace_file*, const DMABusyPtr_t&, const std::string&) {}

    class TMUBusy_t : public BaseBusy_t {
        public:
            explicit TMUBusy_t() { busy = false; }
            bool isBusy() const override { return busy; }
            std::vector<uint64_t> in_data_ids;
            std::vector<uint64_t> out_data_ids;
            std::shared_ptr<BaseBusy_t> clone() const override {
                return std::make_shared<TMUBusy_t>(*this);
            }
            friend inline void sc_trace(sc_core::sc_trace_file* tf, const TMUBusy_t& obj, const std::string& name) {
                sc_core::sc_trace(tf, obj.busy, name + ".busy");
            }
    };
    using TMUBusyPtr_t = std::shared_ptr<TMUBusy_t>;
    // sc_trace no-op for sc_buffer<TMUBusyPtr_t>
    inline void sc_trace(sc_core::sc_trace_file*, const TMUBusyPtr_t&, const std::string&) {}
    
    /* Redundant busy states for connecting ports */
    class DRAMBusy_t : public BaseBusy_t { // just a redundant busy state for connecting ports
        public:
            explicit DRAMBusy_t() { busy = false; }
            bool isBusy() const override { 
                return num_reqs() > 0; // if there are pending DRAM reqs, then it's busy. In real design, we can have more complex logic to determine busy state based on the reqs (e.g., check if the reqs are for the same bank/partition to determine if the DRAM is busy or not)
             }
            std::shared_ptr<BaseBusy_t> clone() const override {
                return std::make_shared<DRAMBusy_t>(*this);
            }
            int num_reqs() const {
                return pending_reqs.size();
            }
            friend inline void sc_trace(sc_core::sc_trace_file* tf, const DRAMBusy_t& obj, const std::string& name) {
                sc_core::sc_trace(tf, obj.busy, name + ".busy");
            }
            std::vector<LLM_SIM_DRAM::ReqPtr_t> pending_reqs = {}; // track pending DRAM reqs for more complex busy logic in the future (e.g., if we want to track busy state of each bank/partition separately, we can check the reqs' addresses to determine which bank/partition is busy)
    };
    using DRAMBusyPtr_t = std::shared_ptr<DRAMBusy_t>;

    // This Busy State type will be used in for compatability with the old version design
    class NoCBusy_t : public BaseBusy_t {
        public:
            explicit NoCBusy_t() { busy = false; }
            bool isBusy() const override { return busy; }
            std::shared_ptr<BaseBusy_t> clone() const override {
                return std::make_shared<NoCBusy_t>(*this);
            }
            bool isBusy(bool is_send) const { // it can check every kind of buffer
                // For simplicity, we assume the NoC is shared by all buffers and any access to NoC causes the whole NoC to be busy. In real design, we can track the busy state of each channel/partition of the NoC separately.
                if(is_send) {
                    return num_send_pending_insts() > 0;
                } else {
                    return num_recv_pending_insts() > 0;
                }
            }
            int num_send_pending_insts() const {
                return pending_noc_send_insts.size();
            }
            int num_recv_pending_insts() const {
                return pending_noc_recv_insts.size();
            }
            friend inline void sc_trace(sc_core::sc_trace_file* tf, const NoCBusy_t& obj, const std::string& name) {
                sc_core::sc_trace(tf, obj.busy, name + ".busy");
            }
            // LLM_SIM_BUF::BaseAddrPtr_t dst_addr_ptr; // packet dst buffer addr
            std::vector<LLM_SIM_INST::NoCInstPtr_t> pending_noc_recv_insts;
            std::vector<LLM_SIM_INST::NoCInstPtr_t> pending_noc_send_insts;
    };
    using NoCBusyPtr_t = std::shared_ptr<NoCBusy_t>;
}
#endif