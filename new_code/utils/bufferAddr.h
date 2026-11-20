#ifndef LLM_SIM_BUFFADDR_H
#define LLM_SIM_BUFFADDR_H

#include <memory>
#include <string>
#include <systemc.h>


/*
    Note: This file is deprecated in the next version
    Becase the buffers are merged in one collective buffer pool

*/


namespace LLM_SIM_BUF {
    enum class ModuleType_t {
        ARRAY, VECTOR, DMA, DRAM, NOC, NUM_MODULES
    };

    class BaseAddr_t {
        public:
            virtual ~BaseAddr_t() = default;
            virtual ModuleType_t getModuleType() const = 0;
    };
    using BaseAddrPtr_t = std::shared_ptr<BaseAddr_t>;


    // This one will be deprecated in the next version
    class ArrayAddr_t : public BaseAddr_t {
        public:
            enum class ArrayBufType_t {
                IN, OUT, BUF_NUM
            };
            ArrayAddr_t(int id, ArrayBufType_t buf_type) : id_(id), buf_type_(buf_type) {}
            int getId() const { return id_; }
            ModuleType_t getModuleType() const override {
                return ModuleType_t::ARRAY;
            }
            ArrayBufType_t getBufType() const {
                return buf_type_;
            }
            friend inline void sc_trace(sc_core::sc_trace_file* tf, const ArrayAddr_t& obj, const std::string& name) {
                sc_core::sc_trace(tf, static_cast<int>(obj.getModuleType()), name + ".module_type");
                sc_core::sc_trace(tf, static_cast<int>(obj.getBufType()),    name + ".buf_type");
            }
        private:
            const int id_;
            const ArrayBufType_t buf_type_;
    };
    using ArrayAddrPtr_t = std::shared_ptr<ArrayAddr_t>;



    // This one will be deprecated in the next version
    class VectorAddr_t : public BaseAddr_t {
        public:
            enum class VectorBufType_t {
                IN, WEIGHT, OUT, BUF_NUM
            };
            VectorAddr_t(int id, VectorBufType_t buf_type) : id_(id), buf_type_(buf_type) {}
            int getId() const { return id_; }
            ModuleType_t getModuleType() const override {
                return ModuleType_t::VECTOR;
            }
            VectorBufType_t getBufType() const {
                return buf_type_;
            }
            friend inline void sc_trace(sc_core::sc_trace_file* tf, const VectorAddr_t& obj, const std::string& name) {
                sc_core::sc_trace(tf, static_cast<int>(obj.getModuleType()), name + ".module_type");
                sc_core::sc_trace(tf, static_cast<int>(obj.getBufType()),    name + ".buf_type");
            }
        private:
            const int id_;
            const VectorBufType_t buf_type_;
    };
    using VectorAddrPtr_t = std::shared_ptr<VectorAddr_t>;



    // This one will be deprecated in the next version
    class DMAAddr_t : public BaseAddr_t {
        public:
            enum class DMABufType_t {
                K_BUFFER, V_BUFFER, RMS_BUF, BUF_NUM 
            };
            DMAAddr_t(int id, DMABufType_t buf_type) : id_(id), buf_type_(buf_type) {}
            int getId() const { return id_; }
            ModuleType_t getModuleType() const override {
                return ModuleType_t::DMA;
            }
            DMABufType_t getBufType() const {
                return buf_type_;
            }
            friend inline void sc_trace(sc_core::sc_trace_file* tf, const DMAAddr_t& obj, const std::string& name) {
                sc_core::sc_trace(tf, static_cast<int>(obj.getModuleType()), name + ".module_type");
                sc_core::sc_trace(tf, static_cast<int>(obj.getBufType()),    name + ".buf_type");
            }
        private:
            const int id_;
            const DMABufType_t buf_type_;
    };
    using DMAAddrPtr_t = std::shared_ptr<DMAAddr_t>;



    class DRAMAddr_t : public BaseAddr_t {
        public:
            enum class DramBufType_t {
                READ, WRITE, BUF_NUM
            };
            DRAMAddr_t(int id, DramBufType_t buf_type) : id_(id), buf_type_(buf_type) {}
            int getId() const { return id_; }
            ModuleType_t getModuleType() const override {
                return ModuleType_t::DRAM;
            }
            DramBufType_t getBufType() const {
                return buf_type_;
            }
            friend inline void sc_trace(sc_core::sc_trace_file* tf, const DRAMAddr_t& obj, const std::string& name) {
                sc_core::sc_trace(tf, static_cast<int>(obj.getModuleType()), name + ".module_type");
                sc_core::sc_trace(tf, static_cast<int>(obj.getBufType()),    name + ".buf_type");
            }
        private:
            const int id_;
            const DramBufType_t buf_type_;
    };
    using DRAMAddrPtr_t = std::shared_ptr<DRAMAddr_t>;



    class NocAddr_t : public BaseAddr_t {
        public:
            enum class NocBufType_t {
                READ, WRITE, BUF_NUM
            };
            NocAddr_t(int id, NocBufType_t buf_type) : id_(id), buf_type_(buf_type) {}
            int getId() const { return id_; }
            ModuleType_t getModuleType() const override {
                return ModuleType_t::NOC;
            }
            NocBufType_t getBufType() const {
                return buf_type_;
            }
            friend inline void sc_trace(sc_core::sc_trace_file* tf, const NocAddr_t& obj, const std::string& name) {
                sc_core::sc_trace(tf, static_cast<int>(obj.getModuleType()), name + ".module_type");
                sc_core::sc_trace(tf, static_cast<int>(obj.getBufType()),    name + ".buf_type");
            }
        private:
            const int id_;
            const NocBufType_t buf_type_;
    };
    using NocAddrPtr_t = std::shared_ptr<NocAddr_t>;
}

#endif