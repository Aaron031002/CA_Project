#ifndef LLM_SIM_DATA_H
#define LLM_SIM_DATA_H

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <stdint.h>
#include <systemc.h>

#include "utils/BaseInst.h"

/*
    Actually, this is a Data Struct for recording data value and data id
*/


namespace LLM_SIM_DATA{
template <typename T>
    class Data_t {
        public:
            // constructors
            explicit Data_t() {}
            explicit Data_t(uint64_t v) : id(v) {}
            // member variables
            uint64_t id = 0; // unique identifier for the data instance
            std::vector<T> value = {};
            std::vector<int> size_; // (dim_0, dim_1, ...)
            std::vector<int> size() const { return size_; }
            // Logical bit width of one element in this Data_t, for transfer-
            // latency modelling (§1). Defaults to the
            // physical size of T so all existing call sites are unaffected.
            // A producer that represents quantized data (e.g. Q8_0 round-trip
            // output, still physically stored as T=float) can override this to
            // record the "as if transferred at N bits" width without changing
            // the arithmetic path.
            int elem_bits = sizeof(T) * 8;
            // Phase B: the NoC packet_id this
            // Data_t was delivered under, stamped by Router::packet2data.
            // 0 (default) means "not stamped" -- every existing construction
            // site is unaffected. Exists so a receiver fed by multiple
            // simultaneously-outstanding recv requests (DMA's pending_recvs_)
            // can tell which one a given arrival belongs to; Router itself
            // already uses packet_id for exactly this purpose internally
            // (packet_pool), this just carries that identity one hop further
            // to the consumer instead of dropping it at the Router/DMA
            // boundary.
            uint64_t packet_id = 0;
    };
    template <typename T>
    using DataPtr_t = std::shared_ptr<Data_t<T>>;

    // The only sanctioned way to build a CacheWriteReq_t: always derived
    // from the same id/size_/elem_bits a producer is about to write to
    // Cache, so the write-request FIFO and the data FIFO never disagree
    // about what they're describing.
    template <typename T>
    inline LLM_SIM_INST::CacheWriteReqPtr_t make_cache_write_req(const DataPtr_t<T>& data) {
        return std::make_shared<LLM_SIM_INST::CacheWriteReq_t>(data->id, data->size_, data->elem_bits);
    }

    // note: these classes maybe deprecated in the next version
    // View classes: provide multi-dimensional read/write access to the flat 1D vector in Data_t
    template <typename T>
    class View2D {
    public:
        View2D(Data_t<T>& data) : data_(data) {
            if (data_.size_ != 2) {
                throw std::runtime_error("View2D: data size_ must have exactly 2 dimensions.");
            }
            if (static_cast<size_t>(data_.size_[0] * data_.size_[1]) != data_.value.size()) {
                throw std::runtime_error("View2D: dimension mismatch with data size.");
            }
        }
        T& operator()(int i, int j) {
            return data_.value[i * data_.size_[1] + j];
        }
        const T& operator()(int i, int j) const {
            return data_.value[i * data_.size_[1] + j];
        }
        int dim(int axis) const { return data_.size_.at(axis); }
        std::vector<int> size() const { return data_.size_; }
    private:
        Data_t<T>& data_;
    };

    template <typename T>
    class View3D {
    public:
        View3D(Data_t<T>& data) : data_(data) {
            if (data_.size_ != 3) {
                throw std::runtime_error("View3D: data size_ must have exactly 3 dimensions.");
            }
            if (static_cast<size_t>(data_.size_[0] * data_.size_[1] * data_.size_[2]) != data_.value.size()) {
                throw std::runtime_error("View3D: dimension mismatch with data size.");
            }
        }
        T& operator()(int i, int j, int k) {
            return data_.value[(i * data_.size_[1] + j) * data_.size_[2] + k];
        }
        const T& operator()(int i, int j, int k) const {
            return data_.value[(i * data_.size_[1] + j) * data_.size_[2] + k];
        }
        int dim(int axis) const { return data_.size_.at(axis); }
        std::vector<int> size() const { return data_.size_; }
    private:
        Data_t<T>& data_;
    };

    template <typename T>
    class View4D {
    public:
        View4D(Data_t<T>& data) : data_(data) {
            if (data_.size_ != 4) {
                throw std::runtime_error("View4D: data size_ must have exactly 4 dimensions.");
            }
            if (static_cast<size_t>(data_.size_[0] * data_.size_[1] * data_.size_[2] * data_.size_[3]) != data_.value.size()) {
                throw std::runtime_error("View4D: dimension mismatch with data size.");
            }
        }
        T& operator()(int i, int j, int k, int l) {
            return data_.value[((i * data_.size_[1] + j) * data_.size_[2] + k) * data_.size_[3] + l];
        }
        const T& operator()(int i, int j, int k, int l) const {
            return data_.value[((i * data_.size_[1] + j) * data_.size_[2] + k) * data_.size_[3] + l];
        }
        int dim(int axis) const { return data_.size_.at(axis); }
        std::vector<int> size() const { return data_.size_; }
    private:
        Data_t<T>& data_;
    };

    // ── sc_trace no-op: placed in LLM_SIM_DATA so ADL finds it when
    // sc_in/sc_out<DataPtr_t<T>> is instantiated (two-phase lookup, phase 2).
    template <typename T>
    inline void sc_trace(sc_core::sc_trace_file*,
                         const DataPtr_t<T>&,
                         const std::string&) {}

} // namespace LLM_SIM_DATA

#endif