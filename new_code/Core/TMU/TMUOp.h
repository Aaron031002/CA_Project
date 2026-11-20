#ifndef LLM_SIM_TMUOP_H
#define LLM_SIM_TMUOP_H

#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <utility>

#include "Core/TMU/tmu_params.h"
#include "utils/BaseInst.h"
#include "utils/Data.h"

template <typename T>
class TMUOp {
    public:
        TMUOp();
        // user API
        std::pair<std::vector<LLM_SIM_DATA::DataPtr_t<T>>, uint64_t> execute(const LLM_SIM_INST::TMUInstPtr_t& inst_ptr, const std::vector<LLM_SIM_DATA::DataPtr_t<T>>& input_data);
        // Compute latency only (no data movement) — for compute_=false transaction mode
        uint64_t latency_only(const LLM_SIM_INST::TMUInstPtr_t& inst_ptr, const std::vector<LLM_SIM_DATA::DataPtr_t<T>>& input_data) const;

        // Per-output {shape, elem_bits} metadata execute() would produce for
        // this instruction, mirroring each mode's own out->size_/elem_bits
        // assignment exactly (scatter/split/cancat/trans/view/pad/gather
        // above) WITHOUT moving any data. The returned vector's length
        // always matches what execute() would actually produce for
        // input_data (e.g. scatter/split: input_data.size() * num_parts;
        // concat/gather: 1; trans/view/pad: input_data.size()) -- the
        // transaction-only path zero-fills each entry in order.
        struct OutputMeta { std::vector<int> shape; int elem_bits; };
        std::vector<OutputMeta> output_meta(
            const LLM_SIM_INST::TMUInstPtr_t& inst_ptr,
            const std::vector<LLM_SIM_DATA::DataPtr_t<T>>& input_data) const;

    private:
        // real operation functions
        std::pair<std::vector<LLM_SIM_DATA::DataPtr_t<T>>, uint64_t> scatter(const std::vector<LLM_SIM_DATA::DataPtr_t<T>>& input_data, const int num_parts) const;
        std::pair<std::vector<LLM_SIM_DATA::DataPtr_t<T>>, uint64_t> split(const std::vector<LLM_SIM_DATA::DataPtr_t<T>>& input_data, const int dim, const int num_parts) const;
        std::pair<std::vector<LLM_SIM_DATA::DataPtr_t<T>>, uint64_t> cancat(const std::vector<LLM_SIM_DATA::DataPtr_t<T>>& input_data, const int dim) const;
        std::pair<std::vector<LLM_SIM_DATA::DataPtr_t<T>>, uint64_t> trans(const std::vector<LLM_SIM_DATA::DataPtr_t<T>>& input_data) const;
        std::pair<std::vector<LLM_SIM_DATA::DataPtr_t<T>>, uint64_t> view(const std::vector<LLM_SIM_DATA::DataPtr_t<T>>& input_data, const std::vector<int>& view_shape) const;
        std::pair<std::vector<LLM_SIM_DATA::DataPtr_t<T>>, uint64_t> pad(const std::vector<LLM_SIM_DATA::DataPtr_t<T>>& input_data, const int dim, const int num_parts) const;
        // gather: index-select along dim
        // input_data[0] = source tensor; input_data[1] = indices tensor (flat, stored as float-cast ints)
        // output shape: source shape with dim-axis size replaced by number of indices
        std::pair<std::vector<LLM_SIM_DATA::DataPtr_t<T>>, uint64_t> gather(const std::vector<LLM_SIM_DATA::DataPtr_t<T>>& input_data, const int dim) const;
        // helper functions
        uint64_t latency_cal(const int data_size) const;
};



#endif