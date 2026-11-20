#include "Core/TMU/TMUOp.h"
#include <cmath>
#include <limits>
#include <iostream>
#include <cstdlib>
#include <stdexcept>

template <typename T>
TMUOp<T>::TMUOp() {}

template <typename T>
std::pair<std::vector<LLM_SIM_DATA::DataPtr_t<T>>, uint64_t> TMUOp<T>::execute(
        const LLM_SIM_INST::TMUInstPtr_t& inst_ptr,
        const std::vector<LLM_SIM_DATA::DataPtr_t<T>>& input_data) {
    switch (inst_ptr->mode) {
        case LLM_SIM_INST::TMUInst_t::TMUMode_t::concat:
            return cancat(input_data, inst_ptr->dim);
        case LLM_SIM_INST::TMUInst_t::TMUMode_t::scatter:
            return scatter(input_data, inst_ptr->num_parts);
        case LLM_SIM_INST::TMUInst_t::TMUMode_t::split:
            return split(input_data, inst_ptr->dim, inst_ptr->num_parts);
        case LLM_SIM_INST::TMUInst_t::TMUMode_t::trans:
            return trans(input_data);
        case LLM_SIM_INST::TMUInst_t::TMUMode_t::view:
            return view(input_data, inst_ptr->view_shape);
        case LLM_SIM_INST::TMUInst_t::TMUMode_t::pad:
            return pad(input_data, inst_ptr->dim, inst_ptr->num_parts);
        case LLM_SIM_INST::TMUInst_t::TMUMode_t::gather:
            return gather(input_data, inst_ptr->dim);
        default:
            std::cerr << "Error: Invalid TMU mode.\n";
            return {{}, 0};
    }
}

// ── scatter ──────────────────────────────────────────────────────────────────
// Splits each input along dim 0 into num_parts equal parts.
// Latency: proportional to total_input_size * num_parts
// (each element is copied to num_parts destinations).
template <typename T>
std::pair<std::vector<LLM_SIM_DATA::DataPtr_t<T>>, uint64_t> TMUOp<T>::scatter(
        const std::vector<LLM_SIM_DATA::DataPtr_t<T>>& input_data,
        const int num_parts) const {
    std::vector<LLM_SIM_DATA::DataPtr_t<T>> output_data;
    int total_flat_in = 0;

    for (const auto& data_ptr : input_data) {
        int ndim = static_cast<int>(data_ptr->size_.size());
        int d0 = data_ptr->size_[0];
        int part_d0 = d0 / num_parts;

        // Number of elements in one dim-0 slice
        int slice_size = 1;
        for (int d = 1; d < ndim; d++) slice_size *= data_ptr->size_[d];

        // Output shape: same as input but dim 0 = part_d0
        std::vector<int> out_shape = data_ptr->size_;
        out_shape[0] = part_d0;

        for (int p = 0; p < num_parts; p++) {
            auto out = std::make_shared<LLM_SIM_DATA::Data_t<T>>();
            out->size_ = out_shape;
            // propagate the
            // source's logical bit width (e.g. Q8_0-quantized data,
            // elem_bits=8) -- without this, every TMUOp output silently
            // reverts to the sizeof(T)*8 default, discarding any upstream
            // producer's elem_bits tag (Data.h's own doc comment: never
            // touches value/arithmetic, latency/capacity accounting only).
            out->elem_bits = data_ptr->elem_bits;
            int offset = p * part_d0 * slice_size;
            out->value.assign(data_ptr->value.begin() + offset,
                              data_ptr->value.begin() + offset + part_d0 * slice_size);
            output_data.push_back(out);
        }
        total_flat_in += d0 * slice_size;
    }
    // Scatter cost: data is broadcast to num_parts destinations
    return {output_data, latency_cal(total_flat_in * num_parts)};
}

// ── split ─────────────────────────────────────────────────────────────────────
// Splits each input along 'dim' into num_parts equal parts.
// Negative dim is resolved Python-style: -1 = last dim, -2 = second-to-last, etc.
template <typename T>
std::pair<std::vector<LLM_SIM_DATA::DataPtr_t<T>>, uint64_t> TMUOp<T>::split(
        const std::vector<LLM_SIM_DATA::DataPtr_t<T>>& input_data,
        const int dim, const int num_parts) const {
    std::vector<LLM_SIM_DATA::DataPtr_t<T>> output_data;
    int total_flat_in = 0;

    if (std::getenv("DEBUG_DUMP_SPLIT_INPUT")) {
        for (const auto& data_ptr : input_data) {
            double vmin = std::numeric_limits<double>::infinity();
            double vmax = -std::numeric_limits<double>::infinity();
            int n_nan = 0, n_inf = 0;
            for (T x : data_ptr->value) {
                double d = static_cast<double>(x);
                if (std::isnan(d)) { ++n_nan; continue; }
                if (std::isinf(d)) { ++n_inf; continue; }
                if (d < vmin) vmin = d;
                if (d > vmax) vmax = d;
            }
            std::cerr << "[SPLIT-INPUT] in_id=" << data_ptr->id
                       << " size_=[";
            for (int s : data_ptr->size_) std::cerr << s << ",";
            std::cerr << "] value.size()=" << data_ptr->value.size()
                       << " min=" << vmin << " max=" << vmax
                       << " n_nan=" << n_nan << " n_inf=" << n_inf << "\n" << std::flush;
        }
    }

    for (const auto& data_ptr : input_data) {
        int ndim = static_cast<int>(data_ptr->size_.size());
        const int actual_dim = (dim < 0) ? ndim + dim : dim;
        if (actual_dim < 0 || actual_dim >= ndim) {
            throw std::runtime_error(
                "TMUOp::split: dim=" + std::to_string(dim) + " (actual_dim=" +
                std::to_string(actual_dim) + ") out of range for input (id=" +
                std::to_string(data_ptr->id) + ") ndim=" + std::to_string(ndim));
        }
        int dim_size = data_ptr->size_[actual_dim];
        int part_size = dim_size / num_parts;

        int in_flat = 1;
        for (int s : data_ptr->size_) in_flat *= s;
        if (in_flat != static_cast<int>(data_ptr->value.size())) {
            throw std::runtime_error(
                "TMUOp::split: input (id=" + std::to_string(data_ptr->id) +
                ") size_ product=" + std::to_string(in_flat) + " but value.size()=" +
                std::to_string(data_ptr->value.size()) + " (data corruption)");
        }
        total_flat_in += in_flat;

        // Degenerate split: dim is too small to divide (e.g. [1,16] split 16
        // times along dim=0 yields part_size=0).  Fall back to replicating the
        // full input data num_parts times so each consumer gets a complete copy.
        // Shape is preserved (including the size-1 split dim) so that downstream
        // concat operations see the expected multi-dimensional layout.
        if (part_size == 0) {
            for (int p = 0; p < num_parts; p++) {
                auto out = std::make_shared<LLM_SIM_DATA::Data_t<T>>();
                out->size_  = data_ptr->size_;   // keep original shape (e.g. {1,16})
                out->value  = data_ptr->value;   // full copy
                out->elem_bits = data_ptr->elem_bits;  // see scatter()'s doc comment
                output_data.push_back(out);
            }
            continue;
        }

        // Input strides (row-major)
        std::vector<int> in_stride(ndim, 1);
        for (int d = ndim - 2; d >= 0; d--)
            in_stride[d] = in_stride[d + 1] * data_ptr->size_[d + 1];

        // Output shape: same as input but actual_dim = part_size
        std::vector<int> out_shape = data_ptr->size_;
        out_shape[actual_dim] = part_size;

        // Output strides
        std::vector<int> out_stride(ndim, 1);
        for (int d = ndim - 2; d >= 0; d--)
            out_stride[d] = out_stride[d + 1] * out_shape[d + 1];

        int out_total = 1;
        for (int s : out_shape) out_total *= s;

        for (int p = 0; p < num_parts; p++) {
            auto out = std::make_shared<LLM_SIM_DATA::Data_t<T>>();
            out->size_ = out_shape;
            out->value.resize(out_total);
            out->elem_bits = data_ptr->elem_bits;  // see scatter()'s doc comment

            for (int idx = 0; idx < out_total; idx++) {
                // Decompose output flat index to multi-dim coordinates
                std::vector<int> coord(ndim);
                int tmp = idx;
                for (int d = ndim - 1; d >= 0; d--) {
                    coord[d] = tmp % out_shape[d];
                    tmp /= out_shape[d];
                }
                // Shift the split dimension to the correct input slice
                coord[actual_dim] += p * part_size;
                // Compute input flat index
                int in_idx = 0;
                for (int d = 0; d < ndim; d++) in_idx += coord[d] * in_stride[d];
                out->value[idx] = data_ptr->value[in_idx];
            }
            output_data.push_back(out);
        }
    }
    return {output_data, latency_cal(total_flat_in)};
}

// ── concat ────────────────────────────────────────────────────────────────────
// Concatenates all inputs along 'dim'.
// Negative dim is resolved Python-style: -1 = last dim, -2 = second-to-last, etc.
// All inputs must match in every dimension except 'dim'.
template <typename T>
std::pair<std::vector<LLM_SIM_DATA::DataPtr_t<T>>, uint64_t> TMUOp<T>::cancat(
        const std::vector<LLM_SIM_DATA::DataPtr_t<T>>& input_data,
        const int dim) const {
    if (input_data.empty()) return {{}, 0};
    int ndim = static_cast<int>(input_data[0]->size_.size());
    const int actual_dim = (dim < 0) ? ndim + dim : dim;

    if (actual_dim < 0 || actual_dim >= ndim) {
        throw std::runtime_error(
            "TMUOp::cancat: dim=" + std::to_string(dim) + " (actual_dim=" +
            std::to_string(actual_dim) + ") out of range for input[0] ndim=" +
            std::to_string(ndim) + " (size_.size()=" + std::to_string(ndim) + ")");
    }
    for (size_t k = 0; k < input_data.size(); k++) {
        const auto& sz = input_data[k]->size_;
        int in_total = 1;
        for (int s : sz) in_total *= s;
        if (static_cast<int>(sz.size()) != ndim) {
            throw std::runtime_error(
                "TMUOp::cancat: input[" + std::to_string(k) + "] (id=" +
                std::to_string(input_data[k]->id) + ") has ndim=" +
                std::to_string(sz.size()) + ", expected " + std::to_string(ndim) +
                " (must match input[0]'s ndim)");
        }
        if (in_total != static_cast<int>(input_data[k]->value.size())) {
            throw std::runtime_error(
                "TMUOp::cancat: input[" + std::to_string(k) + "] (id=" +
                std::to_string(input_data[k]->id) + ") size_ product=" +
                std::to_string(in_total) + " but value.size()=" +
                std::to_string(input_data[k]->value.size()) + " (data corruption)");
        }
    }

    // Build output shape: sum along actual_dim
    std::vector<int> out_shape = input_data[0]->size_;
    for (size_t k = 1; k < input_data.size(); k++)
        out_shape[actual_dim] += input_data[k]->size_[actual_dim];

    int total_out = 1;
    for (int s : out_shape) total_out *= s;

    // Output strides
    std::vector<int> out_stride(ndim, 1);
    for (int d = ndim - 2; d >= 0; d--)
        out_stride[d] = out_stride[d + 1] * out_shape[d + 1];

    if (std::getenv("DEBUG_DUMP_CONCAT_INPUT")) {
        for (const auto& data_ptr : input_data) {
            std::cerr << "[CONCAT-INPUT] in_id=" << data_ptr->id << " size_=[";
            for (int s : data_ptr->size_) std::cerr << s << ",";
            std::cerr << "] value.size()=" << data_ptr->value.size() << "\n" << std::flush;
        }
        std::cerr << "[CONCAT-OUTPUT] out_shape=[";
        for (int s : out_shape) std::cerr << s << ",";
        std::cerr << "] total_out=" << total_out << "\n" << std::flush;
    }

    auto out = std::make_shared<LLM_SIM_DATA::Data_t<T>>();
    out->size_ = out_shape;
    out->value.resize(total_out);
    // / see scatter()'s doc comment
    // above for why this matters. concat merges multiple inputs -- only
    // propagate elem_bits when every input agrees (the common case, e.g.
    // K's per-head tmu_concat where every head shares the same source
    // format); a genuine mismatch falls back to Data_t's own default
    // rather than silently picking an arbitrary input's value.
    {
        bool all_same = true;
        for (const auto& dp : input_data) {
            if (dp->elem_bits != input_data[0]->elem_bits) { all_same = false; break; }
        }
        if (all_same) out->elem_bits = input_data[0]->elem_bits;
    }

    int dim_offset = 0;
    for (const auto& data_ptr : input_data) {
        int in_total = 1;
        for (int s : data_ptr->size_) in_total *= s;

        // Input strides
        std::vector<int> in_stride(ndim, 1);
        for (int d = ndim - 2; d >= 0; d--)
            in_stride[d] = in_stride[d + 1] * data_ptr->size_[d + 1];

        for (int idx = 0; idx < in_total; idx++) {
            // Decompose to multi-dim coordinates
            std::vector<int> coord(ndim);
            int tmp = idx;
            for (int d = ndim - 1; d >= 0; d--) {
                coord[d] = tmp % data_ptr->size_[d];
                tmp /= data_ptr->size_[d];
            }
            // Offset the concat dimension
            coord[actual_dim] += dim_offset;
            // Compute output flat index
            int out_idx = 0;
            for (int d = 0; d < ndim; d++) out_idx += coord[d] * out_stride[d];
            out->value[out_idx] = data_ptr->value[idx];
        }
        dim_offset += data_ptr->size_[actual_dim];
    }
    if (std::getenv("DEBUG_DUMP_CONCAT_INPUT")) {
        std::cerr << "[CONCAT-RESULT] size_=[";
        for (int s : out->size_) std::cerr << s << ",";
        std::cerr << "] value.size()=" << out->value.size() << "\n" << std::flush;
    }
    return {{out}, latency_cal(total_out)};
}

// ── trans ─────────────────────────────────────────────────────────────────────
// Transposes the last two dimensions of each input tensor.
// Input shape [..., R, C] -> output shape [..., C, R].
// All leading dimensions are treated as an outer batch loop.
template <typename T>
std::pair<std::vector<LLM_SIM_DATA::DataPtr_t<T>>, uint64_t> TMUOp<T>::trans(
        const std::vector<LLM_SIM_DATA::DataPtr_t<T>>& input_data) const {
    std::vector<LLM_SIM_DATA::DataPtr_t<T>> output_data;
    int total_size = 0;

    for (const auto& data_ptr : input_data) {
        int ndim = static_cast<int>(data_ptr->size_.size());
        int R = data_ptr->size_[ndim - 2];  // second-to-last dim
        int C = data_ptr->size_[ndim - 1];  // last dim

        // Product of all leading (batch) dimensions
        int outer = 1;
        for (int d = 0; d < ndim - 2; d++) outer *= data_ptr->size_[d];

        auto out = std::make_shared<LLM_SIM_DATA::Data_t<T>>();
        out->size_ = data_ptr->size_;
        std::swap(out->size_[ndim - 2], out->size_[ndim - 1]);  // [..., C, R]
        out->value.resize(static_cast<size_t>(outer) * R * C);
        out->elem_bits = data_ptr->elem_bits;  // see scatter()'s doc comment

        for (int b = 0; b < outer; b++) {
            for (int r = 0; r < R; r++) {
                for (int c = 0; c < C; c++) {
                    int in_idx  = b * R * C + r * C + c;
                    int out_idx = b * R * C + c * R + r;
                    out->value[out_idx] = data_ptr->value[in_idx];
                }
            }
        }
        output_data.push_back(out);
        total_size += outer * R * C;
    }
    return {output_data, latency_cal(total_size)};
}

// ── view ──────────────────────────────────────────────────────────────────────
// Reinterprets each input tensor with a new shape without moving data.
// view_shape total elements must equal the input's total element count,
// OR the input can be broadcast to the target shape (numpy-style broadcasting).
// Latency: a pure reshape (in_total == new_total) touches no data — it is a
// metadata-only operation, so it is billed as a fixed 1-cycle op regardless of
// size. Only the broadcast-expansion path (which actually materialises new
// elements) is billed proportional to its output size.
template <typename T>
std::pair<std::vector<LLM_SIM_DATA::DataPtr_t<T>>, uint64_t> TMUOp<T>::view(
        const std::vector<LLM_SIM_DATA::DataPtr_t<T>>& input_data,
        const std::vector<int>& view_shape) const {
    int new_total = 1;
    for (int s : view_shape) new_total *= s;

    std::vector<LLM_SIM_DATA::DataPtr_t<T>> output_data;
    int broadcast_total_size = 0;

    for (const auto& data_ptr : input_data) {
        int in_total = static_cast<int>(data_ptr->value.size());
        auto out = std::make_shared<LLM_SIM_DATA::Data_t<T>>();
        out->elem_bits = data_ptr->elem_bits;  // see scatter()'s doc comment
        if (in_total == new_total) {
            // Simple reshape — no data copy needed, metadata-only, free.
            out->value = data_ptr->value;
        } else {
            // Try broadcast: input shape dims must be 1 or equal to view_shape dims
            // (right-aligned, numpy style). If compatible, expand input to view_shape.
            const auto& in_shape = data_ptr->size_;
            int in_ndim  = static_cast<int>(in_shape.size());
            int out_ndim = static_cast<int>(view_shape.size());
            bool ok = true;
            // Check broadcast compatibility (right-align shapes)
            for (int d = 0; d < out_ndim; ++d) {
                int in_d = d - (out_ndim - in_ndim);
                int in_sz  = (in_d >= 0) ? in_shape[in_d] : 1;
                int out_sz = view_shape[d];
                if (in_sz != 1 && in_sz != out_sz) { ok = false; break; }
            }
            if (!ok) {
                std::cerr << "Error: view size mismatch: input has " << in_total
                          << " elements but view_shape implies " << new_total << ".\n";
                return {{}, 0};
            }
            // Perform broadcast expansion
            out->value.resize(new_total);
            // Build expanded in_shape (left-padded with 1s)
            std::vector<int> bcast_in_shape(out_ndim, 1);
            for (int d = 0; d < in_ndim; ++d)
                bcast_in_shape[out_ndim - in_ndim + d] = in_shape[d];
            // Build input strides (treating broadcast dims as stride=0)
            std::vector<int> in_stride(out_ndim, 0);
            {
                int s = 1;
                for (int d = out_ndim - 1; d >= 0; --d) {
                    in_stride[d] = (bcast_in_shape[d] == 1) ? 0 : s;
                    s *= bcast_in_shape[d];
                }
            }
            // Build output strides
            std::vector<int> out_stride(out_ndim, 1);
            for (int d = out_ndim - 2; d >= 0; --d)
                out_stride[d] = out_stride[d + 1] * view_shape[d + 1];
            // Fill output
            for (int idx = 0; idx < new_total; ++idx) {
                int in_idx = 0;
                int tmp = idx;
                for (int d = out_ndim - 1; d >= 0; --d) {
                    int coord = tmp % view_shape[d];
                    tmp /= view_shape[d];
                    in_idx += coord * in_stride[d];
                }
                out->value[idx] = data_ptr->value[in_idx];
            }
            broadcast_total_size += new_total;
        }
        out->size_ = view_shape;
        output_data.push_back(out);
    }
    return {output_data, latency_cal(broadcast_total_size)};
}

// ── pad ──────────────────────────────────────────────────────────────────────
// Pads num_parts zeros along 'dim' (appended at the end of that dimension).
// Negative dim is resolved Python-style: -1 = last dim.
// Latency: proportional to total output size.
template <typename T>
std::pair<std::vector<LLM_SIM_DATA::DataPtr_t<T>>, uint64_t> TMUOp<T>::pad(
        const std::vector<LLM_SIM_DATA::DataPtr_t<T>>& input_data,
        const int dim, const int num_parts) const {
    std::vector<LLM_SIM_DATA::DataPtr_t<T>> output_data;
    int total_out = 0;

    for (const auto& data_ptr : input_data) {
        int ndim = static_cast<int>(data_ptr->size_.size());
        const int actual_dim = (dim < 0) ? ndim + dim : dim;

        // Output shape: same as input but actual_dim enlarged by num_parts
        std::vector<int> out_shape = data_ptr->size_;
        out_shape[actual_dim] += num_parts;

        int out_total = 1;
        for (int s : out_shape) out_total *= s;

        // Output strides (row-major)
        std::vector<int> out_stride(ndim, 1);
        for (int d = ndim - 2; d >= 0; d--)
            out_stride[d] = out_stride[d + 1] * out_shape[d + 1];

        auto out = std::make_shared<LLM_SIM_DATA::Data_t<T>>();
        out->size_  = out_shape;
        out->value.assign(out_total, T(0));  // zero-initialise (pads included)
        out->elem_bits = data_ptr->elem_bits;  // see scatter()'s doc comment

        // Copy input values; pad positions stay 0
        int in_total = static_cast<int>(data_ptr->value.size());
        for (int idx = 0; idx < in_total; idx++) {
            // Decompose input flat index to multi-dim coordinates
            std::vector<int> coord(ndim);
            int tmp = idx;
            for (int d = ndim - 1; d >= 0; d--) {
                coord[d] = tmp % data_ptr->size_[d];
                tmp      /= data_ptr->size_[d];
            }
            // Same coordinates map to the same position in the output
            // (padding is appended, so original indices are unchanged)
            int out_idx = 0;
            for (int d = 0; d < ndim; d++) out_idx += coord[d] * out_stride[d];
            out->value[out_idx] = data_ptr->value[idx];
        }

        output_data.push_back(out);
        total_out += out_total;
    }
    return {output_data, latency_cal(total_out)};
}

template <typename T>
uint64_t TMUOp<T>::latency_cal(const int data_size) const {
    // Guarantee at least 1 cycle so SC wait(latency) never gets 0.
    const uint64_t cycles = static_cast<uint64_t>(data_size) / LLM_SIM_TMU::TMU_BANDWIDTH;
    return cycles > 0 ? cycles : 1;
}

// ── gather ────────────────────────────────────────────────────────────────────
// Index-select (gather) along a given dimension.
//
//   input_data[0]: source tensor with shape [d0, d1, ..., N, ..., dK] (N at axis 'dim')
//   input_data[1]: index tensor — flat vector of K entries stored as float-cast ints
//   dim          : axis along which to gather
//
//   output shape : source shape with the dim-axis replaced by K (= input_data[1]->value.size())
//   output[..., i, ...] = source[..., int(indices[i]), ...]
//
// Memory is treated as a linearised C-order array throughout.

template <typename T>
std::pair<std::vector<LLM_SIM_DATA::DataPtr_t<T>>, uint64_t>
TMUOp<T>::gather(const std::vector<LLM_SIM_DATA::DataPtr_t<T>>& input_data, const int dim) const {
    if (input_data.size() < 2)
        throw std::runtime_error("TMUOp::gather: need 2 inputs (source and indices)");

    const auto& src  = input_data[0];
    const auto& idx  = input_data[1];
    const std::vector<int>& src_shape = src->size_;
    const int ndim = static_cast<int>(src_shape.size());

    if (dim < 0 || dim >= ndim)
        throw std::runtime_error("TMUOp::gather: dim out of range");

    const int K = static_cast<int>(idx->value.size());  // number of selected indices

    // Build output shape: same as src but replace dim-axis with K
    std::vector<int> out_shape = src_shape;
    out_shape[dim] = K;

    // Compute total elements
    int total_out = 1;
    for (int d : out_shape) total_out *= d;

    // Strides for source (C-order)
    std::vector<int> strides(ndim, 1);
    for (int d = ndim - 2; d >= 0; --d)
        strides[d] = strides[d + 1] * src_shape[d + 1];

    // outer: product of dimensions before 'dim'
    int outer = 1;
    for (int d = 0; d < dim; ++d) outer *= src_shape[d];

    // inner: product of dimensions after 'dim'
    int inner = 1;
    for (int d = dim + 1; d < ndim; ++d) inner *= src_shape[d];

    auto output = std::make_shared<LLM_SIM_DATA::Data_t<T>>();
    output->id    = src->id;
    output->size_ = out_shape;
    output->value.resize(total_out);
    output->elem_bits = src->elem_bits;  // see scatter()'s doc comment (top of file)

    for (int o = 0; o < outer; ++o) {
        for (int i = 0; i < K; ++i) {
            const int src_idx = static_cast<int>(static_cast<float>(idx->value[i]));
            const int src_base = o * src_shape[dim] * inner + src_idx * inner;
            const int dst_base = o * K * inner + i * inner;
            for (int j = 0; j < inner; ++j)
                output->value[dst_base + j] = src->value[src_base + j];
        }
    }

    return {{output}, latency_cal(total_out)};
}


// ─── latency_only ─────────────────────────────────────────────────────────────
// Returns the same latency as execute() but does NO data movement — for use in
// compute_=false (transaction-only) mode.  Calls latency_cal() with sizes only.

template <typename T>
uint64_t TMUOp<T>::latency_only(
    const LLM_SIM_INST::TMUInstPtr_t& inst_ptr,
    const std::vector<LLM_SIM_DATA::DataPtr_t<T>>& input_data) const
{
    int total_in = 0;
    for (const auto& d : input_data)
        if (d) total_in += static_cast<int>(d->value.size());

    using TM = LLM_SIM_INST::TMUInst_t::TMUMode_t;
    switch (inst_ptr->mode) {
        case TM::scatter:
            return latency_cal(total_in * inst_ptr->num_parts);
        case TM::split:
            return latency_cal(total_in);
        case TM::concat:
            return latency_cal(total_in);
        case TM::trans:
            return latency_cal(total_in);
        case TM::view: {
            int new_total = 1;
            for (int s : inst_ptr->view_shape) new_total *= s;
            // Mirror execute()'s view(): pure reshapes (in_total == new_total)
            // are metadata-only and free; only broadcast expansions bill
            // proportional to output size.
            int broadcast_total_size = 0;
            for (const auto& d : input_data) {
                if (!d) continue;
                const int in_total = static_cast<int>(d->value.size());
                if (in_total != new_total) broadcast_total_size += new_total;
            }
            return latency_cal(broadcast_total_size);
        }
        case TM::pad: {
            // Compute padded output size from shapes
            int total_out = 0;
            for (const auto& d : input_data) {
                if (!d) continue;
                int ndim = static_cast<int>(d->size_.size());
                if (ndim == 0) { total_out += (int)d->value.size(); continue; }
                const int adim = (inst_ptr->dim < 0) ? ndim + inst_ptr->dim : inst_ptr->dim;
                int sz = 1;
                for (int i = 0; i < ndim; ++i)
                    sz *= (i == adim) ? d->size_[i] + inst_ptr->num_parts : d->size_[i];
                total_out += sz;
            }
            return latency_cal(total_out > 0 ? total_out : total_in);
        }
        case TM::gather: {
            if (input_data.size() < 2 || !input_data[0] || !input_data[1])
                return latency_cal(total_in);
            const auto& src = input_data[0];
            const auto& idx = input_data[1];
            const int ndim = static_cast<int>(src->size_.size());
            const int K    = static_cast<int>(idx->value.size());
            const int adim = (inst_ptr->dim < 0) ? ndim + inst_ptr->dim : inst_ptr->dim;
            if (adim < 0 || adim >= ndim) return latency_cal(total_in);
            int outer = 1; for (int d = 0;      d < adim; ++d) outer *= src->size_[d];
            int inner = 1; for (int d = adim+1; d < ndim; ++d) inner *= src->size_[d];
            return latency_cal(outer * K * inner);
        }
        default:
            return 1;
    }
}

// ─── output_meta ──────────────────────────────────────────────────────────────
// Per-output {shape, elem_bits} metadata, mirroring each mode's execute()
// implementation's own out->size_/elem_bits assignment exactly, without
// moving any data. Used by TMU.cpp's transaction-only path so a run with
// compute_=false tags Cache write requests with the same shape/elem_bits a
// compute_=true run would.
template <typename T>
std::vector<typename TMUOp<T>::OutputMeta> TMUOp<T>::output_meta(
    const LLM_SIM_INST::TMUInstPtr_t& inst_ptr,
    const std::vector<LLM_SIM_DATA::DataPtr_t<T>>& input_data) const
{
    using TM = LLM_SIM_INST::TMUInst_t::TMUMode_t;
    std::vector<OutputMeta> out;

    switch (inst_ptr->mode) {
        case TM::scatter: {
            const int num_parts = inst_ptr->num_parts;
            for (const auto& data_ptr : input_data) {
                std::vector<int> out_shape = data_ptr->size_;
                out_shape[0] = out_shape[0] / num_parts;
                for (int p = 0; p < num_parts; ++p)
                    out.push_back({out_shape, data_ptr->elem_bits});
            }
            break;
        }
        case TM::split: {
            const int num_parts = inst_ptr->num_parts;
            const int dim = inst_ptr->dim;
            for (const auto& data_ptr : input_data) {
                int ndim = static_cast<int>(data_ptr->size_.size());
                const int actual_dim = (dim < 0) ? ndim + dim : dim;
                int dim_size = data_ptr->size_[actual_dim];
                int part_size = dim_size / num_parts;
                std::vector<int> out_shape = data_ptr->size_;
                // Degenerate split (part_size==0): execute() replicates the
                // full input num_parts times instead, preserving shape.
                if (part_size != 0) out_shape[actual_dim] = part_size;
                for (int p = 0; p < num_parts; ++p)
                    out.push_back({out_shape, data_ptr->elem_bits});
            }
            break;
        }
        case TM::concat: {
            if (input_data.empty()) break;
            int ndim = static_cast<int>(input_data[0]->size_.size());
            const int dim = inst_ptr->dim;
            const int actual_dim = (dim < 0) ? ndim + dim : dim;
            std::vector<int> out_shape = input_data[0]->size_;
            for (size_t k = 1; k < input_data.size(); ++k)
                out_shape[actual_dim] += input_data[k]->size_[actual_dim];
            bool all_same = true;
            for (const auto& dp : input_data)
                if (dp->elem_bits != input_data[0]->elem_bits) { all_same = false; break; }
            const int elem_bits = all_same ? input_data[0]->elem_bits
                                            : static_cast<int>(sizeof(T) * 8);
            out.push_back({out_shape, elem_bits});
            break;
        }
        case TM::trans: {
            for (const auto& data_ptr : input_data) {
                int ndim = static_cast<int>(data_ptr->size_.size());
                std::vector<int> out_shape = data_ptr->size_;
                std::swap(out_shape[ndim - 2], out_shape[ndim - 1]);
                out.push_back({out_shape, data_ptr->elem_bits});
            }
            break;
        }

        case TM::view: {
            for (const auto& data_ptr : input_data)
                out.push_back({inst_ptr->view_shape, data_ptr->elem_bits});
            break;
        }
        case TM::pad: {
            const int dim = inst_ptr->dim;
            const int num_parts = inst_ptr->num_parts;
            for (const auto& data_ptr : input_data) {
                int ndim = static_cast<int>(data_ptr->size_.size());
                const int actual_dim = (dim < 0) ? ndim + dim : dim;
                std::vector<int> out_shape = data_ptr->size_;
                out_shape[actual_dim] += num_parts;
                out.push_back({out_shape, data_ptr->elem_bits});
            }
            break;
        }
        case TM::gather: {
            if (input_data.size() < 2) break;
            const auto& src = input_data[0];
            const auto& idx = input_data[1];
            const int dim = inst_ptr->dim;
            std::vector<int> out_shape = src->size_;
            out_shape[dim] = static_cast<int>(idx->value.size());
            out.push_back({out_shape, src->elem_bits});
            break;
        }
        default:
            break;
    }
    return out;
}

template class TMUOp<int>;
template class TMUOp<float>;
