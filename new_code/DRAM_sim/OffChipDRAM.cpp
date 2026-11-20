#include "DRAM_sim/OffChipDRAM.h"
#include "DRAM_sim/DataManager.h"
#include "utils/precision_utils.h"

#include <filesystem>
#include <fstream>
#include <iostream>

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
template <typename T>
OffChipDRAM<T>::OffChipDRAM(sc_module_name name, int node_id, bool compute)
    : sc_module(name), node_id_(node_id), compute_(compute)
{
    SC_THREAD(run);
    sensitive << clk.pos();
}

// ---------------------------------------------------------------------------
// read_data / write_data
//   Same byte-offset technique as DataManager<T>::read_data/write_data
//   (element (r,c) at byte offset (r*row_stride+c)*sizeof(T)), but WITHOUT
//   a per-file mutex: this module is driven by a single SC_THREAD, so there
//   is no concurrent host-level access to guard against within one instance
//   (unlike local DataManager, which is a singleton shared by 64 cores'
//   worth of DMA/Array traffic).
// ---------------------------------------------------------------------------
template <typename T>
std::vector<T> OffChipDRAM<T>::read_data(const LLM_SIM_DRAM::DRAMAddrPtr_t& addr_ptr) {
    const uint64_t n_rows = (addr_ptr->row_end > addr_ptr->row_start)
        ? static_cast<uint64_t>(addr_ptr->row_end - addr_ptr->row_start) : 0;
    const uint64_t n_cols = (addr_ptr->col_end > addr_ptr->col_start)
        ? static_cast<uint64_t>(addr_ptr->col_end - addr_ptr->col_start) : 0;
    if (n_rows == 0 || n_cols == 0) return {};

    if (!compute_) {
        const uint64_t out_cols = (addr_ptr->pad_to > 0 && static_cast<uint64_t>(addr_ptr->pad_to) > n_cols)
            ? static_cast<uint64_t>(addr_ptr->pad_to) : n_cols;
        return std::vector<T>(n_rows * out_cols, T{});
    }

    const std::string& path = addr_ptr->file_path;
    const int64_t row_start = addr_ptr->row_start;
    const int64_t row_end   = addr_ptr->row_end;
    const int64_t col_start = addr_ptr->col_start;
    const uint64_t row_stride = addr_ptr->row_stride;

    std::ifstream fin(path, std::ios::in | std::ios::binary);
    if (!fin.is_open()) {
        std::cerr << "[OffChipDRAM node=" << node_id_ << "] read_data: cannot open file: "
                  << path << std::endl;
        return {};
    }

    std::vector<T> result;
    result.reserve(n_rows * n_cols);

    for (int64_t r = row_start; r < row_end; ++r) {
        if (r < 0 || col_start < 0) {
            result.insert(result.end(), n_cols, static_cast<T>(0));
            continue;
        }
        const uint64_t byte_offset =
            (static_cast<uint64_t>(r) * row_stride + static_cast<uint64_t>(col_start)) * sizeof(T);
        fin.seekg(static_cast<std::streamoff>(byte_offset), std::ios::beg);
        if (!fin) {
            std::cerr << "[OffChipDRAM node=" << node_id_ << "] read_data: seekg failed: "
                      << path << " row=" << r << std::endl;
            return {};
        }
        std::vector<T> row_buf(n_cols);
        fin.read(reinterpret_cast<char*>(row_buf.data()), static_cast<std::streamsize>(n_cols * sizeof(T)));
        if (!fin) {
            std::cerr << "[OffChipDRAM node=" << node_id_ << "] read_data: read failed: "
                      << path << " row=" << r << std::endl;
            return {};
        }
        result.insert(result.end(), row_buf.begin(), row_buf.end());

        if (addr_ptr->pad_to > 0 && addr_ptr->pad_to > static_cast<int64_t>(n_cols)) {
            const uint64_t pad_elems = static_cast<uint64_t>(addr_ptr->pad_to) - n_cols;
            result.insert(result.end(), pad_elems, static_cast<T>(0));
        }
    }
    return result;
}

template <typename T>
void OffChipDRAM<T>::write_data(const LLM_SIM_DRAM::DRAMAddrPtr_t& addr_ptr,
                                 const std::vector<T>& data) {
    if (data.empty()) return;
    if (!compute_) return;

    const std::string& path = addr_ptr->file_path;
    const int64_t row_start = addr_ptr->row_start;
    const int64_t row_end   = addr_ptr->row_end;
    const int64_t col_start = addr_ptr->col_start;
    const int64_t col_end   = addr_ptr->col_end;
    const uint64_t row_stride = addr_ptr->row_stride;
    const uint64_t n_rows = (row_end > row_start) ? static_cast<uint64_t>(row_end - row_start) : 0;
    const uint64_t n_cols = (col_end > col_start) ? static_cast<uint64_t>(col_end - col_start) : 0;
    if (n_rows == 0 || n_cols == 0) return;

    std::error_code ec;
    std::filesystem::path fs_path(path);
    auto parent = fs_path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent, ec);
    }

    std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
    if (!f.is_open()) {
        { std::ofstream tmp(path, std::ios::out | std::ios::binary); }
        f.open(path, std::ios::in | std::ios::out | std::ios::binary);
    }
    if (!f.is_open()) {
        std::cerr << "[OffChipDRAM node=" << node_id_ << "] write_data: cannot open file: "
                  << path << std::endl;
        return;
    }

    for (uint64_t r = 0; r < n_rows; ++r) {
        const int64_t logical_row = row_start + static_cast<int64_t>(r);
        if (logical_row < 0 || col_start < 0) continue;
        if ((r + 1) * n_cols > data.size()) {
            std::cerr << "[OffChipDRAM node=" << node_id_ << "] write_data: data size mismatch: "
                      << path << " row=" << logical_row << std::endl;
            continue;
        }
        const uint64_t byte_offset =
            (static_cast<uint64_t>(logical_row) * row_stride + static_cast<uint64_t>(col_start)) * sizeof(T);
        f.seekp(static_cast<std::streamoff>(byte_offset), std::ios::beg);
        if (!f) {
            std::cerr << "[OffChipDRAM node=" << node_id_ << "] write_data: seekp failed: "
                      << path << " row=" << logical_row << std::endl;
            return;
        }
        f.write(reinterpret_cast<const char*>(data.data() + r * n_cols),
                static_cast<std::streamsize>(n_cols * sizeof(T)));
        if (!f) {
            std::cerr << "[OffChipDRAM node=" << node_id_ << "] write_data: write failed: "
                      << path << " row=" << logical_row << std::endl;
            return;
        }
    }
    f.flush();
}

// ---------------------------------------------------------------------------
// get_offchip_latency
// ---------------------------------------------------------------------------
template <typename T>
uint64_t OffChipDRAM<T>::get_offchip_latency() const {
    return static_cast<uint64_t>(LLM_SIM_DRAM::OFFCHIP_DRAM_OVERHEAD);
}

// ---------------------------------------------------------------------------
// do_read / do_write
// ---------------------------------------------------------------------------
template <typename T>
void OffChipDRAM<T>::do_read(const LLM_SIM_DRAM::ReqPtr_t& req) {
    std::vector<T> data = read_data(req->addr_ptr);

    // Phase E: data_manager_ is optional here
    // (format lookup only, see setDataManager()'s doc comment) -- falls back
    // to the fp32 default when unset.
    const int elem_bits = data_manager_
        ? data_manager_->elem_bits_for(req->addr_ptr->file_path) : 32;
    const uint64_t latency = get_offchip_latency()
                            + LLM_SIM_PRECISION::transfer_cycles(
                                  req->len, elem_bits,
                                  static_cast<uint64_t>(LLM_SIM_DRAM::OFFCHIP_DRAM_BANDWIDTH));
    wait(latency * sc_time(1, SC_NS));

    auto data_ptr   = std::make_shared<LLM_SIM_DATA::Data_t<T>>();
    data_ptr->id    = req->tid;
    data_ptr->value = std::move(data);
    data_ptr->size_ = {static_cast<int>(req->len)};
    data_ptr->elem_bits = elem_bits;
    while (!read_port.nb_write(data_ptr))
        wait(clk.posedge_event());
}

template <typename T>
void OffChipDRAM<T>::do_write(const LLM_SIM_DRAM::ReqPtr_t& req) {
    LLM_SIM_DATA::DataPtr_t<T> data_ptr;
    while (!write_port.nb_read(data_ptr))
        wait(clk.posedge_event());

    // Commit to disk before the latency wait, same rationale as
    // DRAMSim::do_write (so sc_stop() cannot race the write).
    write_data(req->addr_ptr, data_ptr->value);

    const int elem_bits = data_manager_
        ? data_manager_->elem_bits_for(req->addr_ptr->file_path) : 32;
    const uint64_t latency = get_offchip_latency()
                            + LLM_SIM_PRECISION::transfer_cycles(
                                  req->len, elem_bits,
                                  static_cast<uint64_t>(LLM_SIM_DRAM::OFFCHIP_DRAM_BANDWIDTH));
    wait(latency * sc_time(1, SC_NS));
}

// ---------------------------------------------------------------------------
// run -- single SC_THREAD: poll req_port, dispatch to do_read/do_write.
// No multi-source arbitration needed (see class comment) -- this node has
// exactly one upstream source, its own Depacketizer.
// ---------------------------------------------------------------------------
template <typename T>
void OffChipDRAM<T>::run() {
    while (true) {
        LLM_SIM_DRAM::ReqPtr_t req;
        while (!req_port.nb_read(req))
            wait(clk.posedge_event());
        if (node_id_ == 64) {
            std::cerr << "[DBG2-NODE64-RECV] tid=" << req->tid << " @" << sc_time_stamp()
                       << "\n" << std::flush;
        }

        // Coarse proxy for utilization: counts requests dispatched, not
        // literal per-cycle occupancy (unlike DRAMSim::busy_cycle_, which
        // increments every cycle while any request is pending across a
        // multi-source arbiter/queue). This module has a single source and
        // processes strictly one request at a time end-to-end within this
        // thread, so a per-cycle tracker would need a second thread purely
        // to sample "busy" -- not worth the complexity for a metric no
        // caller currently consumes.
        ++busy_cycle_;
        if (req->is_read) {
            do_read(req);
        } else {
            do_write(req);
        }
    }
}

template class OffChipDRAM<int>;
template class OffChipDRAM<float>;
