#ifndef LLM_SIM_REQ_H
#define LLM_SIM_REQ_H

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <cstdint>

#include "utils/BaseInst.h"

namespace LLM_SIM_DRAM {

class DRAMAddr_t {
    public:
        // row_stride: total number of columns per row in the file (used as the byte-offset
        // multiplier: element (r, c) is at byte offset (r * row_stride + c) * sizeof(T)).
        // When omitted, row_stride defaults to col_end (backward-compatible: the access
        // covers [col_start, col_end) which is also the full row width).
        // pad_to: if > (col_end - col_start), the returned data vector is zero-padded
        // to this length per row.  Latency is still computed from the actual read size.
        // Use for TP sharding where shard_dim % q_block_size != 0 (e.g. shard_dim=720,
        // pad_to=736 so Q8_0 block boundaries align).  Default 0 = no padding.
        DRAMAddr_t(const std::string& path,
                 const int64_t row_start, const int64_t row_end,
                 const int64_t col_start, const int64_t col_end,
                   const uint64_t row_stride = 0,
                   const int64_t  pad_to     = 0)
            : file_path(path), row_start(row_start), row_end(row_end),
              col_start(col_start), col_end(col_end),
              row_stride(row_stride > 0 ? row_stride : static_cast<uint64_t>(col_end > 0 ? col_end : 0)),
              pad_to(pad_to) {}

        const std::string file_path;
          const int64_t     row_start, row_end;  // row indices [row_start, row_end) to access
          const int64_t     col_start, col_end;  // col indices [col_start, col_end) to access
        const uint64_t    row_stride;           // total cols in file; element (r,c) at offset (r*row_stride+c)*sizeof(T)
        const int64_t     pad_to;              // pad output to this many cols per row (0 = no pad)
};
using DRAMAddrPtr_t = std::shared_ptr<DRAMAddr_t>;


class Req_t: public LLM_SIM_INST::BaseInst_t {
    public:
        explicit Req_t(DRAMAddrPtr_t addr_ptr, bool is_read) :
        addr_ptr(addr_ptr),
        data_shape({static_cast<uint64_t>((addr_ptr->row_end > addr_ptr->row_start) ? (addr_ptr->row_end - addr_ptr->row_start) : 0),
                    static_cast<uint64_t>(
                        addr_ptr->pad_to > 0
                        ? addr_ptr->pad_to
                        : ((addr_ptr->col_end > addr_ptr->col_start) ? (addr_ptr->col_end - addr_ptr->col_start) : 0))}),
        len(data_shape[0] * data_shape[1]),
        is_read(is_read)
        { LLM_SIM_INST::BaseInst_t::type_ = LLM_SIM_INST::InstType_t::DRAM; }
        uint64_t tid = 0; // transaction id for tracking request-response pairs

        // Phase C: which core should actually
        // service this read, set by DMA::try_dispatch once it has resolved
        // (via DataManager::find_resident_group_member()) which group
        // member currently holds this path resident. -1 (default) means
        // "not set" -- DRAMRouter falls back to a fresh
        // AddressMapper::owner_of() lookup for callers that bypass DMA
        // (direct DRAMRouter unit tests). Mutable like tid above (the only
        // other non-const members here) -- everything else is fixed at
        // construction.
        int target_core = -1;

        const DRAMAddrPtr_t addr_ptr; // address for accessing value
        const std::vector<uint64_t> data_shape; // {num_rows, num_cols} of the accessed submatrix
        const uint64_t len;      // total elements = num_rows * num_cols; used for latency estimation
        const bool is_read;      // true for read request, false for write request
};

using ReqPtr_t = std::shared_ptr<Req_t>;
}

#endif