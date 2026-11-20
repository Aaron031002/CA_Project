#ifndef LLM_SIM_BASEINST_H
#define LLM_SIM_BASEINST_H

#include <iostream>
#include <memory>
#include <string>
#include <systemc.h>

#include "utils/bufferAddr.h"
// Forward-declare Req_t to break the circular include (Req.h -> BaseInst.h -> Req.h)
#include "NoC_sim/noc_utils.h"

namespace LLM_SIM_DRAM {
    class Req_t;
    using ReqPtr_t = std::shared_ptr<Req_t>;
}

namespace LLM_SIM_INST {
enum InstType_t{
    NONE,
    TCORE,
    ARRAY,
    VECTOR,
    DMA,
    TMU,
    NOC,
    DRAM,
    NUM_TYPES
};

class BaseInst_t {
    public:
        BaseInst_t() : type_(InstType_t::NONE) {}  // Default to NONE
        virtual ~BaseInst_t() = default;  // Virtual destructor for proper cleanup
        InstType_t getType() const { return type_; }
        InstType_t type_;    
};
using BaseInstPtr_t = std::shared_ptr<BaseInst_t>;

class TCoreInst_t: public BaseInst_t {
    public:
        enum TCoreMode_t {
            WAIT,
            LOCAL_SYNC,   // stall until all local units (Array/Vector/DMA/TMU) are idle
            GLOBAL_SYNC,  // distributed barrier: stall until all participant cores arrive
            NUM_MODES
        };

        // Constructor for WAIT and LOCAL_SYNC (no extra parameters)
        explicit TCoreInst_t(TCoreMode_t mode)
            : mode(mode), barrier_id(0), participants({})
        { this->type_ = InstType_t::TCORE; }

        // Constructor for GLOBAL_SYNC
        TCoreInst_t(TCoreMode_t mode, uint64_t bid, std::vector<int> parts)
            : mode(mode), barrier_id(bid), participants(std::move(parts))
        { this->type_ = InstType_t::TCORE; }

        const TCoreMode_t  mode;
        const uint64_t     barrier_id;    // used by GLOBAL_SYNC
        const std::vector<int> participants; // core IDs that must all arrive (GLOBAL_SYNC)
};
using TCoreInstPtr_t = std::shared_ptr<TCoreInst_t>;


// Inherited classes for different instruction types
class ArrayInst_t : public BaseInst_t {
    public:
    // Enum declared first so it can be used in constructor parameters below
        // Note: Every operation only change the size of the last 2 dimensions
        // However, it means the rest of the dimension should be "completely the same !!!!"
        enum ArrayMode_t {
            NORMAL, // (..., M, K) * (..., K, N) -> (..., M, N)
            DOT_PRODUCT, // (..., M, K) * (..., M, N) -> (..., K, N)  (reduction along the M dimension)
            QK, // special GEMM method for QK
            SV, // special GEMM method for SV
            NUM_MODES
        };
        // Systolic array dataflow (§A).
        // AUTO resolves by mode: NORMAL/DOT_PRODUCT -> WEIGHT_STATIONARY
        // (K*N weight tile stays resident, M tokens stream through);
        // QK/SV -> OUTPUT_STATIONARY (K/V differ per token position, nothing
        // can stay resident, so the M*N output tile accumulates while K streams).
        enum Dataflow_t {
            AUTO,
            WEIGHT_STATIONARY,
            OUTPUT_STATIONARY
        };
        // Where ArrayUnit should fetch the weight tile from for this instruction.
        // DRAM (default): unchanged behavior -- ArrayUnit issues req_ptr to DRAM.
        // CACHE: ArrayUnit issues a CacheReq_t(w_id, w_reserved) to the shared
        // Cache instead; req_ptr is still built/carried by the decoder (for
        // row/col shape derivation) but unused for the actual fetch.
        enum WeightSrc_t {
            DRAM,
            CACHE
        };
        // kv_head_num: number of K/V heads for GQA; -1 means same as Q head count (non-GQA)
        // q_seq_s:     global Q token start position (for causal mask in QK); -1 = no mask
        // k_seq_s_real: global K token start position for real K (position 1 in K file, pos 0 is sink)
        // win_size:    sliding window size for causal mask; -1 = no window constraint
        // has_sink:    if true, SV scores have an extra leading column (the sink) to skip
        // q_head_num:  SV: total number of Q heads per token (for multi-token GQA kh mapping); -1 = infer from old formula
        // dataflow:    systolic array dataflow override; AUTO = infer from mode (see Dataflow_t above)
        // weight_src:  DRAM (default) or CACHE -- see WeightSrc_t above
        // w_id/w_reserved: cache data id / evict-after-read flag, only used when weight_src == CACHE
        explicit ArrayInst_t(ArrayMode_t mode, int row, int col, LLM_SIM_DRAM::ReqPtr_t req_ptr, uint64_t in_id, uint64_t out_id, bool in_reserved = false, int kv_head_num = -1,
                             int q_seq_s = -1, int k_seq_s_real = -1, int win_size = -1, bool has_sink = false, int q_head_num = -1, Dataflow_t dataflow = AUTO,
                             WeightSrc_t weight_src = DRAM, uint64_t w_id = 0, bool w_reserved = false, uint64_t new_in_id = 0, uint64_t new_w_id = 0) :
        req_ptr(req_ptr), row(row), col(col), mode(mode), in_id(in_id), out_id(out_id), in_reserved(in_reserved), kv_head_num(kv_head_num),
        q_seq_s(q_seq_s), k_seq_s_real(k_seq_s_real), win_size(win_size), has_sink(has_sink), q_head_num(q_head_num), dataflow(dataflow),
        weight_src(weight_src), w_id(w_id), w_reserved(w_reserved), new_in_id(new_in_id), new_w_id(new_w_id) {
            this->type_ = InstType_t::ARRAY;
        }
        /*
        explicit ArrayInst_t(ArrayMode_t mode, int row, int col, LLM_SIM_DRAM::ReqPtr_t req_ptr) : req_ptr(req_ptr), row(row), col(col), mode(mode),id(id) {
            this->type_ = InstType_t::ARRAY;
        }
        */
        const LLM_SIM_DRAM::ReqPtr_t req_ptr; // optional, only used for ARRAY instructions that directly access DRAM
        const int row;
        const int col;
        const ArrayMode_t mode;
        const uint64_t in_id, out_id;
        const bool in_reserved;
        const int kv_head_num; // -1 = non-GQA (kv_heads == q_heads); >0 = GQA kv head count
        const int q_seq_s;      // QK: global Q token start for causal mask (-1 = disabled)
        const int k_seq_s_real; // QK: global K start of real tokens (pos 1 in K file)
        const int win_size;     // QK: sliding window size (-1 = disabled)
        const bool has_sink;    // SV: if true, first score column is sink and should be skipped
        const int q_head_num;   // SV: total Q heads per token for multi-token GQA kh mapping (-1 = use old formula)
        const Dataflow_t dataflow; // systolic array dataflow; AUTO = infer from mode
        const WeightSrc_t weight_src; // DRAM (default) or CACHE -- see WeightSrc_t above
        const uint64_t w_id;       // cache data id for the weight tile, only used when weight_src == CACHE
        const bool w_reserved;     // evict weight from cache after read, only used when weight_src == CACHE
        // Rename-on-terminal-read (0 = no rename, today's behavior unchanged).
        // Only meaningful together with in_reserved/w_reserved == true: Cache
        // re-keys the same physical entry to this id instead of erasing it.
        const uint64_t new_in_id;
        const uint64_t new_w_id;

        // Resolve AUTO to a concrete dataflow based on mode.
        Dataflow_t resolved_dataflow() const {
            if (dataflow != AUTO) return dataflow;
            switch (mode) {
                case NORMAL:
                case DOT_PRODUCT:
                    return WEIGHT_STATIONARY;
                case QK:
                case SV:
                    return OUTPUT_STATIONARY;
                default:
                    return WEIGHT_STATIONARY;
            }
        }
};
using ArrayInstPtr_t = std::shared_ptr<ArrayInst_t>;


// VectorInst
class VectorInst_t: public BaseInst_t {
    public:
        // Enum declared first so it can be used in constructor parameters below
        enum VectorMode_t {
            IDLE = 0,
            // Element-wise arithmetic
            ADD = 1,        // element-wise addition
            SUB = 2,        // element-wise subtraction
            MUL = 3,        // element-wise multiplication
            RECIP = 4,      // reciprocal: 1/x
            INV_SQRT = 5,   // inverse square root: 1/sqrt(x)
            // Activation functions
            RELU = 6,       // ReLU: max(0, x)
            SIG = 7,        // sigmoid: 1/(1+exp(-x))
            SILU = 8,       // SiLU: x * sigmoid(x)
            SWIGLU = 9,     // SwiGLU: SiLU(in[i]) * w[i] (in=activation, w=gate); same shape as in
            // Normalisation / transform
            RMS = 10,       // RMSNorm along the dim dimension
            SOFTMAX = 11,   // Softmax along the dim dimension
            ROPE = 12,      // Rotary position embedding
            // Reduction
            ADD_REDUCE = 13,  // if all: sum all elements → scalar; else sum along axis dim
            DOT_PRODUCT = 14, // if all: dot product of in and w → scalar; else dot along axis dim
            TOP_K = 15,       // argsort descending along dim → index output
            DIV_CONSTANT = 16, // element-wise division by a pre-defined constant: out[i] = in[i] / DIV_CONSTANT_VALUE
            BIAS_ADD = 17,     // broadcast bias add: out[r*dim+d] = in[r*dim+d] + bias[d], bias shape=[dim], input shape=[...,dim]
            SPARSE_GATE = 18,  // sparse MoE gate: topk_idx[chunk*top_k] + topk_val[chunk*top_k] → gate[chunk*n_experts]; dim=n_experts, out_id2=top_k
            // Q8_0 quantize+dequantize round-trip (§3):
            // per-32-element-block symmetric int8 quantization, scale stored as
            // F16. Models the activation-precision loss real gpt-oss-20b incurs
            // before each MXFP4-weight expert matmul (ggml_vec_dot_mxfp4_q8_0).
            // Output is still full-precision T (the "value" changes, not the
            // storage type) -- no weight input needed.
            QUANTIZE_DEQUANT_Q8_0 = 19,
            // Scalar max(abs) over the first `dim` elements → 1-element output.
            // Used for cross-shard Q8_0 scale synchronisation: each shard
            // computes its local max(abs(swiglu)), exchanges scalars with
            // neighbours, then calls QUANTIZE_DEQUANT_Q8_0_EXT_SCALE.
            REDUCE_MAX_ABS = 20,
            // Q8_0 round-trip with an externally-supplied scale.
            // w_id must point to a 1-element Data_t whose value[0] is the
            // global max_abs (= global_scale * 127).  All 32-element blocks
            // use this single scale instead of computing their own.
            QUANTIZE_DEQUANT_Q8_0_EXT_SCALE = 21,
            // Q8_0 round-trip with standard per-block scale EXCEPT the last block,
            // whose scale is max(local_last_block_max_abs, ext_scale_vec[0]).
            // Used by TP-4 shards to fix the cross-shard boundary block.
            QUANTIZE_DEQUANT_Q8_0_BOUNDARY_FIX = 22,
            NUM_OPS = 23,
        };
        // General constructor (all modes except ROPE).
        // out_id2: second output id used by TOP_K (values output); 0 = not used
        explicit VectorInst_t(VectorMode_t mode, int dim, uint64_t in_id, uint64_t w_id, uint64_t out_id, bool in_reserved = false, bool w_reserved = false, bool all = false, uint64_t out_id2 = 0, uint64_t new_in_id = 0, uint64_t new_w_id = 0) :
        dim(dim), all(all), mode(mode), in_id(in_id), w_id(w_id), out_id(out_id), in_reserved(in_reserved), w_reserved(w_reserved), out_id2(out_id2), rope_token_pos(0), rope_theta(10000.0f), new_in_id(new_in_id), new_w_id(new_w_id) {
            this->type_ = InstType_t::VECTOR;
        }
        // ROPE constructor: no weight needed; cos/sin computed on-the-fly from token_pos & head_dim (=dim).
        // Format: Vector ROPE <head_dim> <token_pos> <x_id> <out_id> <x_reserved> [theta]
        explicit VectorInst_t(VectorMode_t mode, int head_dim, int token_pos, uint64_t in_id, uint64_t out_id, bool in_reserved = false, float rope_theta = 10000.0f, uint64_t new_in_id = 0) :
        dim(head_dim), all(false), mode(mode), in_id(in_id), w_id(0), out_id(out_id), in_reserved(in_reserved), w_reserved(false), out_id2(0), rope_token_pos(token_pos), rope_theta(rope_theta), new_in_id(new_in_id), new_w_id(0) {
            this->type_ = InstType_t::VECTOR;
        }
        const int dim;
        const bool all;
        const bool in_reserved, w_reserved; // for checking if the data in the cache should be cleaned
        // Rename-on-terminal-read (0 = no rename). Only meaningful together
        // with in_reserved/w_reserved == true, respectively.
        const uint64_t new_in_id, new_w_id;
        const VectorMode_t mode;
        const uint64_t in_id, w_id, out_id;
        const uint64_t out_id2; // second output (indices DataPtr for TOP_K values); 0 = unused
        const int rope_token_pos; // ROPE only: token position for computing rotary frequencies on-the-fly
        const float rope_theta;    // ROPE only: RoPE base frequency (default 10000.0f)
        
};
using VectorInstPtr_t = std::shared_ptr<VectorInst_t>;

// NoC Inst for Router
class NoCInst_t: public BaseInst_t {
    public:
        // Send instruction: pack data and route src → dst_ids.
        // dst_buf becomes packet.payload.dst_addr_ptr so the receiver knows
        // where to deposit the data in its local buffer hierarchy.
        explicit NoCInst_t(int src_id, std::vector<int> dst_ids, uint64_t packet_id,
                           LLM_SIM_NOC::CastType cast_type = LLM_SIM_NOC::CastType::UNICAST,
                           uint64_t op_id = 0)
            : src_id(src_id), dst_ids(std::move(dst_ids)), packet_id(packet_id),
              id(0), op_id(op_id), cast_type(cast_type) {
            this->type_ = InstType_t::NOC;
        }
        // Recv-only instruction: packet_id selects the incoming packet from the pool;
        // id is stamped onto the output DataPtr_t so the local core can identify
        // which data buffer this delivery targets.
        explicit NoCInst_t(uint64_t packet_id, uint64_t id, std::vector<int> data_size)
            : src_id(-1), dst_ids({}), packet_id(packet_id), data_size(data_size),
              id(id), cast_type(LLM_SIM_NOC::CastType::UNICAST) {
            this->type_ = InstType_t::NOC;
        }

        // NOC-specific members
        const int src_id;
        const std::vector<int> dst_ids;
        const uint64_t packet_id;
        const uint64_t id; // DataPtr_t id
        const uint64_t op_id = 0; // dynamic-packet operation id
        const std::vector<int> data_size;
        // const LLM_SIM_BUF::BaseAddrPtr_t dst_buf; // send: payload.dst_addr_ptr
        const LLM_SIM_NOC::CastType cast_type;

        // Phase 7b: off-chip DRAM fetch/write-back request
        // riding inside a Packet_t addressed to an off-chip node (dst_ids =
        // {offchip_node_id}). Mutable: DMA constructs the NoCInst_t first via
        // the normal constructor, then fills this in before handing it to
        // noc_inst_out. nullptr for every ordinary send (zero effect on
        // existing traffic). Router::data2packet copies this straight into
        // Packet_t<T>::offchip_dram_req (see noc_utils.h).
        mutable LLM_SIM_DRAM::ReqPtr_t offchip_dram_req = nullptr;
};
using NoCInstPtr_t = std::shared_ptr<NoCInst_t>;


class TMUInst_t: public BaseInst_t {
    public:
        enum class TMUMode_t {
            concat,
            scatter,
            split,
            trans,
            view,
            pad, 
            gather,  // index-select: output[...,i,...] = src[...,indices[i],...] along dim
            // gather, 
            NUM_MODES
        };
        // General constructor (scatter / split / concat / trans).
        // num_parts: used by scatter / split (and by Decoder for backward-compat).
        // dim:       used by concat / split; negative = Python-style reverse index.
        //            -1 means "not applicable" or "default to dim 0".
        explicit TMUInst_t(const std::vector<uint64_t>& input_ids,
                           const std::vector<bool>& input_reserved,
                           const std::vector<uint64_t>& output_ids,
                           TMUMode_t mode,
                           int num_parts = -1,
                           int dim = -1,
                           const std::vector<uint64_t>& new_input_ids = {})
            : input_ids(input_ids), input_reserved(input_reserved), output_ids(output_ids),
              mode(mode), dim(dim), view_shape(), num_parts(num_parts), new_input_ids(new_input_ids) {
            this->type_ = InstType_t::TMU;
        }
        // Constructor for view mode: target shape must have the same total
        // element count as each input tensor.
        explicit TMUInst_t(const std::vector<uint64_t>& input_ids,
                           const std::vector<bool>& input_reserved,
                           const std::vector<uint64_t>& output_ids,
                           TMUMode_t mode,
                           std::vector<int> view_shape,
                           const std::vector<uint64_t>& new_input_ids = {})
            : input_ids(input_ids), input_reserved(input_reserved), output_ids(output_ids),
              mode(mode), dim(-1), view_shape(std::move(view_shape)), num_parts(-1), new_input_ids(new_input_ids) {
            this->type_ = InstType_t::TMU;
        }
        const std::vector<uint64_t> input_ids; // the data ids of the input
        const std::vector<bool> input_reserved; // for checking if the data in the cache should be cleaned. We can also track the reserved state of each buffer separately if needed.
        const std::vector<uint64_t> output_ids; // the data ids of the output
        const TMUMode_t mode;
        const int dim; // along with the designated dimension
        const std::vector<int> view_shape; // only used for view mode to indicate the shape of the view. The total size should match the size of the input data
        const int num_parts;
        // Rename-on-terminal-read, sparse and parallel to input_ids: empty =
        // no input renames at all; otherwise same length as input_ids, with
        // 0 at index j meaning "no rename for input j". Only meaningful for
        // an index whose input_reserved[j] == true.
        const std::vector<uint64_t> new_input_ids;
        // const bool from_dma;
    };
using TMUInstPtr_t = std::shared_ptr<TMUInst_t>;



// Forward declarations so DMAInst_t can hold CacheReqPtr_t
class CacheReq_t;
using CacheReqPtr_t = std::shared_ptr<CacheReq_t>;

// DMAInst 
// Warning: Integrate DRAM and NoC instruction in this version
class DMAInst_t: public BaseInst_t {
    public:
        // For DMA Normal instructions that do not directly correspond to DRAM accesses and we do not need to track them with a Req_t, we can use this constructor
        explicit DMAInst_t(const LLM_SIM_INST::NoCInstPtr_t noc_inst_ptr, const uint64_t data_id, const int length, const int elem_bits = 32, bool truncate = false, uint64_t new_data_id = 0) :
        is_noc(true), is_dram(false), is_cache(false), is_cache_copy(false), is_cache_mc(false), noc_inst_ptr(noc_inst_ptr), dram_req_ptr(nullptr), cache_req_ptr(nullptr), data_id(data_id), dst_data_id(0), length(length), elem_bits(elem_bits), truncate(truncate), new_data_id(new_data_id) {
            this->type_ = InstType_t::DMA;
        }
        explicit DMAInst_t(const LLM_SIM_DRAM::ReqPtr_t dram_req_ptr, const uint64_t data_id, const int length, const int elem_bits = 32, bool truncate = false, uint64_t new_data_id = 0) :
        is_noc(false), is_dram(true), is_cache(false), is_cache_copy(false), is_cache_mc(false), noc_inst_ptr(nullptr), dram_req_ptr(dram_req_ptr), cache_req_ptr(nullptr), data_id(data_id), dst_data_id(0), length(length), elem_bits(elem_bits), truncate(truncate), new_data_id(new_data_id) {
            this->type_ = InstType_t::DMA;
        }
        explicit DMAInst_t(const LLM_SIM_INST::CacheReqPtr_t cache_req_ptr, const uint64_t data_id, const int length, const int elem_bits = 32, bool truncate = false, uint64_t new_data_id = 0) :
        is_noc(false), is_dram(false), is_cache(true), is_cache_copy(false), is_cache_mc(false), noc_inst_ptr(nullptr), dram_req_ptr(nullptr), cache_req_ptr(cache_req_ptr), data_id(data_id), dst_data_id(0), length(length), elem_bits(elem_bits), truncate(truncate), new_data_id(new_data_id) {
            this->type_ = InstType_t::DMA;
        }
        // Cache copy: read cache[src_data_id] into pool, then write pool to cache[dst_data_id].
        // This is the only way to rename/alias a cache entry without cross-module pointers.
        // dst_size (activation-shape normalization bug): optional
        // expected shape for the copied entry, mirroring Router::packet2data's own
        // "receiver's declared shape wins" convention for NoC-received data. Empty
        // (default) preserves the source's existing size_ unchanged, exactly as
        // before this parameter existed. Non-empty overrides dst's size_ after the
        // copy -- needed because a purely local (non-NoC) copy would otherwise
        // silently keep whatever size_ the source data happened to carry (e.g. the
        // 1-D {len} shape DRAMSim.cpp stamps on every DRAM read), while the NoC path
        // for the SAME logical activation always normalizes to the receiver's own
        // declared {1, act_n_elem} -- an inconsistency that let TMU concat/split see
        // two different ndims for what should be the same tensor.
        // elem_bits (step 2): defaults to 32
        // for full backward compatibility with every pre-existing call site
        // (rename/alias uses of this constructor are unaffected). A caller
        // that knows the copied entry's true logical bit width (e.g. a
        // Q8_0-quantized activation's same-core TP-1 self-dispatch copy)
        // can pass it explicitly so DMA::cal_lat()'s non-DRAM branch sees
        // the right value for this copy's own resource-slot timer.
        explicit DMAInst_t(uint64_t src_data_id, uint64_t dst_data_id_,
                            std::vector<int> dst_size = {}, int elem_bits = 32) :
        is_noc(false), is_dram(false), is_cache(false), is_cache_copy(true), is_cache_mc(false), noc_inst_ptr(nullptr), dram_req_ptr(nullptr), cache_req_ptr(nullptr), data_id(src_data_id), dst_data_id(dst_data_id_), length(0), elem_bits(elem_bits), truncate(false), dst_size(std::move(dst_size)) {
            this->type_ = InstType_t::DMA;
        }
        // Cache read for MainCtrl: DMA 發 req 給 Cache port 6，結果直送 MainCtrl.cache_data_in。
        // DMA 不占用 cache_slot_，發完 req 立刻完成。
        explicit DMAInst_t(uint64_t cache_read_id, bool /*for_main_ctrl_tag*/) :
        is_noc(false), is_dram(false), is_cache(false), is_cache_copy(false), is_cache_mc(true), noc_inst_ptr(nullptr), dram_req_ptr(nullptr), cache_req_ptr(nullptr), data_id(cache_read_id), dst_data_id(0), length(0), elem_bits(32), truncate(false) {
            this->type_ = InstType_t::DMA;
        }
        // DMA-specific members and methods
        const bool is_noc;
        const bool is_dram;
        const bool is_cache;
        const bool is_cache_copy; // true → cache[data_id] → pool → cache[dst_data_id]
        const bool is_cache_mc;   // true → cache[data_id] → MainCtrl.cache_data_in (no slot)
        const LLM_SIM_INST::NoCInstPtr_t noc_inst_ptr; // optional, only used for DMA instructions that correspond to NoC sends/recvs so we can track the NoC insts in the busy state
        const LLM_SIM_DRAM::ReqPtr_t dram_req_ptr; // optional, only used for DMA instructions that correspond to DRAM accesses so we can track the DRAM reqs in the busy state
        const LLM_SIM_INST::CacheReqPtr_t cache_req_ptr; // optional, only used for DMA instructions that correspond to Cache accesses so we can track the Cache insts in the busy state
        const uint64_t data_id;     // src (cache read key / primary key)
        const uint64_t dst_data_id; // cache_copy only: dst key for the cache write
        const int length;
        // Logical bit width per element for this transfer.
        // Defaults to 32 (float) so all existing call sites/instruction
        // text (without this field) are unaffected.
        const int elem_bits;
        // When true: the cache read that prefetches this data_id uses is_reserved=true,
        // so Cache automatically erases cache_data[data_id] after delivering it to DMA.
        // Use for NoC send / DRAM write of intermediate results that are never read again.
        const bool truncate = false;
        // Rename-on-terminal-read (0 = no rename, today's behavior
        // unchanged). Only meaningful together with truncate == true; not
        // applicable to is_cache_copy/is_cache_mc (left at its default 0
        // for those constructors -- see "明確不做的部分" in the rename plan,
        // DMA Cache copy's whole purpose is non-destructive duplication).
        const uint64_t new_data_id = 0;
        // cache_copy only: see the cache-copy constructor's comment above.
        const std::vector<int> dst_size;
};
using DMAInstPtr_t = std::shared_ptr<DMAInst_t>;

// This the request for Cache

class CacheReq_t {
    public:
        // new_data_id (0 = no rename, today's behavior unchanged): only
        // meaningful together with is_reserved == true. On a hit, Cache
        // re-keys the same physical entry from addr to new_data_id instead
        // of erasing it -- see Cache.cpp's rename-on-terminal-read logic.
        explicit CacheReq_t(uint64_t addr, bool is_reserved = false, uint64_t new_data_id = 0)
            : addr(addr), is_reserved(is_reserved), new_data_id(new_data_id) {}
        uint64_t getAddr() const { return addr; }
        bool getIsReserved() const { return is_reserved; }
        uint64_t getNewDataId() const { return new_data_id; }
    private:
        uint64_t addr;
        bool is_reserved;
        uint64_t new_data_id;
};
using CacheReqPtr_t = std::shared_ptr<CacheReq_t>;

// Write-side counterpart of CacheReq_t: metadata-only "intent to write"
// sent ahead of (same cycle as) the actual Data_t on data_in[i], so Cache
// can compute latency/capacity and reserve space before it ever reads the
// payload off the data FIFO. Mirrors Data_t<T>'s id/size_/elem_bits exactly
// -- see utils/Data.h's make_cache_write_req() for the only sanctioned way
// to build one.
class CacheWriteReq_t {
    public:
        explicit CacheWriteReq_t(uint64_t id, std::vector<int> size, int elem_bits)
            : id(id), size(std::move(size)), elem_bits(elem_bits) {}
        const uint64_t id;
        const std::vector<int> size;
        const int elem_bits;
};
using CacheWriteReqPtr_t = std::shared_ptr<CacheWriteReq_t>;

/*  
    These instruction types are not used right now
    They may be used in the next version of hardware
*/
/*
class DRAMInst_t: public BaseInst_t {
    public:
        explicit DRAMInst_t(std::string access_name, size_t length, bool is_read) : access_name(access_name), length(length), is_read(is_read) { this->type_ = InstType_t::DRAM; }
        // DRAM-specific members and methods
        const std::string access_name;
        const size_t length;
        const bool is_read; // true for read, false for write
};
using DRAMInstPtr_t = std::shared_ptr<DRAMInst_t>;
*/
}

#endif