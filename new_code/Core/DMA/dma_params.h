#ifndef LLM_SIM_DMA_PARAMS_H
#define LLM_SIM_DMA_PARAMS_H

#include <vector>
#include <string>
#include <memory>

namespace LLM_SIM_DMA {
    // Hardware resources parameters
    constexpr int NUM_MAX_CHANNEL = 2;
    // Buffer info parameters (deprecated)
    /*
    constexpr int ARRAY_NUM = 1;
    enum class ARRRY_BUF_TYPES { IN, OUT, BUF_NUM }; // the sequence of the buffer type is also the index of the port
    constexpr int ARRAY_OFF = ARRAY_NUM * static_cast<int>(ARRRY_BUF_TYPES::BUF_NUM); // the offset of the array buffer in the global buffer index (for example, if we have 2 array buffers, then the first array buffer will use global buffer index 0 and 1, and the second array buffer will use global buffer index 2 and 3, so the offset for the next module will be 4)
    constexpr int ARRAY_IN_BUF_NUM = 1;
    constexpr int ARRAY_OUT_BUF_NUM = 1;

    constexpr int VECTOR_NUM = 1;
    enum class VECTOR_BUF_TYPES { IN, WEIGHT, OUT, BUF_NUM };
    constexpr int VECTOR_OFF = VECTOR_NUM * static_cast<int>(VECTOR_BUF_TYPES::BUF_NUM);
    constexpr int VECTOR_IN_BUF_NUM = 2;
    constexpr int VECTOR_OUT_BUF_NUM = 1;
    
    constexpr int DRAM_NUM = 1;
    enum class DRAM_BUF_TYPES { READ, WRITE, BUF_NUM };
    constexpr int DRAM_OFF = DRAM_NUM * static_cast<int>(DRAM_BUF_TYPES::BUF_NUM);
    constexpr int DRAM_IN_BUF_NUM = 1;
    constexpr int DRAM_OUT_BUF_NUM = 1;
    
    constexpr int NOC_NUM = 1;
    enum class NOC_BUF_TYPES { READ, WRITE, BUF_NUM };
    constexpr int NOC_OFF = NOC_NUM * static_cast<int>(NOC_BUF_TYPES::BUF_NUM);
    constexpr int NOC_IN_BUF_NUM = 1;
    constexpr int NOC_OUT_BUF_NUM = 1;

    enum class OTHER_BUF_TYPES { K_BUFFER, V_BUFFER, RMS_BUF, BUF_NUM };
    constexpr int OTHER_OFF = static_cast<int>(OTHER_BUF_TYPES::BUF_NUM);
    // Note: this sequence of adding is also the sequence of these ports
    constexpr int DMA_BUF_NUM = ARRAY_NUM * static_cast<int>(ARRRY_BUF_TYPES::BUF_NUM) + VECTOR_NUM * static_cast<int>(VECTOR_BUF_TYPES::BUF_NUM) + DRAM_NUM * static_cast<int>(DRAM_BUF_TYPES::BUF_NUM) + NOC_NUM * static_cast<int>(NOC_BUF_TYPES::BUF_NUM) + static_cast<int>(OTHER_BUF_TYPES::BUF_NUM);
    constexpr int DMA_OUT_BUF_NUM = ARRAY_NUM * static_cast<int>(ARRRY_BUF_TYPES::IN) + VECTOR_NUM * static_cast<int>(VECTOR_BUF_TYPES::IN) + DRAM_NUM * static_cast<int>(DRAM_BUF_TYPES::READ) + NOC_NUM * static_cast<int>(NOC_BUF_TYPES::READ);
    constexpr int DMA_IN_BUF_NUM = ARRAY_NUM * static_cast<int>(ARRRY_BUF_TYPES::OUT) + VECTOR_NUM * static_cast<int>(VECTOR_BUF_TYPES::OUT) + DRAM_NUM * static_cast<int>(DRAM_BUF_TYPES::WRITE) + NOC_NUM * static_cast<int>(NOC_BUF_TYPES::WRITE);
    constexpr int DMA_OTHER_BUF_NUM = static_cast<int>(OTHER_BUF_TYPES::BUF_NUM);
    */
    // resource type parameters
    // true for in (output from a module), false for out (input to a module)
    // const std::vector<BufferIO_t> buffer_io_type = {}; 
    // latency estimate parameters
    const uint64_t RBW = 8;
    const uint64_t WBW = 8;
    const uint64_t overhead = 5;

    // Phase B: receive descriptor / outstanding-
    // transaction table depth -- how many NoC recv requests (instruction-
    // driven NOC_RECV plus DMA-synthesized flows like off-chip fetch/Phase F
    // fill) this core's DMA may have simultaneously registered and awaiting
    // their packet, before the actual payload transfer itself (still strictly
    // one at a time -- there is only one physical NoC ingress channel per
    // core; see PendingRecv/recv_xfer_active_ in DMA.h). Mirrors a real NIC's
    // receive-descriptor-ring or an NoC router's outstanding-transaction-id
    // table: tracking "what I'm expecting" is a separate, explicitly-sized
    // resource from the single-lane data path itself.
    //
    // PEND_PKT_RESERVED is carved out of MAX_PEND_PKT exclusively for DMA's
    // own synthesized flows (never consumed by ordinary instruction-driven
    // NOC_RECV) -- without this reservation, an instruction stream that fills
    // every slot with ordinary recvs could starve an off-chip fetch/fill that
    // itself needs a slot to retrieve the data those very recvs are waiting
    // on, a new deadlock. Mirrors a reserved response virtual channel /
    // credit pool in real NoC designs.
    //
    // MAX_PEND_PKT=4 (cap=3 for INSTRUCTION-kind recvs) deadlocked TP4
    // fan-out (S512 investigation):
    // a TP4 final IH's dispatch_extra_recv() (MainCtrl.cpp) registers
    // n_ih=4 extra-recv NOC_RECVs back-to-back in one processDynamicReq()
    // call, synchronously blocked inside dispatchDMAInst()'s while(!dispatch())
    // wait() loop -- MainCtrl::run()'s SC_THREAD cannot return to its own
    // inst_idx loop while stuck there, so this core's OWN later instructions
    // (e.g. its own K-cache chain-forward NoC send) never execute either.
    // Traced live: core 32 (final IH for one TP4 expert, and also the h=2
    // non-final-IH shard for that same expert) needed core 33 to send it a
    // partial sum, but core 33's own kernel dispatch was gated on receiving
    // core 32's K-cache forward -- a genuine circular wait, not just slow
    // queueing (core 32's 4th extra-recv registration was blocked on
    // can_admit_recv() capacity, which starved core 33, which starved core
    // 32's own ability to progress past the same call). n_ih is currently
    // {0, 1, 4} only (_gen_decode_real_shared.py's own assert), so cap must
    // be >= 4 to admit all of a TP4 final IH's extra-recvs in one call --
    // raised MAX_PEND_PKT by 1 so cap=4 covers this without touching
    // PEND_PKT_RESERVED's own reservation semantics.
    constexpr int MAX_PEND_PKT      = 5;
    constexpr int PEND_PKT_RESERVED = 1;
}
#endif