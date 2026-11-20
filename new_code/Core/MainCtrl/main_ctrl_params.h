#ifndef LLM_SIM_MAINCTRL_PARAMS_H
#define LLM_SIM_MAINCTRL_PARAMS_H

// D_MODEL is AUTO-GENERATED from config/model.yaml by
// tools/gen_config_headers.py (do not edit that line by hand -- edit the
// YAML and re-run the generator). The rest of this file is still
// hand-maintained pending a Phase 2 migration pass.

namespace LLM_SIM_MAINCTRL {
    // Configurable parameters for MainCtrl.
    // NOC and DRAM are no longer standalone units — they are handled as
    // sub-types inside DMA instructions.
    constexpr int NUM_ARRAY_UNITS  = 1;
    constexpr int NUM_VECTOR_UNITS = 1;
    constexpr int NUM_DMA_UNITS    = 1;
    constexpr int NUM_TMU_UNITS    = 1;

    // Model activation width, shared by all RPC recv/dispatch default sizes
    // (was previously duplicated as two independent local
    // `static constexpr int D_MODEL = 2880;` inside MainCtrl.cpp).
    constexpr int D_MODEL = 2880;

    // ── RPC packet_id namespace ───────────────────────────────────────────────
    // Static packet_ids (gen-script assigned) occupy [0, DYNAMIC_PKT_BASE).
    // (packet-ID budget redesign, this round):
    // this block used to describe a single conflated ledger, but the ranges
    // below actually come from THREE independent, mutually-non-comparing
    // numbering schemes -- each only needs internal non-collision plus
    // staying under DYNAMIC_PKT_BASE, never cross-checked against the other
    // two:
    //   (a) decode, config_real.py:            PARTIAL_O=0, EXP_DISPATCH=
    //       5000, EXP_RESULT=6000, RPC_DISPATCH=8000, AGG_GATHER=9000,
    //       AGG_GRANT=9200 (max B=16 -> ceiling ~9280, see config_real.py
    //       itself). Unraised -- nowhere near the new base below.
    //   (b) decode TP-fanout variant, 6tier/gen_real_decode_tp.py:
    //       TP_REDUCE=3000, TP_DISPATCH=4000, TP_GATHER=7000, RPC_DISPATCH=
    //       8000. Unraised -- same margin as (a).
    //   (c) prefill, config_real_prefill_1_1024.py (+5 siblings): the ONE
    //       scheme actually scaled by PKT_BATCH_OFFSET*b (BATCH up to 32
    //       in config_real_prefill_32_1024.py, currently unused by any
    //       driver script but sized safely for it anyway) --
    //       K_TAIL=0, V_TAIL=1000, EXP_DISP=2000, EXP_RESULT=10000,
    //       RING_QKV=20000, RING_OPROJ=30000, RING_CONCAT=40000 (each
    //       ring range 4096-wide, Stage 1/3/4 --
    //       core_id*NUM_CORES+h reaches (NUM_CORES-1)^2+NUM_CORES-2=4094),
    //       AGG_GATHER=50000, AGG_GRANT=50500, PKT_BATCH_OFFSET=100000.
    //       Per-batch-item ceiling ~54595; full BATCH=32 ceiling
    //       31*100000+54595=3154595 -- see scheme (c)'s own config files
    //       for the authoritative per-constant values (this comment
    //       summarizes, doesn't duplicate the source of truth).
    // Dynamic packet_ids (RPCDispatch runtime-allocated) start at
    // DYNAMIC_PKT_BASE=5000000, raised 100x
    // from the old 50000 to keep a wide safety margin above scheme (c)'s new
    // BATCH=32 ceiling (3154595, ~1.6x margin) while staying comfortably
    // below offchip_fetch_pkt_id_base=20000000 (see config/hardware.yaml) --
    // this pool's own range extends to DYNAMIC_PKT_BASE +
    // NUM_CORES*DYNAMIC_POOL_SIZE = 5000000+64*20480=6310720, ~3.2x margin
    // clear of 20000000.
    // Router checks packet_id >= DYNAMIC_PKT_BASE to set is_dynamic = true.
    constexpr uint64_t DYNAMIC_PKT_BASE     = 5000000;

    // Runtime data_id pool: dynamic data_ids allocated by RPCDispatch start
    // here.  Static gen-script ids must stay below this value.
    // Raised to 10,000,000 (the global-call_n fix):
    // with the per-op_id call_n counter replaced by ONE global-per-core
    // counter (see MainCtrl::global_call_cnt_), call_n now increments on
    // EVERY dynamic request this core handles all layer, not just repeats of
    // the same op_id -- a much faster-growing count. Solving
    // call_n * KERNEL_ID_CALL_STRIDE + max_kernel_internal_id < DYNAMIC_DATA_ID_BASE
    // for max_kernel_internal_id ~= 9000 + 80*KERNEL_ID_STRIDE (80 kernels
    // observed registered on one three_mode core, see gen_decode_three_mode's
    // core_to_kernels) with KERNEL_ID_STRIDE=64 gives a safe call_n ceiling
    // of ~998 at this base -- comfortably above every realistic per-layer,
    // per-core dynamic-request count (loose worst-case estimate ~320: B*TOP_K
    // dispatches * (n_ih_max+1) hops, B=16/TOP_K=4/n_ih_max=4).
    constexpr uint64_t DYNAMIC_DATA_ID_BASE = 10000000;

    // Pool size per core: how many dynamic packet/data ids can be in-flight
    // simultaneously.  Circular allocation wraps at this count -- if a core
    // issues more than this many allocDynPktId() calls before the OLDEST
    // ones are actually consumed (packet delivered / ret-pkt matched), the
    // wraparound silently reissues an id that is still outstanding, and two
    // unrelated in-flight RPCs end up sharing one packet_id (whichever
    // arrives at the receiver's packet_id-keyed pending_recvs_ match first
    // wins, corrupting the other). Raised from 256 to 2048 (Bug F,
    // 2026-08-13): seq_len=1024's
    // Mode 2/3 ring dispatch drove a single busy leader core past 512
    // allocDynPktId() calls within 2.4ms sim time while earlier RPCs were
    // still in flight -- confirmed via DBG_RACE_TRACE that packet_id 50562
    // was allocated twice by core 2 only 291ns apart, the first send not
    // yet delivered, producing exactly the TMUOp::split size_/value.size()
    // mismatch symptom (a smaller, unrelated payload landed on the recv
    // that was really waiting for the larger ring-hop activation). 256 was
    // never validated against seq_len=1024's actual concurrent-RPC demand;
    // 2048 (8x, same safety-margin convention as this file's DYNAMIC_PKT_BASE
    // 10000->50000 fix for the sibling AGG_GRANT collision) is a generous
    // margin above the observed worst case, not a value derived from an
    // exact concurrency bound. Raised 10x to 20480 alongside DYNAMIC_PKT_BASE's
    // 100x raise (packet-ID budget redesign)
    // -- keeps NUM_CORES*DYNAMIC_POOL_SIZE proportionally similar relative to
    // the new DYNAMIC_PKT_BASE/OFFCHIP_FETCH_PKT_ID_BASE gap (~3.2x margin,
    // see DYNAMIC_PKT_BASE's own comment above), not re-derived from a new
    // concurrency measurement.
    constexpr uint64_t DYNAMIC_POOL_SIZE    = 20480;

    // ── Kernel-internal data_id namespace ────────────────────────────────────
    // Kernel scratch ids (IDAllocator(start=9000)) AND the fixed per-core WID
    // constants (WID_DISPATCH_ACT=21, WID_SCALAR_BOUNDARY_OUT=28,
    // WID_SCALAR_BOUNDARY=29, WID_TP_PARTIAL_BASE=30+) are remapped per
    // dynamic-request call to prevent stale/cross-request cache aliasing on
    // the same core. Remap formula: remapped = call_n * KERNEL_ID_CALL_STRIDE
    // + original_id, where call_n is now a SINGLE counter shared by every
    // op_id this core ever dispatches (MainCtrl::global_call_cnt_), not a
    // per-op_id counter -- found necessary via a real Mode2/3 numerical
    // corruption bug (multi-batch testing, B=16):
    // the old per-op_id counter let two DIFFERENT op_ids each reach call_n=0
    // on their own first call, both mapping WID_DISPATCH_ACT=21 to the
    // SAME literal 21 -- a real collision the old scheme never protected
    // against, since its "call_n=0 is always safe" guarantee only ever
    // covered the 9000+ kernel-internal range (whose per-kernel id_start
    // blocks are non-overlapping by construction), never the WID constants
    // (which are the SAME literal value across every kernel/op_id).
    // call_n=0 (the very first dynamic request this core ever handles) still
    // keeps original ids unshifted, matching every existing non-remapped
    // caller's assumption.
    constexpr uint64_t KERNEL_ID_BASE         = 9002;   // 9000/9001 = bias (static, never remap)
    constexpr uint64_t KERNEL_ID_CALL_STRIDE  = 10000;  // per-call block size (>= max kernel id ~9999)

    // Per-kernel scratch-id budget (Python gen-time convention, mirrored here
    // for the C++-side sanity margin computation above): every kernel
    // builder's id_start increments by this amount per kernel registered on
    // one core (id_start = 9000 + slot_idx * KERNEL_ID_STRIDE). Raised from
    // 16 to 64 -- the widest real kernel today (Mode 2's "final" ring role)
    // already uses 15 scratch ids, one short of the old 16-wide budget, with
    // zero generation-time enforcement catching an overflow into the next
    // kernel's block. Gen-time Python code (KERNEL_ID_STRIDE in gen_real_
    // prefill_128rp.py / gen_real_decode_tp.py) MUST be kept in sync with
    // this value by hand -- there is no shared header between the two
    // languages for this constant.
    constexpr uint64_t KERNEL_ID_STRIDE       = 64;
}

#endif
