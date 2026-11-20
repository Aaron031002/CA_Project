#ifndef LLM_SIM_ROUTER_H
#define LLM_SIM_ROUTER_H

#include <iostream>
#include <systemc.h>
#include <vector>
#include <queue>
#include <memory>
#include <unordered_map>
#include <algorithm>

#include "NoC_sim/noc_utils.h"
#include "noc_param.h"
#include "utils/fifo.h"
#include "utils/Data.h"
#include "utils/bufferAddr.h"
#include "utils/busyState.h"


template <typename T>
class Router : public sc_module {
    public:
        // Constructor
        Router(sc_module_name name, int id);

        // ── Clock ────────────────────────────────────────────────────────
        sc_in_clk SC_NAMED(clk);

        // ── Send path ─────────────────────────────────────────────────────
        // Local core supplies (data, send-inst) independently; both must be
        // available before the Router packs a Packet and forwards it to the NoC.
        sc_fifo_in<LLM_SIM_DATA::DataPtr_t<T>>      send_in_ports;   // data from local core
        fifo_in<LLM_SIM_INST::NoCInstPtr_t>         send_inst_in;    // send NoCInsts
        sc_fifo_out<LLM_SIM_NOC::PacketPtr_t<T>>    send_out_packet; // packets to NoC

        // ── Recv path ─────────────────────────────────────────────────────
        // Packets from the NoC are stored in packet_pool until a matching
        // recv NoCInst (keyed on packet_id) is issued by the local core.
        fifo_in<LLM_SIM_NOC::PacketPtr_t<T>>        recv_in_packet;  // packets from NoC
        fifo_in<LLM_SIM_INST::NoCInstPtr_t>         recv_inst_in;    // recv NoCInsts
        sc_fifo_out<LLM_SIM_DATA::DataPtr_t<T>>     recv_out_ports;  // data to local core

        // ── Busy state ────────────────────────────────────────────────────
        sc_out<LLM_SIM_BUSY::NoCBusyPtr_t> router_busy;

        // ── Dynamic-packet notification ───────────────────────────────────
        // When a packet with is_dynamic == true arrives from the NoC,
        // recv_packet_thread writes a DynamicReq_t here so that MainCtrl
        // can dispatch the corresponding predefined operation with priority.
        sc_fifo_out<LLM_SIM_NOC::DynamicReq_t> dynamic_notify_out;

    private:
        const int router_id;

        // Shared busy-state object written to router_busy.
        // recv_inst_thread is the sole driver of router_busy.
        // send_thread mutates current_busy_ in place so that any module
        // reading router_busy (and dereferencing the stored shared_ptr) always
        // sees the latest pending_noc_send_insts without a second port write.
        std::shared_ptr<LLM_SIM_BUSY::NoCBusy_t> current_busy_;

        // Pool of packets received from the NoC, keyed by packet_id.
        // recv_packet_thread inserts; recv_inst_thread removes.
        std::unordered_map<uint64_t, LLM_SIM_NOC::PacketPtr_t<T>> packet_pool;

        // Phase B (B-4): recv NoCInsts admitted
        // from recv_inst_in but not yet delivered (their packet hasn't
        // arrived in packet_pool yet, or recv_out_ports was full when it
        // did). Recv_inst_thread used to block on ONE specific packet_id
        // before even looking at the next queued inst -- a slow arrival
        // (e.g. an off-chip fetch, tens of thousands of cycles) would starve
        // every other recv this core is waiting on, including ones whose
        // packets had already arrived. Holding multiple outstanding entries
        // and scanning all of them each cycle removes that head-of-line
        // block; delivery is still one-at-a-time (recv_out_ports is a single
        // fifo, mirroring the one physical NoC ingress channel).
        std::vector<LLM_SIM_INST::NoCInstPtr_t> outstanding_;

        // ── Threads ──────────────────────────────────────────────────────
        // send_thread   : send_inst_in + send_in_ports → Packet → send_out_packet
        //                 Updates pending_noc_send_insts (mutate in-place, no port write).
        void send_thread();

        // recv_packet_thread : recv_in_packet → packet_pool (buffering)
        void recv_packet_thread();

        // recv_inst_thread   : recv_inst_in → match packet_pool → recv_out_ports
        //                      Updates pending_noc_recv_insts and drives router_busy.
        void recv_inst_thread();

        // ── Helpers ──────────────────────────────────────────────────────
        LLM_SIM_NOC::PacketPtr_t<T> data2packet(const LLM_SIM_DATA::DataPtr_t<T>& data_ptr,
                                                 const LLM_SIM_INST::NoCInstPtr_t& inst_ptr);
        LLM_SIM_DATA::DataPtr_t<T>  packet2data(const LLM_SIM_NOC::PacketPtr_t<T>& packet_ptr,
                                                 const LLM_SIM_INST::NoCInstPtr_t& inst_ptr);
};


#endif