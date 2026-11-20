#include "Core/MainCtrl/MainCtrl.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>

MainCtrl::MainCtrl(sc_module_name name, std::string inst_file_path, int core_id)
    : sc_module(name),
      inst_file_path_(std::move(inst_file_path)),
      core_id_(core_id)
{
    using namespace LLM_SIM_MAINCTRL;

    array_inst_out.init(NUM_ARRAY_UNITS);   array_busy_in.init(NUM_ARRAY_UNITS);
    vector_inst_out.init(NUM_VECTOR_UNITS); vector_busy_in.init(NUM_VECTOR_UNITS);
    dma_inst_out.init(NUM_DMA_UNITS);       dma_busy_in.init(NUM_DMA_UNITS);
    tmu_inst_out.init(NUM_TMU_UNITS);       tmu_busy_in.init(NUM_TMU_UNITS);

    port_bundle_ = new PortBundle();
    decoder_     = new Decoder();

    for (int i = 0; i < NUM_ARRAY_UNITS; ++i) {
        port_bundle_->addPort(
            InstType_t::ARRAY,
            [this, i](const BaseInstPtr_t& inst) -> bool {
                auto ai = std::dynamic_pointer_cast<ArrayInst_t>(inst);
                if (!ai) return false;
                return array_inst_out[i].nb_write(ai);
            },
            [this, i]() -> bool {
                return array_inst_out[i].num_free() == 0;
            });
    }
    for (int i = 0; i < NUM_VECTOR_UNITS; ++i) {
        port_bundle_->addPort(
            InstType_t::VECTOR,
            [this, i](const BaseInstPtr_t& inst) -> bool {
                auto vi = std::dynamic_pointer_cast<VectorInst_t>(inst);
                if (!vi) return false;
                return vector_inst_out[i].nb_write(vi);
            },
            [this, i]() -> bool {
                return vector_inst_out[i].num_free() == 0;
            });
    }
    for (int i = 0; i < NUM_DMA_UNITS; ++i) {
        port_bundle_->addPort(
            InstType_t::DMA,
            [this, i](const BaseInstPtr_t& inst) -> bool {
                auto di = std::dynamic_pointer_cast<DMAInst_t>(inst);
                if (!di) return false;
                return dma_inst_out[i].nb_write(di);
            },
            [this, i]() -> bool {
                return dma_inst_out[i].num_free() == 0;
            });
    }
    for (int i = 0; i < NUM_TMU_UNITS; ++i) {
        port_bundle_->addPort(
            InstType_t::TMU,
            [this, i](const BaseInstPtr_t& inst) -> bool {
                auto ti = std::dynamic_pointer_cast<TMUInst_t>(inst);
                if (!ti) return false;
                return tmu_inst_out[i].nb_write(ti);
            },
            [this, i]() -> bool {
                return tmu_inst_out[i].num_free() == 0;
            });
    }

    SC_THREAD(run);
    sensitive << clk.pos();
}

void MainCtrl::run() {
    // ── Wait for start to be asserted ────────────────────────────────────────
    done.write(false);
    while (!start.read()) wait();

    {
        std::ifstream ifs(inst_file_path_);
        if (!ifs.is_open()) {
            SC_REPORT_ERROR("MainCtrl",
                            ("Cannot open instruction file: " + inst_file_path_).c_str());
            return;
        }
        std::string raw;
        int line_no = 0;
        while (std::getline(ifs, raw)) {
            ++line_no;
            // Drop a comment ('#' at the start of the line or after
            // whitespace), then surrounding whitespace incl. a CRLF '\r', so
            // blank/indented/commented lines never reach the decoder.
            std::string line = raw;
            for (size_t p = line.find('#'); p != std::string::npos; p = line.find('#', p + 1)) {
                if (p == 0 || std::isspace(static_cast<unsigned char>(line[p - 1]))) {
                    line.erase(p);
                    break;
                }
            }
            const size_t first = line.find_first_not_of(" \t\r\n");
            if (first == std::string::npos) continue;
            line = line.substr(first, line.find_last_not_of(" \t\r\n") - first + 1);

            // A line that doesn't decode exactly stops the simulation: running
            // it as some default instruction (or skipping it) would only show
            // up later as a wrong result or a deadlock.
            BaseInstPtr_t inst;
            std::string err;
            try {
                inst = decoder_->decode(line);
                if (!inst) err = "decoder returned no instruction";
            } catch (const DecodeError& e) {
                err = e.what();
            }
            if (!err.empty()) {
                SC_REPORT_FATAL("MainCtrl/DECODE",
                                ("core " + std::to_string(core_id_) + ", " + inst_file_path_ +
                                 ":" + std::to_string(line_no) + ": " + err +
                                 "\n    > " + line).c_str());
                return;
            }
            inst_queue_.push_back(std::move(inst));
        }
    }

    size_t inst_idx = 0;
    int stall_count = 0;
    size_t last_idx = SIZE_MAX;
    size_t progress_last_idx = SIZE_MAX;
    size_t hot_last_idx = SIZE_MAX;
    int hot_block_count = 0;
    while (inst_idx < inst_queue_.size()) {
        wait();

        // Trace stalls: print if core 0 has been stuck on same instruction for N cycles
        // Also monitor critical expert-chain cores for B=4 TP-2 debugging
        static const int MONITOR_CORES[] = {0,8,14,15,23,31,32,46,47,55,63};
        static const int N_MONITOR = 11;
        bool is_monitored = false;
        for (int _i = 0; _i < N_MONITOR; ++_i) if (MONITOR_CORES[_i] == core_id_) { is_monitored = true; break; }

        if (core_id_ == 0) {
            if (inst_idx != progress_last_idx && inst_idx % 50 == 0) {
                std::cout << "[MC-PROG] @" << sc_time_stamp()
                          << " inst[" << inst_idx << "]\n";
                progress_last_idx = inst_idx;
            }
        }

        if (is_monitored) {
            if (inst_idx == last_idx) {
                ++stall_count;
                if (stall_count == 100000 || stall_count % 200000 == 0) {
                    std::cout << "[STALL" << core_id_ << "] @" << sc_time_stamp()
                              << " stuck on inst[" << inst_idx << "] for "
                              << stall_count << " cycles"
                              << " | arr_free=" << (array_inst_out.size() ? array_inst_out[0].num_free() : -1)
                              << " vec_free=" << (vector_inst_out.size() ? vector_inst_out[0].num_free() : -1)
                              << " dma_free=" << (dma_inst_out.size() ? dma_inst_out[0].num_free() : -1)
                              << " tmu_free=" << (tmu_inst_out.size() ? tmu_inst_out[0].num_free() : -1)
                              << " type=" << static_cast<int>(inst_queue_[inst_idx]->getType())
                              << "\n";
                }
            } else {
                stall_count = 0;
                last_idx = inst_idx;
            }
        }

        const BaseInstPtr_t& inst = inst_queue_[inst_idx];
        const InstType_t     type = inst->getType();
        const bool trace_prefill_hotspot =
            (core_id_ == 0 && ((inst_idx >= 1418 && inst_idx <= 1424) ||
                               (inst_idx >= 1742 && inst_idx <= 1746)));

        if (trace_prefill_hotspot && inst_idx != hot_last_idx) {
            std::cout << "[MC-HOT-TRY] @" << sc_time_stamp()
                      << " inst[" << inst_idx << "]"
                      << " type=" << static_cast<int>(type) << "\n";
            hot_last_idx = inst_idx;
            hot_block_count = 0;
        }

        // ── TCore instructions (handled inline, never sent to a unit FIFO)
        if (type == InstType_t::TCORE) {
            auto tc = std::dynamic_pointer_cast<TCoreInst_t>(inst);
            if (tc) {
                if (tc->mode == TCoreInst_t::LOCAL_SYNC) {
                    handleLocalSync();
                } else if (tc->mode == TCoreInst_t::GLOBAL_SYNC) {
                    handleGlobalSync(tc);
                }
                // WAIT: advance past the instruction with no extra stall.
            }
            ++inst_idx;
            continue;
        }

        if (!port_bundle_->hasType(type)) {
            ++inst_idx;
            continue;
        }
        if (port_bundle_->dispatch(inst)) {
            if (trace_prefill_hotspot) {
                std::cout << "[MC-HOT-OK] @" << sc_time_stamp()
                          << " inst[" << inst_idx << "]"
                          << " blocked_cycles=" << hot_block_count << "\n";
            }
            ++inst_idx;
        } else {
            if (trace_prefill_hotspot) {
                ++hot_block_count;
                if (hot_block_count == 1 || hot_block_count % 10000 == 0) {
                    std::cout << "[MC-HOT-BLOCK] @" << sc_time_stamp()
                              << " inst[" << inst_idx << "]"
                              << " blocked_cycles=" << hot_block_count << "\n";
                }
            }
        }
    }
    // All instructions dispatched.
    done.write(true);
}

bool MainCtrl::allUnitsIdle() const {
    using namespace LLM_SIM_MAINCTRL;
    for (int i = 0; i < NUM_ARRAY_UNITS; ++i) {
        auto b = array_busy_in[i].read();
        if (b && b->isBusy()) return false;
    }
    for (int i = 0; i < NUM_VECTOR_UNITS; ++i) {
        auto b = vector_busy_in[i].read();
        if (b && b->isBusy()) return false;
    }
    for (int i = 0; i < NUM_DMA_UNITS; ++i) {
        auto b = dma_busy_in[i].read();
        if (b && b->isBusy()) return false;
    }
    for (int i = 0; i < NUM_TMU_UNITS; ++i) {
        auto b = tmu_busy_in[i].read();
        if (b && b->isBusy()) return false;
    }
    return true;
}

void MainCtrl::handleLocalSync() {
    // Wait until all local units have been idle for two consecutive clock cycles.
    int consecutive_idle = 0;
    int wait_count = 0;
    bool logged = false;
    // (TOKENS_PER_CORE>1 livelock investigation,
    // 2026-08-12): DBG_NOC_TRACE=1 (same env var as the NoC_sim.cpp/DMA.cpp
    // trace added earlier this investigation) also lowers this function's own
    // pre-existing SYNC-STUCK print threshold from every 1,000,000 wait()
    // calls down to every 1,000 -- the coarse 1e6 gate is why SYNC-STUCK
    // printing went silent after the ~350us throughput collapse even though
    // handleLocalSync() was very likely still spinning (consecutive_idle kept
    // resetting to 0, so allUnitsIdle() was flapping busy/idle, but at a rate
    // too fast to ever hit the old 1e6-cycle gate before consecutive_idle
    // reached 2 or the watchdog fired). Also records which specific unit
    // (dma/arr/vec/tmu) was busy at the LAST idle->busy transition, not just
    // a periodic snapshot, so a flapping unit is distinguishable from one
    // that's been continuously busy the whole time.
    const bool dbg_trace = std::getenv("DBG_NOC_TRACE") != nullptr;
    const int  log_period = dbg_trace ? 1000 : 1000000;
    int flap_count = 0;
    while (consecutive_idle < 2) {
        wait();
        ++wait_count;
        if (allUnitsIdle()) {
            ++consecutive_idle;
        } else {
            consecutive_idle = 0;
            ++flap_count;
            // Waiting at a sync while a long operation finishes is normal, so the
            // early (wait=101) report is printed only with DBG_NOC_TRACE.
            if (wait_count > 100 && ((dbg_trace && wait_count == 101) || wait_count % log_period == 0)) {
                using namespace LLM_SIM_MAINCTRL;
                bool dma_b=false, arr_b=false, vec_b=false, tmu_b=false;
                for(int i=0;i<NUM_DMA_UNITS;++i){auto b=dma_busy_in[i].read();if(b&&b->isBusy()){dma_b=true;}}
                for(int i=0;i<NUM_ARRAY_UNITS;++i){auto b=array_busy_in[i].read();if(b&&b->isBusy()){arr_b=true;}}
                for(int i=0;i<NUM_VECTOR_UNITS;++i){auto b=vector_busy_in[i].read();if(b&&b->isBusy()){vec_b=true;}}
                for(int i=0;i<NUM_TMU_UNITS;++i){auto b=tmu_busy_in[i].read();if(b&&b->isBusy()){tmu_b=true;}}
                uint64_t stuck_pkt = dma_ ? dma_->noc_recv_waiting_pkt() : 0;
                std::string dma_slots = dma_ ? dma_->debug_slots() : "no_dma";
                std::cerr << "[SYNC-STUCK] c=" << core_id_ << " wait=" << wait_count
                          << " flap_count=" << flap_count
                          << " dma=" << dma_b << " arr=" << arr_b << " vec=" << vec_b
                          << " tmu=" << tmu_b
                          << " noc_recv_pkt=" << stuck_pkt
                          << " slots=[" << dma_slots << "]"
                          << " @" << sc_time_stamp() << "\n" << std::flush;
                logged = true;
            }
        }
    }
}

void MainCtrl::handleGlobalSync(const TCoreInstPtr_t& tc) {
    // R1 counter breakdown (MainCtrl.h's own r1_barrier_wait_cyc_ doc
    // comment): time this core spends between arriving at the barrier and
    // being released, in NS ticks (converted to the same "cycle" unit
    // Total execution cycles itself uses -- 1 clock tick == 1 NS in this
    // model, see M1ProbeTB's own clk("clk", 1.0, SC_NS)).
    const bool dbg_r1 = std::getenv("DBG_R1_COUNTERS") != nullptr;
    const sc_time barrier_enter_t = dbg_r1 ? sc_time_stamp() : sc_time(0, SC_NS);

    // Notify GlobalSyncManager that this core has reached the barrier.
    GSyncArriveMsg_t msg;
    msg.barrier_id   = tc->barrier_id;
    msg.core_id      = core_id_;
    msg.participants = tc->participants;
    gsync_arrive_out.write(msg);  // blocking — FIFO depth is generous

    // Stall until GlobalSyncManager releases this core for this barrier.
    uint64_t released_id = 0;
    bool released = false;
    while (!released) {
        // Non-blocking poll of release FIFO
        if (gsync_release_in.nb_read(released_id)) {
            if (released_id == tc->barrier_id) {
                released = true;
                break;
            }
            // Wrong barrier_id — shouldn't happen in well-formed programs,
            // but push it back conceptually.  Since sc_fifo has no push_front,
            // we break and hope ordering is correct.
            released = true;
        }
        if (!released) wait();
    }

    if (dbg_r1) {
        const uint64_t wait_ns = static_cast<uint64_t>(
            (sc_time_stamp() - barrier_enter_t) / sc_time(1, SC_NS));
        r1_barrier_wait_cyc_ += wait_ns;
    }
}
