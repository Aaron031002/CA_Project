#ifndef LLM_SIM_CHANNEL_H
#define LLM_SIM_CHANNEL_H

#include <cstdint>
#include <systemc.h>

// ─────────────────────────────────────────────────────────────────────────────
// Channel — 多通道 latency 模擬模組
//
// 每個 channel 各自獨立運作：
//   - 偵測到 start[i] 拉高的那個 cycle → 立刻拉高 busy[i]
//   - 等待 lat_in[i] 個 clk cycle
//   - fin[i] 拉高一個 cycle 後放下，busy[i] 同時放下
//
// 注意：每個 channel 對應一個永久 sc_spawn process（在 constructor 建立），
//       因此不同 channel 的 fin/busy port 永遠只有同一個 process 在寫，
//       不會觸發 SystemC E115（single-driver constraint）。
// ─────────────────────────────────────────────────────────────────────────────
class TransChannel : public sc_module {
public:
    sc_in_clk                  clk;
    sc_vector<sc_in<bool>>     start;   // start[i] 拉高一個 cycle 表示 channel i 開始
    sc_vector<sc_in<uint64_t>> lat_in;  // lat_in[i] 為 channel i 的 latency（cycles）
    sc_vector<sc_out<bool>>    fin;     // fin[i] 完成後拉高一個 cycle
    sc_vector<sc_out<bool>>    busy;    // busy[i] 從 start 到 fin 結束持續為高

    TransChannel(sc_module_name name, int channel_num)
        : sc_module(name),
          start("start"), lat_in("lat_in"), fin("fin"), busy("busy"),
          channel_num_(channel_num)
    {
        start.init(channel_num);
        lat_in.init(channel_num);
        fin.init(channel_num);
        busy.init(channel_num);

        // 每個 channel 在 elaboration 階段各自 spawn 一個永久 process
        // 每個 process 只負責自己 index 的 port，永遠是唯一 driver
        for (int i = 0; i < channel_num_; ++i)
            sc_spawn([this, i]() { channel_proc(i); });
    }

private:
    int channel_num_;

    void channel_proc(int idx) {
        // 初始化輸出（simulation 開始後第一個 delta cycle 執行）
        fin[idx].write(false);
        busy[idx].write(false);

        while (true) {
            // 等待 clk posedge，採樣 start
            wait(clk.posedge_event());
            if (!start[idx].read()) continue;

            // start 偵測到：拉高 busy，讀取本次 latency
            busy[idx].write(true);
            uint64_t lat = lat_in[idx].read();

            // 等待 lat 個 cycle（data-driven：一次性時間戳事件取代逐 cycle wait）
            wait(lat * sc_time(1, SC_NS));

            // 拉高 fin 一個 cycle
            fin[idx].write(true);
            wait(clk.posedge_event());

            // fin 和 busy 同時放下
            fin[idx].write(false);
            busy[idx].write(false);
        }
    }
};

#endif // LLM_SIM_CHANNEL_H
