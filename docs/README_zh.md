# Final Project 1：small_mlp 在 2×2 多核心加速器上的映射 — 完整說明

> 最終（繳交）結果：`python3 run_scripts/score_mlp.py` →
> **`RESULT: PASS  cores=4  latency=1464 cycles  (same on all 3 inputs: in_B1, mnist_2, random)`**
> （範例程式 3097 cycles、及格線 < 2000 cycles）
>
> **2026-11-21 更新**：助教補充說明規則 3 的「任何輸入」是指**任何 784 個 [0, 1] 的值**，評分還會用沒看過、不限隨機的圖片。
> 原本用 sigmoid 自製的 softmax（1461 cycles）在最大 logit 超過約 88 時會算錯（實測見 7.3 節），不符合規定，
> 所以先改用內建 `Vector SOFTMAX`（1618 cycles），之後換成對任何輸入都正確、而且更快的 pairwise softmax（1547 cycles，7.4 節），最後再把它拆到兩個 core 上做 **（1464 cycles，7.5 節）**，這是繳交版。
> 下文第 5–6 節、第 8 節的 1461 與時間軸是最早自製 softmax 版本的歷史紀錄；第 0–4 層的映射各版完全相同。

這份文件從「完全不知道要做什麼」開始，一步一步說明：

1. [這份作業到底要做什麼](#1-這份作業到底要做什麼)
2. [我怎麼知道要做什麼、怎麼知道要怎麼做（資料來源）](#2-我怎麼知道要做什麼怎麼知道要怎麼做)
3. [必備的背景觀念（初學者版）](#3-必備的背景觀念初學者版)
4. [先搞清楚「時間花在哪裡」：從原始碼推導延遲公式 + 實測校正](#4-先搞清楚時間花在哪裡從原始碼推導延遲公式--實測校正)
5. [發想與迭代過程：v1 → v4（每一步為什麼這樣改）](#5-發想與迭代過程v1--v4)
6. [最終設計逐層詳解](#6-最終設計逐層詳解)
7. [Softmax：沒有 exp 指令時怎麼算，以及數值安全範圍](#7-softmax沒有-exp-指令時怎麼算以及數值安全範圍)
8. [結果夠不夠好？時間軸、下界分析、試過但放棄的方法](#8-結果夠不夠好)
9. [檔案、如何重現、如何繳交](#9-檔案如何重現如何繳交)
10. [限制與誠實聲明](#10-限制與誠實聲明)

---

## 1. 這份作業到底要做什麼

**一句話**：把一個手寫數字辨識的小型神經網路（MLP）的「推論」，寫成一個 4 核心加速器（模擬器）要執行的程式，讓它算得**正確**而且**越快越好**（模擬的 cycle 數越少越好）。

| 項目 | 內容 |
|---|---|
| 模型 `small_mlp` | 784 → 128 → 64 → 32 → 16 → 10。前 4 層是 `Linear + bias + ReLU`，最後一層是 `Linear + bias + softmax` |
| 輸入 | 一張 28×28 影像攤平成 784 個 float32：`in_tensor/mlp/in_B1.bin`（batch = 1） |
| 輸出 | 10 個類別機率：`out_tensor/mlp/small_mlp_out_B1.bin` |
| 硬體 | 固定的 2×2 mesh（4 個 core），XY routing，不能改 |
| 我們要交的東西 | `core_0_insts.txt` … `core_3_insts.txt`（每個 core 的指令）＋ `dram_map.yaml`（哪個 core 的 DRAM 放哪些檔案） |
| 評分 | 及格（2%）：`score_mlp.py` PASS 且 < 2000 cycles；排名（1%）：cycle 越少越好；報告 2% |

規則重點（`requirements §5`，都會被 `score_mlp.py` 自動檢查）：

* 指令只能**讀**提供的輸入和 `layer<N>_{weight,bias}.bin`，只能**寫**輸出檔；層與層之間的中間結果**不能**經過 DRAM 檔案，只能在 core 內的 cache 或透過 NoC 傳。
* 對**任何輸入影像**都要正確，而且延遲**不能**跟輸入值有關（會用 3 張圖測：`in_B1`、`mnist_2`、每次隨機產生的一張）。
* 不能用 `TOP_K`、`SPARSE_GATE`、`REDUCE_MAX_ABS`、`QUANTIZE_DEQUANT_Q8_0*`。
* 不能改模擬器、資料、`run_scripts/`；路徑一律是相對 repo 根目錄、用 `/`。

---

## 2. 我怎麼知道要做什麼、怎麼知道要怎麼做

閱讀順序（這也是建議你做類似作業時的順序）：

1. **`CA_FP1_26_requirements.pdf`**：知道「要交什麼、怎麼評分、有哪些規則」。
2. **`USER_GUIDE.md`**（= `CA_FP1_26_USER_GUIDE.pdf`）：知道檔案格式、怎麼 build / run / check，以及一堆「容易搞錯的地方」（例如 multicast send 最後一個數字 0/1 會被當成 truncate flag）。
3. **`new_code/inst_spec.md`**：每一條指令的語法。
4. **`new_code/curr_arch.md`**：每個硬體單元做什麼、彼此怎麼接。
5. **範例程式** `inst_gen/MLP_small_mlp/B1/`：只用 core 0、每條指令後面都 `LOCAL_SYNC`，3097 cycles。它告訴我們指令長什麼樣子，也暗示「這很慢、你要改」。

但文件只告訴你「指令能做什麼」，**沒有告訴你每條指令要花多少 cycle**。要做效能最佳化，一定要知道成本，所以我接著去讀模擬器原始碼（第 4 節），再用小實驗驗證（校正），最後寫一個追蹤工具看每條指令實際在第幾個 cycle 完成。整個流程就是：

```
讀規格 → 讀原始碼推導延遲模型 → 小實驗校正模型 → 做第一版(v1) → 追蹤時間軸找瓶頸
      → 針對最大的瓶頸改設計 → 再量測 → …（重複）→ 最終版 → 官方檢查
```

---

## 3. 必備的背景觀念（初學者版）

### 3.1 MLP 推論就是一串矩陣–向量乘法

每一層：`h_out = ReLU(h_in · W + b)`。`h_in` 是長度 K 的向量、`W` 是 K×N 矩陣、`b` 長度 N。
提供的權重檔已經轉置好了：`layerN_weight.bin` 的形狀是 `[in_dim, out_dim]`，row-major。

各層權重數量：

| 層 | K×N | 權重個數 | 佔比 |
|---|---|---|---|
| L0 | 784×128 | 100,352 | **約 90%** |
| L1 | 128×64 | 8,192 | 7% |
| L2 | 64×32 | 2,048 | 2% |
| L3 | 32×16 | 512 | |
| L4 | 16×10 | 160 | |

→ **L0 一定要拆給 4 個 core 一起做**；後面的小層，拆了反而可能被通訊成本吃掉。

### 3.2 每個 core 裡面有什麼

* **Array**（脈動陣列）：做矩陣乘法。`Array NORMAL` 會**自己從本 core 的 DRAM 讀權重**，輸入向量從 cache 讀，結果寫回 cache。
* **Vector**：逐元素運算（ADD、SUB、MUL、BIAS_ADD、RELU、SIG、RECIP、SOFTMAX…）。
* **DMA**：DRAM ↔ cache（`DMA DRAM read/write`）、core ↔ core（`DMA NoC send/recv`）。
* **TMU**：資料搬移/重排（concat、split…）。
* **cache**：每個 core 的共用暫存記憶體，所有單元都是透過「data id」在 cache 裡交換資料。
* **MainCtrl**：依序把指令丟給各單元（每 cycle 一條，in-order），各單元各自有指令佇列。

### 3.3 兩種拆矩陣的方法

把 `y = x · W`（W 是 K×N）拆給多個 core：

* **按欄拆（column split / N-split）**：每個 core 算一部分輸出 `y[cols]`。每個 core 需要**完整的輸入 x**，算完就是最終值（再接 bias+ReLU），只需要「拼起來」。
* **按列拆（row split / K-split）**：每個 core 拿一段輸入 `x[rows]` 乘 `W[rows, :]`，得到的是**部分和（partial sum）**，要把 4 份加起來才是答案。每個 core 只需要一部分輸入。

關鍵限制：**ReLU 是非線性的**，`ReLU(a+b) ≠ ReLU(a)+ReLU(b)`，所以 K-split 的部分和**必須先加完**，才能做 ReLU。

### 3.4 不用 `LOCAL_SYNC` 也能保證順序：cache 就是同步機制

我讀 `Cache.cpp` 發現：當某個單元要讀一個**還不存在**的 data id 時，cache 只是把請求掛著、每個 cycle 重試（miss 不扣時間），直到有人寫入那個 id。
所以只要「每個結果都有獨一無二的 id」，下游指令自然會等到上游算完才開始——像資料流（dataflow）機器。
範例程式每條指令後面的 `LOCAL_SYNC` 會強迫整個 core 停下來等所有單元閒置，完全不需要。**我的程式一條 SYNC 都沒有**，Array、Vector、DMA 可以同時在做不同的事。

### 3.5 關鍵路徑（critical path）

總延遲 = 最後一個 core 結束的時間。程式裡有很多工作是平行的，但總時間由「一條最長的、彼此相依的工作鏈」決定，最佳化就是一直找出這條鏈、把它變短。

---

## 4. 先搞清楚「時間花在哪裡」：從原始碼推導延遲公式 + 實測校正

### 4.1 從原始碼讀出來的延遲模型（1 cycle = 1 ns）

| 單元 / 動作 | 公式（float32） | 出處 |
|---|---|---|
| cache 讀或寫一筆資料 | `floor(位元數 / 256)`，即 **每 cycle 8 個 float** | `Cache.cpp latency_cal`、`hardware.yaml bus_width_bits: 256` |
| DRAM 讀 | `5 + ceil(n/64)`（每 cycle 64 個 float），**每個 core 的 DRAM 一次只服務一個請求** | `DRAMSim.cpp do_read` |
| DMA 的計時器 | `5 + ceil(len/8)`；`NoC send` 的 len 記為 0 → 5 | `DMA.cpp cal_lat` |
| Array `NORMAL` | 先讀輸入（cache）→ 再向 DRAM 要權重 → 計算 `4·⌈K/16⌉·⌈N/16⌉·M`（fp32 比 16-bit 慢 4 倍）→ 等 `max(0, 計算 − (5+KN/64))` → 再加 **固定 37 cycles**（32 fill/drain + 5）→ 寫回 cache | `ArrayUnit.cpp`、`array_params.h` |
| Vector 逐元素（ADD/RELU/SIG/RECIP…，≤128 個元素） | **14 cycles** + 讀輸入 + 讀第二個運算元（**兩者依序，不重疊**）+ 寫輸出 | `VectorOp.cpp`、`vector_params.h` |
| Vector `SOFTMAX` | `ceil(n/128)·69 + 244` → 10 個元素也要 **313 cycles** | 同上 |
| NoC 每一跳 | `3 + (bytes−1)/8` + 3（每 cycle 8 bytes = 2 個 float） | `NoC_sim.cpp hop_svc_cycle` |

**重要結論**（之後所有設計都根據這幾點）：

1. **L0 的瓶頸是 DRAM 頻寬和 Array 計算，兩者剛好平衡**：K、N 都是 16 倍數時，計算 `= KN/64`，等於 DRAM 讀權重的時間。L0 有 100,352 個權重，4 核平分每核至少要 **392 cycles**。
2. **cache 每 cycle 只能進出 8 個 float**：784 個輸入要「DMA 讀進來 98 + 寫進 cache 98 + Array 讀出來 98」，很貴。
3. **每條 Array 指令有約 46 cycles 固定成本**（DRAM 5 + 37 + 寫回），每條 Vector 指令約 17–30 cycles。
4. **SOFTMAX 單一指令就 313 cycles**，比整個 L3+L4 還久。
5. **跨 core 傳資料不便宜**（下面實測：32 個 float 傳一跳約 49 cycles）。

### 4.2 小實驗校正（`inst_gen/gen_code/exp_calib.py`）

我寫了一個指令產生器小工具（`isa.py`），自動產生很多只有一兩條指令的小 case，量測總 cycle 數再相減，確認上面的模型是對的。空程式本身是 23 cycles（啟動 + testbench 要求「全部閒置 20 cycles」才算結束）。

| 量測 | 結果（cycles） | 模型預測 |
|---|---|---|
| DMA 讀 784 個輸入 + Array 784×32 | 758 | 755 |
| DMA 讀 392 + Array 392×32 | 409 | 406 |
| NoC 傳 10 / 32 / 64 / 128 個 float，1 跳 | 39 / 49 / 65 / 97 | |
| 同上，2 跳 | 46 / 58 / 78 / 118 | |
| BIAS_ADD 10 / 32 / 128 | 21 / 27 / 51 | |
| RELU 10 / 32 / 128 | 18 / 21 / 45 | |
| SIG、RECIP（10 個） | 18、18 | |
| SOFTMAX（10 個） | **317** | 313 + I/O |
| ADD_REDUCE（10 個） | 74 | |
| TMU concat 4×32 → 128 / split 128 → 2×64 | ~37 / ~41 | |

模型和實測差距只有幾個 cycle → 可以放心用它來「預估」不同設計，再實際跑模擬確認。

### 4.3 追蹤工具：看每條指令在第幾個 cycle 完成（`trace.py`）

模擬器的 cache 裡本來就有除錯開關 `DBG_PIN_ID=<id>`：設定之後，那個 data id 被寫進 cache 時會印出 `[PIN-WRITE] @<時間>`。
`trace.py` 對每條指令產生的 id 各跑一次模擬（平行跑，每次 0.3 秒），就能列出**每個 core、每條指令完成的時間**（最終版的完整時間軸在 `docs/final_trace.txt`）。這是之後每一步改進的依據：看時間軸 → 找最長的那條鏈 → 改它。

---

## 5. 發想與迭代過程（v1 → v4）

| 版本 | 主要改變 | latency |
|---|---|---|
| 範例 | 1 個 core、每條指令後 LOCAL_SYNC | 3097 |
| v1 | L0 按欄 4 核平分 + 輸入分兩段管線化；L1 按列算部分和、匯集到 core 0；其餘在 core 0；不用 SYNC | 1655 |
| v2 | SOFTMAX 換成用 sigmoid 算 exp 的版本 | 1521 |
| v3 | 把最後一層的 bias 摺進 softmax 的正規化矩陣 | 1498 |
| v4（最終） | L1 改成兩個匯集點（core 0 收前 32 個輸出、core 1 收後 32 個），L2 兩核拆列；調整送出順序、bias 交給走 2 跳的那份部分和順便加 | **1461** |

### v1：先做出一個正確、合理的平行版本（1655）

**L0（最大的層）怎麼拆？** 兩個選擇：

* 按列（K-split）：每核只需讀 196 個輸入（便宜），但會得到 128 個部分和，要跨核加總（128 個 float 傳一跳 97 cycles + 加法），而且 ReLU 前必須加完。估計更慢。
* 按欄（N-split）：每核算 32 個輸出欄，需要完整 784 個輸入（cache 頻寬貴），但算完直接 bias+ReLU，不用跨核加總。→ **選這個**。

**把輸入分兩段（K-chunk pipelining）**：784 個輸入一次讀完要等約 219 cycles 才能開始算。改成先讀前 176 個（約第 66 cycle 就到了），Array 先算 `x[0:176]·W0[0:176, cols]`；同時 DMA 繼續把剩下 608 個搬進來，正好在 Array 算完第一段時到位；兩段部分和再用一條 Vector ADD 相加。用模擬器掃過切點（96…288、也試了切三段），**176 最好**（三段會多一次 46 cycles 的固定成本，反而變慢）。

**L1 怎麼做？** 每個 core 手上有 h0 的 32 個值，剛好對應 W1 的 32 列 → 每核算 `h0[32c:32c+32] · W1[32c:32c+32, :]`（64 個部分和），送到 core 0 加總 + bias + ReLU。這比「把 h0 片段送到 core 0、用 TMU concat 再做一次大的 L1」快（concat 37 cycles，而且 L1 的 8192 個權重只能由一核慢慢讀）。

**L2–L4 + softmax 在 core 0 上依序做**（這幾層太小，拆到別核賺不回傳輸成本）。

v1 時間軸告訴我們：L0 結束約 695、L1 加總完約 992、L2–L4 到 1302、**SOFTMAX 自己吃掉 316**。

### v2：拿掉 313 cycles 的 SOFTMAX（1521）

指令集沒有 exp，但有 `SIG`（sigmoid）和 `RECIP`（倒數），而
`σ(x)/σ(−x) = e^x`（因為 `σ(x) = 1/(1+e^{-x})`），這是**精確恆等式**，在 float32 也很準（第 7 節）。
再用一條 Array 指令乘上 10×10 的全 1 矩陣，就能一次得到「總和並複製到 10 個位置」，最後 RECIP + MUL 做正規化。約 9 條小指令 ≈ 200 cycles，取代 BIAS_ADD+SOFTMAX 的 336 → **−134**。
需要的常數（0、全 1 矩陣）都用提供的資料算出來（例如 `SUB(b4, b4) = 0`、`σ(0) = 0.5`），而且安排在 core 0 的 Vector 單元閒著的時候做（L0 期間它有約 400 cycles 沒事做）→ 不在關鍵路徑上。

### v3：把 b4 摺進正規化矩陣（1498）

`p_i = e^{z_i}/Σ_j e^{z_j}`，`z = u + b4`。因為 `e^{z_j} = e^{u_j}·e^{b_j}`：

```
T_i = Σ_j e^{u_j} · e^{b_j − b_i}  = (Σ_j e^{z_j}) / e^{b_i}
p_i = e^{u_i} / T_i
```

只要把全 1 矩陣換成 `W'[j][i] = e^{b_j − b_i}`，就**不需要先算 z = u + b4**，關鍵路徑少一條指令（−23）。`W'` 是 `e^{b4}`（行）乘 `e^{-b4}`（列）的外積，用一條 Array 指令（權重從 cache 讀，`wsrc cache`）算出來，安排在 core 0 的 Array 空檔（約 828–918 cycle，那時它在等 L1 的資料）。

### v4：L1 改成兩個匯集點（1461）

v3 的時間軸顯示，core 0 的 Vector 單元在 L1 階段被 3 條 64 寬的 ADD（各 41 cycles）+ 64 寬 RELU 卡住。改成：

* L1 的 64 個輸出分兩半：A（0–31）由 **core 0** 加總、B（32–63）由 **core 1** 加總。每核的 L1 部分和改成兩條 Array 指令（A 半、B 半各一條，各 66 cycles）。32 寬的 ADD 只要 ~27 cycles。
* **送出順序**：每核先算「要送出去」的那一半，尤其是要走 2 跳的（core 3→core 0、core 2→core 1），讓最遠的資料最早出發。
* **bias 交給走 2 跳的那份順便加**：core 3（A 半）、core 2（B 半）的 Vector 單元反正閒著，在送出前先加上 b1 的那一半 → 匯集點少一條指令。
* **加總順序 = 預期到達順序**（因為 Vector 是 in-order，排錯順序會讓先到的資料空等）。第一次嘗試沒排好只快了 2 cycles（1496），重新排序後才變 1461。
* L2 變成兩核各算一半列（core 0 用 h1A、core 1 用 h1B），core 1 把 32 個部分和送給 core 0。

---

## 6. 最終設計逐層詳解

### 6.1 誰做什麼

```
 y=1 │ core 2      core 3          1 跳：0-1、0-2、1-3、2-3
 y=0 │ core 0      core 1          2 跳：0-3、1-2
```

| 階段 | core 0 | core 1 | core 2 | core 3 |
|---|---|---|---|---|
| L0（每核 32 欄 × 784 列，輸入分 176+608 兩段） | 欄 0–31 | 欄 32–63 | 欄 64–95 | 欄 96–127 |
| L0 收尾 | + b0、ReLU → h0 片段（32 個） | 同左 | 同左 | 同左 |
| L1 部分和（自己的 32 列） | 先 B 半（送 core 1）再 A 半（自己） | 先 A 半（送 core 0）再 B 半（自己） | 先 B 半 +b1B（走 2 跳到 core 1），再 A 半（送 core 0） | 先 A 半 +b1A（走 2 跳到 core 0），再 B 半（送 core 1） |
| L1 加總 + ReLU | h1 A 半（32） | h1 B 半（32） | — | — |
| L2（兩核拆列） | h1A·W2[0:32] → 送 core 1；自己的 + b2 + (core 1 的部分和)，ReLU | h1B·W2[32:64] → 送 core 0；自己的 + b2 + (core 0 的部分和)，ReLU | — | — |
| L3、L4 | ✔（兩個 core 各算一次，得到相同的 u） | ✔ | — | — |
| softmax、寫輸出 | 類別 0–4，寫輸出第 0–4 欄 | 類別 5–9，寫輸出第 5–9 欄 | — | — |

（上表是繳交版 `pairwise_2core`，7.5 節。第 5–6 節其餘的指令節錄和時間軸是較早的單核 softmax 版本，第 0–2 層完全相同。）

### 6.2 core 0 的指令（節錄，路徑縮寫為 …/）

```
DMA DRAM read …/in_B1.bin index 0 1 0 176   500001      # 輸入前段（先讀，關鍵）
DMA DRAM read …/in_B1.bin index 0 1 176 784 500002      # 輸入後段
DMA DRAM read …/layer0_bias.bin index 0 1 0 32 500003   # 之後才讀小的 bias
...
Array NORMAL …/layer0_weight.bin index 0 176 0 32 stride 128 500001 500007 1    # 第一段部分和
Array NORMAL …/layer0_weight.bin index 176 784 0 32 stride 128 500002 500008 1  # 第二段部分和
Vector BIAS_ADD 32 500007 500003 500009 1 1             # 第一段 + b0（第二段還在算，不在關鍵路徑）
Vector SUB 10 …  /  SIG / RECIP / MUL …                 # softmax 常數：e^{b4}、e^{-b4}（Vector 正閒著）
Vector ADD 32 500009 500008 500017 1 1                  # 兩段相加
Vector RELU 32 500017 0 500018 1 0                      # h0 片段
Array NORMAL …/layer1_weight.bin index 0 32 32 64 stride 64 500018 500019 0   # L1 部分和 B 半（送出）
Array NORMAL …/layer1_weight.bin index 0 32 0 32 stride 64 500018 500020 1    # L1 部分和 A 半（自己留）
Array NORMAL …/layer4_weight.bin index 0 1 0 10 stride 10 500015 500021 1 wsrc cache 500016 1  # W' 外積
DMA NoC send 1000001 500019 1 1                         # B 半 → core 1（最後的 1 = 送完釋放）
DMA NoC recv 1000002 500056 32   (來自 core 1) / 1000005 (core 3) / 1000004 (core 2)
Vector ADD ×3 → Vector RELU 32 → h1 A 半
Array L2(A) → recv core 1 的 L2 部分和 → BIAS_ADD b2 → ADD → RELU → h2
Array L3 → BIAS_ADD b3 → RELU → h3 → Array L4 → u（不含 b4）
Vector SUB(0,u) → SIG(u) → SIG(-u) → RECIP → MUL = e^u
Array (e^u · W') → RECIP → MUL = 機率 → DMA DRAM write 輸出
```

幾個格式重點：

* `stride` 一定要寫**整個檔案的列寬**（L0 是 128），因為我們只讀其中 32 欄。
* `x_reserved = 1` 代表讀完就把輸入從 cache 刪掉；同一個 h0 要被兩條 L1 指令讀，所以第一條寫 0、第二條寫 1。
* 每個 `send` 都以 truncate flag 結尾（避免 USER_GUIDE 說的「目的地 0/1 被誤認為 flag」陷阱），`send` 與 `recv` 的 packet id、大小一致。
* `dram_map.yaml` 是從同一個 Python 資料結構自動產生的：誰讀了哪個檔就列為 owner，保證和指令一致。

### 6.3 最終時間軸（core 0 的關鍵路徑，取自 `docs/final_trace.txt`）

| cycle | 事件 | 這段花了 |
|---|---|---|
| 66 | 輸入前段（176 個）進到 cache | 66 |
| 224 / 649 | L0 第一段 / 第二段 Array 完成 | 583（Array 一直滿載） |
| 695 | h0 片段（ADD + ReLU）完成 | 46 |
| 763 / 828 | L1 部分和：送出用的一半 / 自己的一半 | 133 |
| 812 / 850 / 877 | 三份 A 半部分和到達 core 0（1 跳、2 跳+bias、1 跳第二份） | |
| 932 | h1 A 半（3 次 ADD + ReLU）；core 1 同時完成 B 半 | 104 |
| ≈1000 / 1049 | L2 部分和（core 0）/ core 1 的 L2 部分和到達 | |
| 1095 | h2（ADD + ReLU） | 163 |
| 1195 | h3（L3 Array + BIAS + ReLU） | 100 |
| 1245 | u = h3·W4 | 50 |
| 1424 | 機率（softmax 8 條指令） | 179 |
| 1461 | DMA 寫出 + 全部閒置 20 cycles → 結束 | 37 |

---

## 7. Softmax：沒有 exp 指令時怎麼算，以及數值安全範圍

### 7.1 數學

* `σ(x) = 1/(1+e^{−x})` ⇒ `σ(x)/σ(−x) = e^{x}`（精確）。
* 實作：`e = SIG(u) · RECIP(SIG(−u))`，`−u = SUB(0, u)`。
* 為什麼不用比較簡單的 `RECIP(σ(−u)) − 1 = e^u`？因為當 `u` 很負時，`1 + e^u` 在 float32 就等於 1，減 1 之後 `e^u` 變 0 → 很小的機率會完全錯（評分有檢查 `log p` 誤差 < 1e-3，就是為了抓這種錯）。`σ(u)` 在 `u` 很負時本身就是精確的 `≈ e^u`，所以用比值才準。
* 正規化用 v3 的 `W'[j][i] = e^{b_j − b_i}`（第 5 節）。

### 7.2 這個做法什麼時候會壞掉？我做了三種檢查

1. **float32 模擬**（`softmax_emul.py`）：2 張提供的影像 + 5000 張隨機影像，最大 log 誤差 **2.1e-6**（限制 1e-3），全部通過。
2. **logit 分布**（`logit_stats.py`）：MNIST 圖最大 logit 約 12–18；20000 張均勻隨機影像最大 logit 介於 4.0–30.7。
3. **最壞情況**（`logit_bounds.py`）：用梯度上升找「像素都在 [0,1] 內、但刻意讓 logit 最大」的對抗影像，可以做到 **185**。

我的 softmax 只要最大 logit 小於約 **85** 就正確（`e^{85}` 還在 float32 範圍內；超過約 88 時 `σ(−u)` 會變 0）。所以：

* 對 MNIST、隨機影像（評分用的三種輸入）都有很大的安全餘裕；
* 但對「刻意構造的極端影像」不保證正確；內建的 `Vector SOFTMAX` 會先減最大值，所以對任何輸入都安全。

因此產生器有開關：

| 選項 | 延遲 | 適用範圍 |
|---|---|---|
| `--softmax pairwise_2core`（成對差值、拆到兩個 core，**預設、繳交版**，7.5 節） | **1464** | 任何輸入（依提供的 Release 編譯，見 7.4 節） |
| `--softmax pairwise`（成對差值，只用 core 0，7.4 節） | 1547 | 同上 |
| `--softmax pairwise_safe`（同上，保證不除以 0） | 1589 | 任何輸入、任何編譯設定 |
| `--softmax builtin`（內建 SOFTMAX） | 1618 | 任何輸入 |
| `--softmax accurate` | 1461 | 最大 logit < ~88（**不符合規則 3**） |
| `--softmax shift`（`e^{u+40} = 1/σ(−(u+40)) − 1`，少一條指令） | 1444 | 最大 logit < ~45（**不符合規則 3**） |

三種都通過官方 `score_mlp.py` 的三張圖，但助教補充說明：規則 3 要求對**任何 784 個 [0, 1] 的值**都正確，評分還會用沒看過的圖。
所以 `accurate`/`shift` 不能用；繳交版是 7.5 節的 `pairwise_2core`。

### 7.3 實測：用極端圖片跑官方檢查

照助教說明的方法（把 784 個 float32 放到一份 repo 副本的 `in_B1.bin` 位置，在副本裡跑 `score_mlp.py`）：

* `inst_gen/gen_code/stress_inputs.py` 產生 19 張圖（全 0、全 1、對每一類做梯度上升、把最大 logit 推過 40/80/90/120/160 的快照、讓 logit 盡量小的圖），每張前兩大機率的差都 ≥ 1e-3。
* `inst_gen/gen_code/stress_score.sh` 自動做「複製 repo → 換 in_B1.bin → 跑 score_mlp.py」。

| 版本 | 結果 |
|---|---|
| `builtin`（1618） | **19/19 PASS** |
| `accurate`（1461） | 最大 logit ≤ 82 的 6 張 PASS；**最大 logit ≥ 92 的 13 張全部 FAIL**（輸出出現非有限值） |

和理論預測一致：σ(−u) 在 u > 約 88 時 float32 下溢為 0。

### 7.4 更快又安全的自製 softmax：pairwise（成對差值）

關鍵：只對 logit 的**差**取指數。`p_i = 1 / Σ_j e^{z_j − z_i}`。

| 步驟 | 指令 | 完成（cycle） |
|---|---|---|
| `u = h3·W4` | Array | 1245 |
| `z = u + b4`（和下一步同時做） | Vector BIAS_ADD | 1265 |
| `R[j][i] = −u_j` | Array 外積（u 當 10×1 輸入 × −1 列，`df OS` 讓計算只要 4 cycles） | 1302 |
| `D[j][i] = z_i − u_j` | Vector BIAS_ADD（100 個值） | 1344 |
| `H = 1/σ(D) = 1 + e^{u_j − z_i}` | Vector SIG + RECIP（100 個值） | 1383 / 1422 |
| `S_i = Σ_j e^{b_j}·H[j][i] = K + Σ_j e^{z_j − z_i}` | Array（e^{b} 當輸入、H 當權重） | 1473 |
| `p_i = 1/(S_i − K)` | Vector SUB + RECIP | 1493 / 1510 |
| 寫出 + 收尾 | DMA | **1547** |

為什麼對任何輸入都安全：若 `p_i > 1e-30`，則所有 `z_j − z_i < 約 72`，第 i 欄沒有任何東西會溢位，結果精確；
若某個 logit 比 z_i 大超過約 88，那欄會出現 inf，`S_i = inf`，`p_i = 1/inf = 0`——而真正的 p_i < 1e-38，檢查允許。

**要注意的一點**：差距超過約 104 時，σ 會下溢成剛好 0，`RECIP` 等於除以 0。模擬器原始碼對此有 `assert`，
但提供的 CMake 預設 Release（`-O3 -DNDEBUG`），assert 不會執行，1/0 = inf，結果仍正確（實測如上）。
不想依賴這點的話，`pairwise_safe` 在 RECIP 前加上 2^-120，保證不會除以 0，代價多一條 100 值的 Vector 指令。

| 版本 | cycles | 33 張圖（原圖＋32 張極端圖，每張再加 mnist_2、隨機圖） | 最大誤差（abs / log） |
|---|---|---|---|
| `builtin` | 1618 | 33/33 PASS | 4.8e-6 / 4.2e-5 |
| `pairwise` | 1547 | 33/33 PASS | 7.6e-6 / 4.3e-5 |
| `pairwise_2core`（**繳交版**，7.5 節） | **1464** | 33/33 PASS | 7.6e-6 / 5.0e-5 |
| `pairwise_safe` | 1589 | 33/33 PASS | 7.6e-6 / 4.0e-5 |

32 張極端圖包含 logit 高達 185、多個大 logit 互相接近、以及「logit 很大且某類機率剛好在 1e-30 附近」（log 檢查真正會作用）的圖。
其中 25 張的 logit 差距超過 104，也就是真的走到 RECIP(0) → inf 這條路，pairwise 全部通過。

**為什麼繳交 `pairwise` 而不是 `pairwise_safe`**：USER_GUIDE §15 寫明助教用「未修改的 `score_mlp.py`、提供的模擬器」評分；
提供的 `build.sh` 只跑 `cmake ..`，`new_code/CMakeLists.txt` 在沒指定時固定用 Release（`-DNDEBUG`），assert 不會執行。
只有刻意改用 Debug 編譯才會在 RECIP(0) 中止，所以多花 42 cycles 換這個保險不划算。
另外，`score_mlp.py` 用的三種輸入根本不會走到這條路：in_B1、mnist_2 的 logit 差距約 37，20 萬張均勻隨機圖最大也只有 62.4（< 88），
只有刻意構造的極端圖才會出現 inf 或 RECIP(0)。

### 7.5 再快 83 cycles：把 softmax 拆到兩個 core（`pairwise_2core`，1464）

在 7.4 節的 pairwise 版本裡，u 在 1245 算完之後，softmax 還要 265 cycles，而且全部擠在 core 0；core 1–3 那時候都閒著。
三個改動（全部用文件允許的指令）：

1. **core 1 也算出 h2、h3、u。** 第 2 層本來是 core 1 把部分和送給 core 0；改成兩個 core **互相交換**原始部分和，
   各自先把 b2 加到自己那份（等待時順便做），收到後 ADD + ReLU。兩邊都在 1095 得到 h2，接著各自做第 3、4 層，
   都在 1245 得到 u。多出來的只是一次反方向的 NoC 傳輸，不在關鍵路徑上。
2. **每個 core 只算 5 個類別。** core 0 算類別 0–4、core 1 算 5–9，各自用 `DMA DRAM write` 寫輸出檔的那 5 欄
   （`dram_map.yaml` 的輸出檔 owners 改成 `[0, 1]`）。原本 100 個值的 Vector 運算變成 50 多個值，每條省十幾 cycles。
3. **用 TMU 取代外積 Array。** 讀 `TMUOp.cpp` 發現 TMU 的延遲只有「元素數 ÷ 8」，小資料非常便宜（之前量到的 37–41 cycles 是 128 個元素的情況）：
   * `TMU split` 把 u 切成 u[0:5]、u[5:10]（4 cycles），各 core 取自己那 5 個加上 b4 → z_C；
   * `TMU view` 支援 numpy 式 **broadcast**：把 −u 當成 [10,1]，broadcast 成 [10,5] 就是 R[j][i] = −u_j（14 cycles，原本外積 Array 要 57）。
     −u 由 `SUB(零, u)` 算出，零向量事先用 `TMU view` 做成 [10,1] 的形狀，SUB 的輸出就直接是 [10,1]。
4. **加一列「墊底列」，省掉最後的 SUB。** 原本 S_i = K + Σ_j e^{z_j−z_i}，要再減 K。現在在 −u 後面接一個 2^32
   （`TMU concat`），D 的第 11 列 = 2^32 + z_i，σ 剛好等於 1、H 剛好等於 1；最後那個 Array 的輸入改成 [e^{b4}, −K]（11 個值），
   第 11 列乘上 −K 就把 K 消掉了，Array 直接輸出 Σ_j e^{z_j−z_i} = 1/p_i，只剩一個 RECIP。

| 步驟（每個 core） | 指令 | 完成（cycle） |
|---|---|---|
| u = h3·W4 | Array | 1245 |
| u 切半 | TMU split | 1249 |
| −u（形狀 [10,1]） | Vector SUB | 1262 |
| [−u; 2^32] → broadcast 成 [11,5] | TMU concat + view | 1267 / 1281 |
| z_C = u_C + b4_C | Vector BIAS_ADD | 1282 |
| D = z_i + R[j][i]（55 個值） | Vector BIAS_ADD | 1312 |
| H = 1/σ(D) | Vector SIG + RECIP | 1339 / 1366 |
| Σ_j e^{b_j}H[j][i] − K = 1/p_i | Array（權重從 cache） | 1411 |
| p_i | Vector RECIP | 1428 |
| 寫出 5 欄 + 收尾 | DMA | **1464** |

常數（e^{b4}、−K、2^32、[10,1] 零向量）在兩個 core 上都趁第 0 層 Array 忙的時候算好（core 0 到 633 做完，離需要 Vector 的 672 還有餘裕）。
數學和 7.4 節完全相同，安全性的分析也一樣（包括 RECIP(0) 依賴 Release 編譯這點）。實測：33 張圖（原圖＋32 張極端圖）× 3 種輸入共 99 次比對全部 PASS、延遲都是 1464；
從乾淨的 `CA_FP1_26.tar.gz` 解壓、照 USER_GUIDE §14 編譯後跑 `score_mlp.py` 也是 PASS / 1464。

寫程式時踩到、值得記下的兩個模擬器細節：

* `ADD`/`SUB`/`MUL`/`BIAS_ADD` 一律處理 **x 的整個 buffer**（`dim` 對它們只是檢查用或 BIAS_ADD 的週期），`SIG`/`RECIP` 才只處理 `dim` 個；
  所以 `SUB 1 <10 個值> <1 個值>` 會讀超出範圍，要用 1 個值的 buffer 當 x。
* 同一個 id 當兩個運算元（例如 `MUL x x` 做平方）時，只能在**第二個**（w_reserved）設 truncate；設在第一個會在讀第二次前就被刪掉 → 卡死。

也試過、但不划算的：**拆到 3 個或 4 個 core**（產生器保留了 `--softmax pairwise_3core` / `pairwise_4core`，實測 1473 / 1495，都比 1464 慢）。原因：core 2、3 要等兩份第 2 層部分和，晚 35 cycles 才拿到 h2；第 2 層改成 multicast 後，送到 core 0 的那份也晚了 10 cycles；而每個 core 少算一個類別只省 1–2 cycles（短向量的 Vector 指令幾乎都是固定成本 14 cycles）。另外也評估過
用 `Vector DOT_PRODUCT` 取代最後的 Array（固定成本 50，比 Array 慢）、讓第 4 層 Array 直接輸出 −u（權重改從 cache 讀反而多 12 cycles）。

---

## 8. 結果夠不夠好

### 8.1 正確性

繳交版（pairwise_2core）：`score_mlp.py` PASS / 1464，三種輸入延遲相同；另外 32 張極端圖片（7.3–7.5 節）也全部 PASS，最大誤差 abs 7.6e-6、log 5.0e-5（上限 1e-4 / 1e-3）。
（自製 softmax 版當時跑 5 次也都 PASS / 1461，誤差：最大絕對誤差 1.8e-7、最大 log 誤差 1.3e-5——但它對極端圖片會失敗。）

### 8.2 效能

* 繳交版：3097 → **1464 cycles**，比範例快 **2.12 倍**，低於及格線 2000（內建 SOFTMAX 版 1618、單核 pairwise 版 1547）。
* 自製 softmax 版（歷史）：1461 cycles；時間分配 L0 約 695（48%）、L1 237、L2–L4 313、softmax 179、收尾 37。
  各版差異全部來自 softmax（u = h3·W4 在各版都是 1245 完成）：之後內建 SOFTMAX 版還要 373 cycles（SOFTMAX 本身固定 313）→ 1618；單核 pairwise 版 302 cycles → 1547；雙核 pairwise 版 219 cycles → 1464，而且對任何輸入都正確。

### 8.3 離理論極限還有多遠？

* **L0 的硬下界**：每核 25,088 個權重 ÷ 64 = 392 cycles 純 DRAM 時間，加上輸入第一段進 cache（約 66）、輸入從 cache 讀出（98）、兩條 Array 的固定成本（約 92），≈ 650。實際 Array 在 649 完成 → **L0 已經幾乎貼著下界**。
* **後面幾層**是一條無法平行的相依鏈（每層都要等上一層的 ReLU），每層至少要「Array ~46 固定 + Vector 14×2 固定」。剩下能省的主要是十幾、二十幾 cycles 等級的微調。

### 8.4 試過但放棄（或算過不划算）的方法

| 想法 | 為什麼不採用 |
|---|---|
| L0 按列拆（K-split） | 128 寬部分和要跨核加總，傳一跳就 97 cycles，比省下的輸入讀取還貴 |
| L0 2×2 混合拆法 | 估算/實測都比按欄拆慢 10–30 cycles |
| 輸入切 3 段 | 多一次 Array 固定成本，掃描結果比 2 段慢 |
| 用 TMU concat/split 重組資料 | 一次就要 37–41 cycles |
| 用 `ADD_REDUCE` 加總 | 74 cycles，比 Array 乘全 1 矩陣（54）慢 |
| L2/L3 再拆到更多核 | 傳輸（≥41）比省下的計算多 |
| softmax 只用一個 sigmoid（`shift`） | 快 17 cycles，但安全範圍小很多（列為選項） |

我也注意到一個可以「鑽漏洞」的地方但**沒有使用**：`DMA NoC recv` 的大小欄位，延遲是用所有數字「相加」算的，形狀卻是用「相乘」；寫成 `8 8` 代表 64 個元素時，延遲只算 16 個。這明顯是模擬器的建模瑕疵，不是真實硬體行為，用了也不符合作業精神，所以我只用 `32` 這種正常寫法。

---

## 9. 檔案、如何重現、如何繳交

```
inst_gen/MLP_small_mlp/B1/          ← 要交的 5 個檔案（由產生器寫出）
inst_gen/gen_code/
  gen_small_mlp.py   最終產生器（預設 --softmax pairwise_2core = 繳交版；pairwise|pairwise_safe|builtin 也安全；accurate|shift 只留作比較）
  stress_inputs.py   產生極端測試圖（[0,1] 內、logit 很大或很小）
  stress_score.sh    在 repo 副本裡用這些圖跑官方 score_mlp.py（主機上執行）
  isa.py             指令產生小工具（自動配發唯一 id、自動產生 dram_map.yaml）
  run_case.py        跑一個 case + 比對答案（開發用）
  trace.py           每條指令完成時間的追蹤工具（DBG_PIN_ID）
  exp_calib.py       延遲校正小實驗
  logit_stats.py / logit_bounds.py / softmax_emul.py   softmax 數值分析
docs/README_zh.md    本文件
docs/final_trace.txt 自製 softmax 版（1461）的完整時間軸；繳交版只有 softmax 那段不同
docs/video/          教學影片：build/small_mlp_tutorial.mp4（9 分 25 秒）、字幕、腳本、投影片、主機上的製作與檢查腳本
```

在容器內（`bash start_env.sh`）、repo 根目錄：

```bash
python3 inst_gen/gen_code/gen_small_mlp.py           # 重新產生 5 個檔案
python3 run_scripts/score_mlp.py                     # 官方檢查（應得 PASS / 1464）
bash run_scripts/run_mlp_configs.sh && python3 run_scripts/compare_output_mlp.py   # 只跑 in_B1 並比對（PASS）
# 注意：score_mlp.py 最後一次執行用的是隨機圖，跑完後輸出檔裝的是「隨機圖的結果」。
# 這時直接跑 compare_output_mlp.py 會拿它去和 in_B1 的答案比，必定 FAIL（例如 sim=[8], ref=[7]）；
# 要先用 run_mlp_configs.sh 重跑 in_B1，或加 --input 指定那張隨機圖（在 sim_logs/score_*/inputs/random.bin）。
python3 inst_gen/gen_code/trace.py inst_gen/MLP_small_mlp/B1   # 看時間軸
```

繳交（`[id]` 換成組號）：zip 最上層直接放 5 個檔案、不要有資料夾：

```bash
cd inst_gen/MLP_small_mlp/B1 && python3 -m zipfile -c ../../../Group_[id]_FP1_code.zip core_0_insts.txt core_1_insts.txt core_2_insts.txt core_3_insts.txt dram_map.yaml && python3 -m zipfile -l ../../../Group_[id]_FP1_code.zip
```

（主機上沒有 `zip` 指令，所以用 Python 內建的 zipfile；最後的 `-l` 會列出內容，應該剛好是這 5 個檔、沒有資料夾。）

報告寫在 `Group_[id]_FP1_report.docx` 模板、輸出成 PDF（本文件第 4–8 節可以直接當報告素材）。

---

## 10. 限制與誠實聲明

* **softmax**：繳交版用 pairwise_2core（7.4、7.5 節），對任何 [0,1] 輸入都正確（32 張極端圖實測）；差距 > ~104 時依賴提供的 Release 編譯讓 RECIP(0) = inf，若要完全不依賴編譯設定可改用 `pairwise_safe`（1589）。自製的 `accurate`/`shift` 版對最大 logit > ~88（或 ~45）的輸入會算錯，不符合規則 3，只留作比較。
* **影片**（`docs/video/`）是在改版前做的，片中的最終結果仍是 1461（sigmoid 版 softmax），需要更新為 pairwise_2core / 1464。
* **模擬器怪癖**：TMU 在「讀取輸入」期間不會被標成 busy；如果程式最後一個動作是 TMU，模擬可能提早被判定結束。我的程式不用 TMU，不受影響。
* 延遲模型是從這份模擬器讀出來的，不代表真實硬體。
* 這份作業的程式與分析是我（AI）完成的；請你讀懂後用自己的話寫報告，並遵守課程對 AI 使用的規定。
