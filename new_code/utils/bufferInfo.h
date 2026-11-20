#ifndef LLM_SIM_BUFFERINFO_H
#define LLM_SIM_BUFFERINFO_H

/*
    Note: This code will be deprecated in the next version
    because the buffers are merged in one collective buffer pool
*/


namespace LLM_SIM_BUF {
    // Buffer info parameters
    constexpr int ARRAY_NUM = 1;
    enum class ARRRY_BUF_TYPES : int { IN = 0, OUT = 1, BUF_NUM = 2 };
    constexpr int ARRAY_OFF = ARRAY_NUM * static_cast<int>(ARRRY_BUF_TYPES::BUF_NUM);
    constexpr int ARRAY_IN_BUF_NUM = 1;
    constexpr int ARRAY_OUT_BUF_NUM = 1;

    constexpr int VECTOR_NUM = 1;
    enum class VECTOR_BUF_TYPES : int { IN = 0, WEIGHT = 1, OUT = 2, BUF_NUM = 3 };
    constexpr int VECTOR_OFF = VECTOR_NUM * static_cast<int>(VECTOR_BUF_TYPES::BUF_NUM);
    constexpr int VECTOR_IN_BUF_NUM = 2;
    constexpr int VECTOR_OUT_BUF_NUM = 1;

    constexpr int DRAM_NUM = 1;
    enum class DRAM_BUF_TYPES : int { READ = 0, WRITE = 1, BUF_NUM = 2 };
    constexpr int DRAM_OFF = DRAM_NUM * static_cast<int>(DRAM_BUF_TYPES::BUF_NUM);
    constexpr int DRAM_IN_BUF_NUM = 1;
    constexpr int DRAM_OUT_BUF_NUM = 1;

    constexpr int NOC_NUM = 1;
    enum class NOC_BUF_TYPES : int { READ = 0, WRITE = 1, BUF_NUM = 2 };
    constexpr int NOC_OFF = NOC_NUM * static_cast<int>(NOC_BUF_TYPES::BUF_NUM);
    constexpr int NOC_IN_BUF_NUM = 1;
    constexpr int NOC_OUT_BUF_NUM = 1;

    enum class OTHER_BUF_TYPES : int { K_BUFFER = 0, V_BUFFER = 1, RMS_BUF = 2, BUF_NUM = 3 };
    constexpr int OTHER_OFF = static_cast<int>(OTHER_BUF_TYPES::BUF_NUM);

    // Note: this sequence of adding is also the sequence of these ports (for DMA)
    constexpr int DMA_BUF_NUM = ARRAY_NUM  * static_cast<int>(ARRRY_BUF_TYPES::BUF_NUM)
                               + VECTOR_NUM * static_cast<int>(VECTOR_BUF_TYPES::BUF_NUM)
                               + DRAM_NUM   * static_cast<int>(DRAM_BUF_TYPES::BUF_NUM)
                               + NOC_NUM    * static_cast<int>(NOC_BUF_TYPES::BUF_NUM)
                               + static_cast<int>(OTHER_BUF_TYPES::BUF_NUM);
    constexpr int DMA_OUT_BUF_NUM = ARRAY_NUM  * ARRAY_IN_BUF_NUM
                                   + VECTOR_NUM * VECTOR_IN_BUF_NUM
                                   + DRAM_NUM   * DRAM_IN_BUF_NUM
                                   + NOC_NUM    * NOC_IN_BUF_NUM;
    constexpr int DMA_IN_BUF_NUM = ARRAY_NUM  * ARRAY_OUT_BUF_NUM
                                  + VECTOR_NUM * VECTOR_OUT_BUF_NUM
                                  + DRAM_NUM   * DRAM_OUT_BUF_NUM
                                  + NOC_NUM    * NOC_OUT_BUF_NUM;
    constexpr int OTHER_BUF_NUM = static_cast<int>(OTHER_BUF_TYPES::BUF_NUM);
}

#endif
