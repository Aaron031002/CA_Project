# Core Library for CBI Simulator

# Include directories for the core library
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/ArrayUnit
    ${CMAKE_CURRENT_SOURCE_DIR}/DMA
)

# Add all source files for the core library
set(CORE_SOURCES
    VectorUnit/VectorOp.cpp
    VectorUnit/VectorUnit.cpp
    ArrayUnit/ArrayOp.cpp
    ArrayUnit/ArrayUnit.cpp
    DMA/DMA.cpp
    # TMU
    TMU/TMU.cpp
    TMU/TMUOp.cpp
    # Cache
    Cache/Cache.cpp
    # MainCtrl and its helpers
    MainCtrl/MainCtrl.cpp
    MainCtrl/PortBundle.cpp
    MainCtrl/Decoder/Decoder.cpp
    MainCtrl/Decoder/ArrayDec/ArrayDec.cpp
    MainCtrl/Decoder/VectorDec/VectorDec.cpp
    MainCtrl/Decoder/DMADec/DMADec.cpp
    MainCtrl/Decoder/TMUDec/TMUDec.cpp
    # Top-level Core module (wires everything together)
    Core.cpp
)

# Create the core library
add_library(core_lib ${CORE_SOURCES})

# Public include directories for targets linking against core_lib
target_include_directories(core_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/ArrayUnit
    ${CMAKE_CURRENT_SOURCE_DIR}/VectorUnit
    ${CMAKE_CURRENT_SOURCE_DIR}/DMA
    ${CMAKE_CURRENT_SOURCE_DIR}/TMU
    ${CMAKE_CURRENT_SOURCE_DIR}/Cache
    ${CMAKE_CURRENT_SOURCE_DIR}/MainCtrl
    ${CMAKE_CURRENT_SOURCE_DIR}/MainCtrl/Decoder
)

# SC_INCLUDE_DYNAMIC_PROCESSES required for sc_spawn in DMA and Core
target_compile_definitions(core_lib PUBLIC SC_INCLUDE_DYNAMIC_PROCESSES)

# Core.cpp includes DRAMSim and Router headers, so link against those libs
target_link_libraries(core_lib PUBLIC
    dram_sim_lib
    router_lib
    utils_lib
    ${SYSTEMC_LIBRARY}
)

message(STATUS "Core library 'core_lib' created (Array + Vector + DMA + MainCtrl + Core).")

