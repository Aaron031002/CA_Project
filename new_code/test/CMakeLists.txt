# Comprehensive Test Configuration for NoCSim
# This CMakeLists.txt builds a single comprehensive test executable

# Use portable include paths relative to the build directory
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../
    ${CMAKE_CURRENT_SOURCE_DIR}/../NoC_sim
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils
    ${CMAKE_CURRENT_SOURCE_DIR}/../buffer_shared_ptr
    ${CMAKE_CURRENT_SOURCE_DIR}/../fifo_shared_ptr
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/VectorUnit
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/ArrayUnit
)

# Find pthread (required by SystemC on Linux)
find_package(Threads REQUIRED)

# Every target from here down to the matching endif() is a developer test
# whose sources are NOT tracked in this branch (only this CMakeLists.txt is),
# so a student checkout has none of them. ENABLE_DEV_TESTS (OFF by default,
# see ../CMakeLists.txt) keeps `cmake ..` from failing there; the *_Acc.cpp
# auto-discovery at the bottom of this file is outside the block and always
# builds.
if(ENABLE_DEV_TESTS)

# Single comprehensive test executable for NoCSim (consolidates all test types)
add_executable(test_noc unit_test/test_noc.cpp)

# Link all required libraries:
# - noc_sim_lib: Contains NoCSim implementation
# - utils_lib: Provides buffer and fifo utilities  
# - SystemC: Core SystemC library
# - Threads: Required by SystemC on Linux
target_link_libraries(test_noc 
    noc_sim_lib 
    utils_lib 
    ${SYSTEMC_LIBRARY}
    Threads::Threads
)

# Essential compile definitions
target_compile_definitions(test_noc PRIVATE
    -DSYSTEMC_DISABLE_COPYRIGHT_MESSAGE
    -DSC_INCLUDE_DYNAMIC_PROCESSES
)

# Output directory for the test executable
set_target_properties(test_noc PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# -----------------------------------------------------------------------
# Off-chip DRAM NoC routing/latency test (Phase 6a)
# -----------------------------------------------------------------------
add_executable(test_noc_offchip unit_test/test_noc_offchip.cpp)

target_link_libraries(test_noc_offchip
    noc_sim_lib
    utils_lib
    ${SYSTEMC_LIBRARY}
    Threads::Threads
)

target_compile_definitions(test_noc_offchip PRIVATE
    -DSYSTEMC_DISABLE_COPYRIGHT_MESSAGE
    -DSC_INCLUDE_DYNAMIC_PROCESSES
)

set_target_properties(test_noc_offchip PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# -----------------------------------------------------------------------
# Off-chip DRAM node + packetizer/depacketizer test (Phase 6b/6c)
# -----------------------------------------------------------------------
add_executable(test_offchip_dram unit_test/test_offchip_dram.cpp)

target_include_directories(test_offchip_dram PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../
    ${SYSTEMC_INCLUDE_DIR}
)

target_compile_definitions(test_offchip_dram PRIVATE
    SC_INCLUDE_DYNAMIC_PROCESSES
    DRAM_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../DRAM_sim/DRAM_data/test_dram"
)

target_link_libraries(test_offchip_dram
    dram_sim_lib
    noc_sim_lib
    utils_lib
    ${SYSTEMC_LIBRARY}
    Threads::Threads
)

set_target_properties(test_offchip_dram PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Vector unit test
add_executable(test_vec
    unit_test/test_vec.cpp
    ../Core/VectorUnit/VectorUnit.cpp
    ../Core/VectorUnit/VectorOp.cpp
)
target_include_directories(test_vec PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/VectorUnit
    ${CMAKE_CURRENT_SOURCE_DIR}/../NoC_sim
)
target_link_libraries(test_vec
    utils_lib
    ${SYSTEMC_LIBRARY}
    Threads::Threads
)
target_compile_definitions(test_vec PRIVATE
    SC_INCLUDE_DYNAMIC_PROCESSES
)
set_target_properties(test_vec PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Array unit test  –  compiles ArrayUnit directly to avoid core_lib dependency on MainCtrl
add_executable(test_array
    unit_test/test_array.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/ArrayUnit/ArrayUnit.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/ArrayUnit/ArrayOp.cpp
)
target_include_directories(test_array PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/ArrayUnit
    ${CMAKE_CURRENT_SOURCE_DIR}/../NoC_sim
)
target_link_libraries(test_array
    dram_sim_lib
    utils_lib
    ${SYSTEMC_LIBRARY}
    Threads::Threads
)
target_compile_definitions(test_array PRIVATE
    SC_INCLUDE_DYNAMIC_PROCESSES
)
set_target_properties(test_array PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Array + Cache runtime integration check for the weight_src==CACHE path —
# wires a real ArrayUnit to a real Cache (3 ports) to verify the new
# w_buf_req/weight_buf_cache_in ports actually deliver a correct weight tile
# at simulation time, not just that ArrayDec parses the new "wsrc" syntax.
add_executable(test_array_cache_weight
    unit_test/test_array_cache_weight.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/ArrayUnit/ArrayUnit.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/ArrayUnit/ArrayOp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/Cache/Cache.cpp
)
target_include_directories(test_array_cache_weight PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/ArrayUnit
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/Cache
    ${CMAKE_CURRENT_SOURCE_DIR}/../NoC_sim
)
target_link_libraries(test_array_cache_weight
    dram_sim_lib
    utils_lib
    ${SYSTEMC_LIBRARY}
    Threads::Threads
)
target_compile_definitions(test_array_cache_weight PRIVATE
    SC_INCLUDE_DYNAMIC_PROCESSES
)
set_target_properties(test_array_cache_weight PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# DMA test  –  compiles DMA.cpp directly to avoid core_lib dependency on MainCtrl
add_executable(test_dma
    unit_test/test_dma.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/DMA/DMA.cpp
)
target_include_directories(test_dma PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/DMA
    ${CMAKE_CURRENT_SOURCE_DIR}/../
)
target_link_libraries(test_dma
    dram_sim_lib
    utils_lib
    ${SYSTEMC_LIBRARY}
    Threads::Threads
)
target_compile_definitions(test_dma PRIVATE
    SC_INCLUDE_DYNAMIC_PROCESSES
)
set_target_properties(test_dma PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Phase 7b off-chip fetch test -- drives DMA<float> directly (same stub-fifo
# pattern as test_dma above) plus a real DataManager (real AddressMapper/
# OffchipMapper) to exercise the owner-miss -> off-chip-fetch state machine
# end to end, which no other test reaches.
add_executable(test_dma_offchip_fetch
    unit_test/test_dma_offchip_fetch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/DMA/DMA.cpp
)
target_include_directories(test_dma_offchip_fetch PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/DMA
    ${CMAKE_CURRENT_SOURCE_DIR}/../
)
target_link_libraries(test_dma_offchip_fetch
    dram_sim_lib
    noc_sim_lib
    utils_lib
    ${SYSTEMC_LIBRARY}
    Threads::Threads
)
target_compile_definitions(test_dma_offchip_fetch PRIVATE
    SC_INCLUDE_DYNAMIC_PROCESSES
)
set_target_properties(test_dma_offchip_fetch PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# -----------------------------------------------------------------------
# Phase F extension: try_dispatch's group-wide-known-miss shortcut -- when
# the group scan itself already covers owner_of(path) and found it
# non-resident, skip the redundant remote read and go straight to the
# off-chip fetch. Isolated fixture (both requester core 0 AND owner core 1
# get a real capacity budget) so it doesn't disturb test_dma_offchip_fetch's
# own degenerate-resident assumptions.
#
# ENABLE_DYNAMIC_DISPATCH: the student handout disables the
# group-remote DRAM mechanism this test exercises (DataManager's
# dram_group_path load call is compiled out -- see DataManager.cpp), so
# this target is guarded off by default (source file untracked too, see
# .gitignore). Both this and test_dma_skip_group_remote below assume the
# mechanism they test is active.
# -----------------------------------------------------------------------
if(ENABLE_DYNAMIC_DISPATCH)
add_executable(test_dma_group_shortcut
    unit_test/test_dma_group_shortcut.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/DMA/DMA.cpp
)
target_include_directories(test_dma_group_shortcut PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/DMA
    ${CMAKE_CURRENT_SOURCE_DIR}/../
)
target_link_libraries(test_dma_group_shortcut
    dram_sim_lib
    noc_sim_lib
    utils_lib
    ${SYSTEMC_LIBRARY}
    Threads::Threads
)
target_compile_definitions(test_dma_group_shortcut PRIVATE
    SC_INCLUDE_DYNAMIC_PROCESSES
)
set_target_properties(test_dma_group_shortcut PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# -----------------------------------------------------------------------
# DataManagerConfig::skip_group_remote
# compile-time switch on DMA::try_dispatch()'s group-scan call site --
# verifies the switch changes WHICH core a remote read targets (a resident
# non-owner group member when off, the raw owner_of() answer when on),
# proving the group scan is actually skipped rather than merely producing
# an equivalent result.
# -----------------------------------------------------------------------
add_executable(test_dma_skip_group_remote
    unit_test/test_dma_skip_group_remote.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/DMA/DMA.cpp
)
target_include_directories(test_dma_skip_group_remote PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/DMA
    ${CMAKE_CURRENT_SOURCE_DIR}/../
)
target_link_libraries(test_dma_skip_group_remote
    dram_sim_lib
    noc_sim_lib
    utils_lib
    ${SYSTEMC_LIBRARY}
    Threads::Threads
)
target_compile_definitions(test_dma_skip_group_remote PRIVATE
    SC_INCLUDE_DYNAMIC_PROCESSES
)
set_target_properties(test_dma_skip_group_remote PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
endif() # ENABLE_DYNAMIC_DISPATCH

# -----------------------------------------------------------------------
# Phase B regression: an instruction-driven
# recv and a DMA-synthesized off-chip fetch (Flow A) concurrently
# outstanding on the same DMA -- the D1/D2 head-of-line/misattribution
# hazards Phase B fixes. Same stub-fifo pattern as test_dma_offchip_fetch.
# -----------------------------------------------------------------------
add_executable(test_dma_concurrent_recv
    unit_test/test_dma_concurrent_recv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/DMA/DMA.cpp
)
target_include_directories(test_dma_concurrent_recv PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/DMA
    ${CMAKE_CURRENT_SOURCE_DIR}/../
)
target_link_libraries(test_dma_concurrent_recv
    dram_sim_lib
    noc_sim_lib
    utils_lib
    ${SYSTEMC_LIBRARY}
    Threads::Threads
)
target_compile_definitions(test_dma_concurrent_recv PRIVATE
    SC_INCLUDE_DYNAMIC_PROCESSES
)
set_target_properties(test_dma_concurrent_recv PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

add_custom_target(run_all_tests
    COMMAND test_noc all
    DEPENDS test_noc
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    COMMENT "Running all NoCSim test suites..."
)

# Core integration test (wires MainCtrl + ArrayUnit + VectorUnit + DMA + DRAMSim + Router)
add_executable(test_core unit_test/test_core.cpp)
target_link_libraries(test_core
    core_lib
    dram_sim_lib
    router_lib
    utils_lib
    ${SYSTEMC_LIBRARY}
    Threads::Threads
)
target_compile_definitions(test_core PRIVATE
    SC_INCLUDE_DYNAMIC_PROCESSES
)
set_target_properties(test_core PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Cache unit test
add_executable(test_cache unit_test/test_cache.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/Cache/Cache.cpp
)
target_include_directories(test_cache PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/Cache
    ${SYSTEMC_INCLUDE_DIR}
)
target_link_libraries(test_cache
    utils_lib
    ${SYSTEMC_LIBRARY}
    Threads::Threads
)
target_compile_definitions(test_cache PRIVATE
    SC_INCLUDE_DYNAMIC_PROCESSES
)
set_target_properties(test_cache PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# MainCtrl scheduling test — compiles only MainCtrl and its decoders directly
# to avoid pulling in Core.cpp (top-level integration) which has its own
# pending changes unrelated to this test.
#
# ENABLE_TRANSFORMER_TESTS: this test directly injects DynamicReq_t/RPC-
# dispatch scenarios and links SchedulingPolicy.cpp, both removed from this
# worktree -- guarded off by default (source file untracked too, see
# .gitignore).
if(ENABLE_TRANSFORMER_TESTS)
add_executable(test_mainctrl
    unit_test/test_mainctrl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/MainCtrl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/PortBundle.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/Decoder/Decoder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/Decoder/ArrayDec/ArrayDec.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/Decoder/VectorDec/VectorDec.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/Decoder/DMADec/DMADec.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/Decoder/TMUDec/TMUDec.cpp
)
target_include_directories(test_mainctrl PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/Decoder
)
target_link_libraries(test_mainctrl
    noc_sim_lib
    dram_sim_lib
    utils_lib
    ${SYSTEMC_LIBRARY}
    Threads::Threads
)
target_compile_definitions(test_mainctrl PRIVATE
    SC_INCLUDE_DYNAMIC_PROCESSES
)
set_target_properties(test_mainctrl PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
endif() # ENABLE_TRANSFORMER_TESTS

# ENABLE_DYNAMIC_DISPATCH: the 7 targets below (test_hstar_dispatch,
# test_hstar16/32/64_dispatch, test_hstar32/64_3level_dispatch,
# test_schedpolicy_hstar_field) unit-test MainCtrl::buildExpertTopo()/
# resolveHierarchicalGroups() and SchedulingPolicy directly via a
# routing_table_path/scheduling_policy_path constructor seam that the
# the student handout disables -- guarded off by default (source files untracked too, see
# .gitignore).
if(ENABLE_DYNAMIC_DISPATCH)
# Hierarchical star all-reduce RPC dispatch test —
# exercises ONLY MainCtrl::buildExpertTopo() (routing-table parsing) and the
# newly extracted MainCtrl::resolveHierarchicalGroups() (pure group-
# resolution logic, no SystemC dependency) directly, without ever calling
# sc_start(). Mirrors test_mainctrl's source-set pattern (compiles only
# MainCtrl and its direct dependencies, avoiding Core.cpp/ArrayUnit/
# attention/DRAM data entirely) so a routing/dispatch-logic-only change
# never needs the full test_acc_8x8_1_1024 prefill binary (attention +
# 24-layer real GPT-OSS weights + full SystemC simulation, many minutes per
# run) just to verify RPC fan-out correctness.
add_executable(test_hstar_dispatch
    unit_test/test_hstar_dispatch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/MainCtrl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/PortBundle.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/Decoder/Decoder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/Decoder/ArrayDec/ArrayDec.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/Decoder/VectorDec/VectorDec.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/Decoder/DMADec/DMADec.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/Decoder/TMUDec/TMUDec.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/SchedulingPolicy/SchedulingPolicy.cpp
)
target_include_directories(test_hstar_dispatch PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/Decoder
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/SchedulingPolicy
)
target_link_libraries(test_hstar_dispatch
    noc_sim_lib
    dram_sim_lib
    utils_lib
    ${SYSTEMC_LIBRARY}
    Threads::Threads
)
target_compile_definitions(test_hstar_dispatch PRIVATE
    SC_INCLUDE_DYNAMIC_PROCESSES
)
set_target_properties(test_hstar_dispatch PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# S_e=16/32/64 hierarchical star all-reduce RPC dispatch tests
# (generalization of test_hstar_dispatch's S_e=8
# test) — same source-set/seam rationale as test_hstar_dispatch above,
# mirrored exactly for each new S_e.
add_executable(test_hstar16_dispatch
    unit_test/test_hstar16_dispatch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/MainCtrl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/PortBundle.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/Decoder/Decoder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/Decoder/ArrayDec/ArrayDec.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/Decoder/VectorDec/VectorDec.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/Decoder/DMADec/DMADec.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/Decoder/TMUDec/TMUDec.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/SchedulingPolicy/SchedulingPolicy.cpp
)
target_include_directories(test_hstar16_dispatch PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/Decoder
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/SchedulingPolicy
)
target_link_libraries(test_hstar16_dispatch
    noc_sim_lib
    dram_sim_lib
    utils_lib
    ${SYSTEMC_LIBRARY}
    Threads::Threads
)
target_compile_definitions(test_hstar16_dispatch PRIVATE
    SC_INCLUDE_DYNAMIC_PROCESSES
)
set_target_properties(test_hstar16_dispatch PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

add_executable(test_hstar32_dispatch
    unit_test/test_hstar32_dispatch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/MainCtrl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/PortBundle.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/Decoder/Decoder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/Decoder/ArrayDec/ArrayDec.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/Decoder/VectorDec/VectorDec.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/Decoder/DMADec/DMADec.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/Decoder/TMUDec/TMUDec.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/SchedulingPolicy/SchedulingPolicy.cpp
)
target_include_directories(test_hstar32_dispatch PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/Decoder
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/SchedulingPolicy
)
target_link_libraries(test_hstar32_dispatch
    noc_sim_lib
    dram_sim_lib
    utils_lib
    ${SYSTEMC_LIBRARY}
    Threads::Threads
)
target_compile_definitions(test_hstar32_dispatch PRIVATE
    SC_INCLUDE_DYNAMIC_PROCESSES
)
set_target_properties(test_hstar32_dispatch PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# S_e=32 TRUE 3-LEVEL hierarchical star (
# SEND/COLLECTION consistency fix) -- separate target from
# test_hstar32_dispatch (the existing 2-level version, kept as-is).
add_executable(test_hstar32_3level_dispatch
    unit_test/test_hstar32_3level_dispatch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/MainCtrl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/PortBundle.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/Decoder/Decoder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/Decoder/ArrayDec/ArrayDec.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/Decoder/VectorDec/VectorDec.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/Decoder/DMADec/DMADec.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/Decoder/TMUDec/TMUDec.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/SchedulingPolicy/SchedulingPolicy.cpp
)
target_include_directories(test_hstar32_3level_dispatch PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/Decoder
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/SchedulingPolicy
)
target_link_libraries(test_hstar32_3level_dispatch
    noc_sim_lib
    dram_sim_lib
    utils_lib
    ${SYSTEMC_LIBRARY}
    Threads::Threads
)
target_compile_definitions(test_hstar32_3level_dispatch PRIVATE
    SC_INCLUDE_DYNAMIC_PROCESSES
)
set_target_properties(test_hstar32_3level_dispatch PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

add_executable(test_hstar64_dispatch
    unit_test/test_hstar64_dispatch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/MainCtrl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/PortBundle.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/Decoder/Decoder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/Decoder/ArrayDec/ArrayDec.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/Decoder/VectorDec/VectorDec.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/Decoder/DMADec/DMADec.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/Decoder/TMUDec/TMUDec.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/SchedulingPolicy/SchedulingPolicy.cpp
)
target_include_directories(test_hstar64_dispatch PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/Decoder
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/SchedulingPolicy
)
target_link_libraries(test_hstar64_dispatch
    noc_sim_lib
    dram_sim_lib
    utils_lib
    ${SYSTEMC_LIBRARY}
    Threads::Threads
)
target_compile_definitions(test_hstar64_dispatch PRIVATE
    SC_INCLUDE_DYNAMIC_PROCESSES
)
set_target_properties(test_hstar64_dispatch PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# S_e=64 TRUE 3-LEVEL hierarchical star (
# SEND/COLLECTION consistency + multi-level collection fixes) -- separate
# target from test_hstar64_dispatch (the existing 2-level version, kept
# as-is). Direct sibling of test_hstar32_3level_dispatch (N_RELAYS=4).
add_executable(test_hstar64_3level_dispatch
    unit_test/test_hstar64_3level_dispatch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/MainCtrl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/PortBundle.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/Decoder/Decoder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/Decoder/ArrayDec/ArrayDec.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/Decoder/VectorDec/VectorDec.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/Decoder/DMADec/DMADec.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/Decoder/TMUDec/TMUDec.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/SchedulingPolicy/SchedulingPolicy.cpp
)
target_include_directories(test_hstar64_3level_dispatch PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/Decoder
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/SchedulingPolicy
)
target_link_libraries(test_hstar64_3level_dispatch
    noc_sim_lib
    dram_sim_lib
    utils_lib
    ${SYSTEMC_LIBRARY}
    Threads::Threads
)
target_compile_definitions(test_hstar64_3level_dispatch PRIVATE
    SC_INCLUDE_DYNAMIC_PROCESSES
)
set_target_properties(test_hstar64_3level_dispatch PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# SchedulingPolicy hstar_block_offset_base round-trip check
# -- compiles ONLY SchedulingPolicy.cpp (no
# MainCtrl.cpp/Decoder/SystemC elaboration needed at all, SchedulingPolicy
# has no sc_module dependency), mirrors test_array_wsrc_decode's minimal
# decode-only pattern below.
add_executable(test_schedpolicy_hstar_field
    unit_test/test_schedpolicy_hstar_field.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/SchedulingPolicy/SchedulingPolicy.cpp
)
target_include_directories(test_schedpolicy_hstar_field PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/SchedulingPolicy
)
target_link_libraries(test_schedpolicy_hstar_field
    noc_sim_lib
    dram_sim_lib
    utils_lib
    ${SYSTEMC_LIBRARY}
    Threads::Threads
)
target_compile_definitions(test_schedpolicy_hstar_field PRIVATE
    SC_INCLUDE_DYNAMIC_PROCESSES
)
set_target_properties(test_schedpolicy_hstar_field PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
endif() # ENABLE_DYNAMIC_DISPATCH

# ArrayDec "wsrc cache <w_id> <w_reserved>" tail-token decode check —
# decode-only (no SystemC elaboration), mirrors test_mainctrl's decode calls.
add_executable(test_array_wsrc_decode
    unit_test/test_array_wsrc_decode.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/Decoder/ArrayDec/ArrayDec.cpp
)
target_include_directories(test_array_wsrc_decode PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../
)
target_link_libraries(test_array_wsrc_decode
    utils_lib
    ${SYSTEMC_LIBRARY}
    Threads::Threads
)
target_compile_definitions(test_array_wsrc_decode PRIVATE
    SC_INCLUDE_DYNAMIC_PROCESSES
)
set_target_properties(test_array_wsrc_decode PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Decode-only check (same style as test_array_wsrc_decode above) for the new
# rename-on-terminal-read trailing tokens across all four decoders
# (data-operation-inst-array-vector-tmu plan): Array/Vector's "rename
# <new_x_id> <new_w_id>", TMU's "newids <idx> <new_id> ...", and DMA's
# "rename <new_id>" on NoC send / DRAM write.
add_executable(test_rename_decode
    unit_test/test_rename_decode.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/Decoder/ArrayDec/ArrayDec.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/Decoder/VectorDec/VectorDec.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/Decoder/TMUDec/TMUDec.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/MainCtrl/Decoder/DMADec/DMADec.cpp
)
target_include_directories(test_rename_decode PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../
)
target_link_libraries(test_rename_decode
    utils_lib
    ${SYSTEMC_LIBRARY}
    Threads::Threads
)
target_compile_definitions(test_rename_decode PRIVATE
    SC_INCLUDE_DYNAMIC_PROCESSES
)
set_target_properties(test_rename_decode PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# -----------------------------------------------------------------------
# DRAM Simulator test
# -----------------------------------------------------------------------
add_executable(test_dram unit_test/test_dram.cpp)

target_include_directories(test_dram PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../
    ${SYSTEMC_INCLUDE_DIR}
)

target_compile_definitions(test_dram PRIVATE
    SC_INCLUDE_DYNAMIC_PROCESSES
    DRAM_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../DRAM_sim/DRAM_data/test_dram"
)

target_link_libraries(test_dram
    dram_sim_lib
    utils_lib
    ${SYSTEMC_LIBRARY}
    Threads::Threads
)

set_target_properties(test_dram PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# -----------------------------------------------------------------------
# AddressMapper test (DRAM owner-mapping YAML, Phase 1 -- pure C++, no SystemC)
# -----------------------------------------------------------------------
add_executable(test_dram_address_mapper unit_test/test_dram_address_mapper.cpp)

target_include_directories(test_dram_address_mapper PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../
)

target_compile_definitions(test_dram_address_mapper PRIVATE
    DRAM_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../DRAM_sim/DRAM_data/test_dram"
)

target_link_libraries(test_dram_address_mapper
    dram_sim_lib
    utils_lib
)

set_target_properties(test_dram_address_mapper PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# -----------------------------------------------------------------------
# FallbackMapper load()/lookup()/
# is_mapped() -- pure C++, no sc_start() needed (same style as
# test_dram_address_mapper).
#
# ENABLE_DYNAMIC_DISPATCH: the student handout disables
# FallbackMapper/dram_group_path wiring in DataManager (see
# DataManager.cpp) -- these two targets (and test_dram_residency,
# test_moe_pull_e2e below) test mechanisms that are now compiled out of
# the live path, so they're guarded off by default (source files
# untracked too, see .gitignore).
# -----------------------------------------------------------------------
if(ENABLE_DYNAMIC_DISPATCH)
add_executable(test_fallback_mapper unit_test/test_fallback_mapper.cpp)

target_include_directories(test_fallback_mapper PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../
)

target_compile_definitions(test_fallback_mapper PRIVATE
    DRAM_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../DRAM_sim/DRAM_data/test_dram"
)

target_link_libraries(test_fallback_mapper
    dram_sim_lib
    utils_lib
)

set_target_properties(test_fallback_mapper PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# -----------------------------------------------------------------------
# DRAM group + cross-core remote read test (Phase 3, needs sc_start())
# -----------------------------------------------------------------------
add_executable(test_dram_group_remote unit_test/test_dram_group_remote.cpp)

target_include_directories(test_dram_group_remote PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../
    ${SYSTEMC_INCLUDE_DIR}
)

target_compile_definitions(test_dram_group_remote PRIVATE
    SC_INCLUDE_DYNAMIC_PROCESSES
    DRAM_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../DRAM_sim/DRAM_data/test_dram"
)

target_link_libraries(test_dram_group_remote
    dram_sim_lib
    utils_lib
    ${SYSTEMC_LIBRARY}
    Threads::Threads
)

set_target_properties(test_dram_group_remote PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
endif() # ENABLE_DYNAMIC_DISPATCH

# -----------------------------------------------------------------------
# Phase 7a LRU/capacity test (DataManager, pure C++ driven from sc_main --
# no sc_start(), no SC_MODULE instantiated)
# -----------------------------------------------------------------------
add_executable(test_dram_lru unit_test/test_dram_lru.cpp)

target_include_directories(test_dram_lru PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../
    ${SYSTEMC_INCLUDE_DIR}
)

target_compile_definitions(test_dram_lru PRIVATE
    SC_INCLUDE_DYNAMIC_PROCESSES
)

target_link_libraries(test_dram_lru
    dram_sim_lib
    utils_lib
    ${SYSTEMC_LIBRARY}
    Threads::Threads
)

set_target_properties(test_dram_lru PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# -----------------------------------------------------------------------
# Phase D warm-start test (DataManager, pure C++, no sc_start()) -- loads
# the real REAL_decode_B1_S1_TP4_RPC dram_map.yaml/dram_group.bin.
# -----------------------------------------------------------------------
add_executable(test_dram_warm_start unit_test/test_dram_warm_start.cpp)

target_include_directories(test_dram_warm_start PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../
    ${SYSTEMC_INCLUDE_DIR}
)

target_compile_definitions(test_dram_warm_start PRIVATE
    SC_INCLUDE_DYNAMIC_PROCESSES
)

target_link_libraries(test_dram_warm_start
    dram_sim_lib
    utils_lib
    ${SYSTEMC_LIBRARY}
    Threads::Threads
)

set_target_properties(test_dram_warm_start PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# -----------------------------------------------------------------------
# Phase G residency dump/load round-trip --
# driven directly against DataManager<float>, same style as
# unit_test/test_dram_warm_start.cpp above.
#
# ENABLE_DYNAMIC_DISPATCH: see the guard note above test_fallback_mapper --
# residency chaining (dram_residency_in_path) is disabled by default.
# -----------------------------------------------------------------------
if(ENABLE_DYNAMIC_DISPATCH)
add_executable(test_dram_residency unit_test/test_dram_residency.cpp)

target_include_directories(test_dram_residency PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../
    ${SYSTEMC_INCLUDE_DIR}
)

target_compile_definitions(test_dram_residency PRIVATE
    SC_INCLUDE_DYNAMIC_PROCESSES
)

target_link_libraries(test_dram_residency
    dram_sim_lib
    utils_lib
    ${SYSTEMC_LIBRARY}
    Threads::Threads
)

set_target_properties(test_dram_residency PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
endif() # ENABLE_DYNAMIC_DISPATCH

# -----------------------------------------------------------------------
# Phase A: KV eviction write-back loop
# regression -- deferred_/is_evicting requeue must not head-of-line block,
# and must eventually complete once clear_evicting() fires. Drives
# DRAMRouter/DRAMSim/DataManager directly (bypassing DMA), needs sc_start().
# -----------------------------------------------------------------------
add_executable(test_dram_evict_retry unit_test/test_dram_evict_retry.cpp)

target_include_directories(test_dram_evict_retry PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../
    ${SYSTEMC_INCLUDE_DIR}
)

target_compile_definitions(test_dram_evict_retry PRIVATE
    SC_INCLUDE_DYNAMIC_PROCESSES
)

target_link_libraries(test_dram_evict_retry
    dram_sim_lib
    utils_lib
    ${SYSTEMC_LIBRARY}
    Threads::Threads
)

set_target_properties(test_dram_evict_retry PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# -----------------------------------------------------------------------
# Phase F fill-and-serve integration test -- drives a REAL DRAMSim<float> +
# REAL DMA<float> wired together via fill_req_out/fill_resp_in (same pattern
# Core.cpp uses in production), with NoC/Router/OffChipDRAM mocked by hand on
# DMA's other ports (same philosophy as test_dma_offchip_fetch above).
# -----------------------------------------------------------------------
add_executable(test_dram_fill
    unit_test/test_dram_fill.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/DMA/DMA.cpp
)

target_include_directories(test_dram_fill PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/DMA
    ${CMAKE_CURRENT_SOURCE_DIR}/../
    ${SYSTEMC_INCLUDE_DIR}
)

target_compile_definitions(test_dram_fill PRIVATE
    SC_INCLUDE_DYNAMIC_PROCESSES
)

target_link_libraries(test_dram_fill
    dram_sim_lib
    utils_lib
    ${SYSTEMC_LIBRARY}
    Threads::Threads
)

set_target_properties(test_dram_fill PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# -----------------------------------------------------------------------
# Router unit test
# -----------------------------------------------------------------------
add_executable(test_router unit_test/test_router.cpp)

target_include_directories(test_router PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../
    ${CMAKE_CURRENT_SOURCE_DIR}/../NoC_sim
    ${SYSTEMC_INCLUDE_DIR}
)

target_compile_definitions(test_router PRIVATE
    SC_INCLUDE_DYNAMIC_PROCESSES
)

target_link_libraries(test_router
    router_lib
    utils_lib
    ${SYSTEMC_LIBRARY}
    Threads::Threads
)

set_target_properties(test_router PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

add_custom_target(run_basic_tests
    COMMAND test_noc basic
    DEPENDS test_noc
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    COMMENT "Running basic NoCSim tests..."
)

add_custom_target(run_payload_tests
    COMMAND test_noc payload
    DEPENDS test_noc
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    COMMENT "Running payload size tests..."
)

add_custom_target(run_float_tests
    COMMAND test_noc float
    DEPENDS test_noc
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    COMMENT "Running float payload tests..."
)

add_custom_target(run_stress_tests
    COMMAND test_noc stress
    DEPENDS test_noc
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    COMMENT "Running stress tests..."
)

# Print test configuration
message(STATUS "Comprehensive NoCSim test will be built as ${CMAKE_BINARY_DIR}/bin/test_noc")
message(STATUS "Available test suites: all, basic, payload, float, stress")

# ── TMU unit test ─────────────────────────────────────────────────────────────
# Tests TMUOp<float/int> directly (scatter, split, concat) without the full
# TMU SystemC module, so no dependency on TMU.cpp.
add_executable(test_tmu
    unit_test/test_tmu.cpp
    ../Core/TMU/TMUOp.cpp
    ../Core/TMU/TMU.cpp
)
target_include_directories(test_tmu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../
    ${SYSTEMC_INCLUDE_DIR}
)
target_link_libraries(test_tmu
    utils_lib
    ${SYSTEMC_LIBRARY}
    Threads::Threads
)
target_compile_definitions(test_tmu PRIVATE
    SC_INCLUDE_DYNAMIC_PROCESSES
)
set_target_properties(test_tmu PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# TMU + Cache runtime integration check for rename-on-terminal-read
# (data-operation-inst-array-vector-tmu plan) — wires a real TMU to a real
# Cache (2 ports) so a rename's effect on cache_data can actually be
# observed, unlike test_tmu.cpp's fake drain_cache_reqs() stand-in. Exercises
# TMU's fan-in matching specifically, since that's the one path in the
# codebase where a rename could break request/response matching if it were
# still id-content-based instead of position-based.
add_executable(test_tmu_cache_rename
    unit_test/test_tmu_cache_rename.cpp
    ../Core/TMU/TMUOp.cpp
    ../Core/TMU/TMU.cpp
    ../Core/Cache/Cache.cpp
)
target_include_directories(test_tmu_cache_rename PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/TMU
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/Cache
    ${SYSTEMC_INCLUDE_DIR}
)
target_link_libraries(test_tmu_cache_rename
    utils_lib
    ${SYSTEMC_LIBRARY}
    Threads::Threads
)
target_compile_definitions(test_tmu_cache_rename PRIVATE
    SC_INCLUDE_DYNAMIC_PROCESSES
)
set_target_properties(test_tmu_cache_rename PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# ── TOP_K dual-output focused test ───────────────────────────────────────────
add_executable(test_vector
    unit_test/test_vector.cpp
    ../Core/VectorUnit/VectorUnit.cpp
    ../Core/VectorUnit/VectorOp.cpp
)
target_include_directories(test_vector PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../
    ${CMAKE_CURRENT_SOURCE_DIR}/../Core/VectorUnit
    ${CMAKE_CURRENT_SOURCE_DIR}/../NoC_sim
)
target_link_libraries(test_vector
    utils_lib
    ${SYSTEMC_LIBRARY}
    Threads::Threads
)
target_compile_definitions(test_vector PRIVATE
    SC_INCLUDE_DYNAMIC_PROCESSES
)
set_target_properties(test_vector PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# ── Acc integration test (N×N Core + NoCSim) ─────────────────────────────────
if(ENABLE_TRANSFORMER_TESTS)
add_executable(test_acc unit_test/test_acc.cpp)
target_link_libraries(test_acc
    acc_lib
    core_lib
    noc_sim_lib
    router_lib
    dram_sim_lib
    utils_lib
    ${SYSTEMC_LIBRARY}
    Threads::Threads
)
target_compile_definitions(test_acc PRIVATE
    SC_INCLUDE_DYNAMIC_PROCESSES
)
set_target_properties(test_acc PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
endif() # ENABLE_TRANSFORMER_TESTS

# -----------------------------------------------------------------------
# (Part D/E): end-to-end RPC-pull test
# over REAL NoC transport (hand-wired 2-core + NoCSim(mesh_n=2), mirrors
# Acc<T>'s own wiring but skips GlobalSyncManager/DRAMRouter -- this test's
# path doesn't touch either).
#
# ENABLE_DYNAMIC_DISPATCH: this test exercises MainCtrl's RPC-pull/
# processDynamicReq/dynamic_req_in machinery, which the student
# handout disables entirely --
# guarded off by default (source file untracked too, see .gitignore).
# -----------------------------------------------------------------------
if(ENABLE_DYNAMIC_DISPATCH)
add_executable(test_moe_pull_e2e unit_test/test_moe_pull_e2e.cpp)
target_link_libraries(test_moe_pull_e2e
    core_lib
    noc_sim_lib
    router_lib
    dram_sim_lib
    utils_lib
    ${SYSTEMC_LIBRARY}
    Threads::Threads
)
target_compile_definitions(test_moe_pull_e2e PRIVATE
    SC_INCLUDE_DYNAMIC_PROCESSES
)
set_target_properties(test_moe_pull_e2e PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
endif() # ENABLE_DYNAMIC_DISPATCH

# -----------------------------------------------------------------------
# (P1): measured vs theoretical NoC transfer
# latency for a single MXFP4-sized (elem_bits=4) MoE expert weight payload.
# Drives Router<T> directly (no Core/DMA/MainCtrl) against a real NoCSim.
# -----------------------------------------------------------------------
add_executable(test_noc_bandwidth_calibration unit_test/test_noc_bandwidth_calibration.cpp)
target_link_libraries(test_noc_bandwidth_calibration
    noc_sim_lib
    router_lib
    utils_lib
    ${SYSTEMC_LIBRARY}
    Threads::Threads
)
target_compile_definitions(test_noc_bandwidth_calibration PRIVATE
    SC_INCLUDE_DYNAMIC_PROCESSES
)
set_target_properties(test_noc_bandwidth_calibration PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# ── GlobalSyncManager unit test ───────────────────────────────────────────────
add_executable(test_gsync
    unit_test/test_gsync.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../Acc/gsync.cpp
)
target_include_directories(test_gsync PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../
    ${SYSTEMC_INCLUDE_DIR}
)
target_link_libraries(test_gsync
    utils_lib
    ${SYSTEMC_LIBRARY}
    Threads::Threads
)
target_compile_definitions(test_gsync PRIVATE
    SC_INCLUDE_DYNAMIC_PROCESSES
)
set_target_properties(test_gsync PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# ── 8×8 Accelerator integration test (64-core, batch=1 seq=1024) ──────────────
# NOTE (分組 A cleanup): this target's fixture
# (REAL_prefill_B1_S1024, and the batch-swept siblings below/
# test_acc_8x8_2_1024/test_acc_8x8_{4,8,16,32,64}_1024) used to be
# regenerable via gen_real_prefill_1024.py's generate_for_batch(), which is
# now deleted (its non-aggregate rpc_dispatch() path is retired along with
# it). All these targets still build/run against their pre-existing
# fixtures on disk; only regeneration is no longer possible via that script.
if(ENABLE_TRANSFORMER_TESTS)
add_executable(test_acc_8x8_1_1024
    prefill/B1/test_acc_8x8_1_1024.cpp
)
target_include_directories(test_acc_8x8_1_1024 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../   # project root
    ${CMAKE_CURRENT_SOURCE_DIR}       # test/ for acc_8x8.h
    ${SYSTEMC_INCLUDE_DIR}
)
target_compile_definitions(test_acc_8x8_1_1024 PRIVATE
    SC_INCLUDE_DYNAMIC_PROCESSES
    DRAM_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../DRAM_sim/DRAM_data"
    INST_DIR_8X8="${CMAKE_CURRENT_SOURCE_DIR}/../../inst_gen/map_data/REAL_prefill_B1_S1024"
)
target_link_libraries(test_acc_8x8_1_1024
    acc_lib
    core_lib
    noc_sim_lib
    router_lib
    dram_sim_lib
    utils_lib
    ${SYSTEMC_LIBRARY}
    Threads::Threads
)
set_target_properties(test_acc_8x8_1_1024 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# -----------------------------------------------------------------------
# (P0): attention-only baseline (P0 Part A's
# REAL_prefill_B1_S64_ATTNONLY64_L1 fixture) on a full 64-core Acc8x8 mesh,
# with synthetic MoE expert-weight-prefetch background traffic injected
# directly onto Core<T>::packet_out_fifo -- measures attention's own
# completion time under NoC contention from concurrent MoE prefetch.
# -----------------------------------------------------------------------
add_executable(test_p0_attn_moe_concurrency
    unit_test/test_p0_attn_moe_concurrency.cpp
)
target_include_directories(test_p0_attn_moe_concurrency PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${SYSTEMC_INCLUDE_DIR}
)
target_compile_definitions(test_p0_attn_moe_concurrency PRIVATE
    SC_INCLUDE_DYNAMIC_PROCESSES
    DRAM_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../DRAM_sim/DRAM_data"
    INST_DIR_ATTN_ONLY="${CMAKE_CURRENT_SOURCE_DIR}/../../inst_gen/map_data/REAL_prefill_B1_S64_ATTNONLY64_L1"
)
target_link_libraries(test_p0_attn_moe_concurrency
    acc_lib
    core_lib
    noc_sim_lib
    router_lib
    dram_sim_lib
    utils_lib
    ${SYSTEMC_LIBRARY}
    Threads::Threads
)
set_target_properties(test_p0_attn_moe_concurrency PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# ── Shard dispatch latency probe (star/ring topologies, 64-core self-dispatch
# straight to expert 0, skips attention/router/mask-gather -- see
# shard_test/test_m1_latency_probe.cpp's own doc comment). Same full Acc8x8
# DUT/library set as test_acc_8x8_1_1024 (no lightweight alternative
# exercises the real NoC/DMA/Cache/RPC dispatch chain) -- but a SEPARATE
# binary/target, since it takes inst_dir as a required argv[1] (no default
# fixture path, generated on demand via
# inst_gen/gen_code/4tier/gen_m1_latency_probe.py -- see that module's own
# docstring for the hstar_generator_fn contract both the star and ring
# generator families satisfy) and always requires
# SCHEDULING_POLICY_PATH/ROUTING_TABLE_PATH.
add_executable(test_m1_latency_probe
    shard_test/test_m1_latency_probe.cpp
)
target_include_directories(test_m1_latency_probe PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../   # project root
    ${CMAKE_CURRENT_SOURCE_DIR}       # test/ for acc_8x8.h
    ${SYSTEMC_INCLUDE_DIR}
)
target_compile_definitions(test_m1_latency_probe PRIVATE
    SC_INCLUDE_DYNAMIC_PROCESSES
    DRAM_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../DRAM_sim/DRAM_data"
)
target_link_libraries(test_m1_latency_probe
    acc_lib
    core_lib
    noc_sim_lib
    router_lib
    dram_sim_lib
    utils_lib
    ${SYSTEMC_LIBRARY}
    Threads::Threads
)
set_target_properties(test_m1_latency_probe PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# ── 8×8 Accelerator integration test (64-core, batch=1 seq=128 real-prompt
# quicktest, TOKENS_PER_CORE=2) — aggregate 2-stage / 3-mode-tiered variants,
# for "延伸：Prefill T>1 驗證" (the first genuine
# T_leader>1 end-to-end test of the aggregate 2-stage RPC mechanism).
#
# (分組 A cleanup): gen_real_prefill_128rp.py's
# --two-stage/--aggregate flags are gone -- both are now unconditionally
# True (handleAggDispatch is the only dispatch mechanism left), so the
# single-stage ("") and plain-2-stage ("_2stage") variants that used to map
# to REAL_prefill_B1_S128rp / REAL_prefill_B1_S128rp_2STAGE are no longer
# producible by any generator call and are removed here. "_adp" (mapping to
# REAL_prefill_B1_S128rp_ADP) was ALREADY stale before this cleanup --
# gen_real_prefill_128rp.py has never had a --dp-adaptive flag nor produced
# an _ADP-suffixed output dir (only _AGG2STAGE/_3MODE, from its suffix =
# "_3MODE" if three_mode else "_AGG2STAGE" logic) -- removed as dead
# alongside the others. Two variants remain, matching the two real,
# reachable output dirs: "" now means the (always-aggregate) baseline
# arm (--three-mode unset -> _AGG2STAGE), "_3mode" means --three-mode
# (-> _3MODE).
foreach(_VARIANT "" "_3mode")
    if(_VARIANT STREQUAL "")
        set(_IDIR "REAL_prefill_B1_S128rp_AGG2STAGE")
    else()
        set(_IDIR "REAL_prefill_B1_S128rp_3MODE")
    endif()
    set(_TGT test_acc_8x8_1_128rp${_VARIANT})
    add_executable(${_TGT}
        test_acc_8x8_1_128rp.cpp
    )
    target_include_directories(${_TGT} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${SYSTEMC_INCLUDE_DIR}
    )
    target_compile_definitions(${_TGT} PRIVATE
        SC_INCLUDE_DYNAMIC_PROCESSES
        DRAM_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../DRAM_sim/DRAM_data"
        INST_DIR_8X8_128RP="${CMAKE_CURRENT_SOURCE_DIR}/../../inst_gen/map_data/${_IDIR}"
    )
    target_link_libraries(${_TGT}
        acc_lib
        core_lib
        noc_sim_lib
        router_lib
        dram_sim_lib
        utils_lib
        ${SYSTEMC_LIBRARY}
        Threads::Threads
    )
    set_target_properties(${_TGT} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endforeach()

# ── 8×8 Accelerator integration test (64-core, batch=2 seq=1024) ──────────────
add_executable(test_acc_8x8_2_1024
    prefill/B2/test_acc_8x8_2_1024.cpp
)
target_include_directories(test_acc_8x8_2_1024 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../   # project root
    ${CMAKE_CURRENT_SOURCE_DIR}       # test/ for acc_8x8.h
    ${SYSTEMC_INCLUDE_DIR}
)
target_compile_definitions(test_acc_8x8_2_1024 PRIVATE
    SC_INCLUDE_DYNAMIC_PROCESSES
    DRAM_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../DRAM_sim/DRAM_data"
    INST_DIR_8X8_2_1024="${CMAKE_CURRENT_SOURCE_DIR}/../../inst_gen/map_data/REAL_prefill_B2_S1024"
)
target_link_libraries(test_acc_8x8_2_1024
    acc_lib
    core_lib
    noc_sim_lib
    router_lib
    dram_sim_lib
    utils_lib
    ${SYSTEMC_LIBRARY}
    Threads::Threads
)
set_target_properties(test_acc_8x8_2_1024 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
endif() # ENABLE_TRANSFORMER_TESTS

# ── 8×8 Accelerator integration tests (64-core, batch=N seq=1024, N=4/8/16/32/64) ──
if(ENABLE_TRANSFORMER_TESTS)
foreach(_BATCH 4 8 16 32 64)
    set(_TGT  test_acc_8x8_${_BATCH}_1024)
    set(_IDIR "${CMAKE_CURRENT_SOURCE_DIR}/../../inst_gen/map_data/REAL_prefill_B${_BATCH}_S1024")
    add_executable(${_TGT}
        prefill/B${_BATCH}/${_TGT}.cpp
    )
    target_include_directories(${_TGT} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${SYSTEMC_INCLUDE_DIR}
    )
    target_compile_definitions(${_TGT} PRIVATE
        SC_INCLUDE_DYNAMIC_PROCESSES
        DRAM_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../DRAM_sim/DRAM_data"
        INST_DIR_8X8_${_BATCH}_1024="${_IDIR}"
    )
    target_link_libraries(${_TGT}
        acc_lib
        core_lib
        noc_sim_lib
        router_lib
        dram_sim_lib
        utils_lib
        ${SYSTEMC_LIBRARY}
        Threads::Threads
    )
    set_target_properties(${_TGT} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endforeach()
endif() # ENABLE_TRANSFORMER_TESTS

# ── Real-model decode tests (64-core Q-head parallel, 2880-dim) ──────────────
# schedule_table_sop_refactor: test_acc_8x8_64_1_1.cpp (B=1-only, no
# BATCH_SIZE parameterization) and test_acc_8x8_64_{2,4,8,16}_1_chain.cpp
# (four byte-identical-except-for-BATCH_SIZE copies;
# the multi-batch extension -- full DRAM_RESIDENCY_IN/OUT,
# SCHEDULING_POLICY_PATH, SCHEDULING_STATE_IN/OUT, ROUTING_TABLE_PATH
# feature set for run_chained_decode_24layer{,_multibatch}.py's --three-mode/
# --lightweight-tp4/--dp-adaptive arms) are merged into one source file,
# test_acc_8x8_64_decode.cpp -- B=1 is simply BATCH_SIZE=1, a strict subset
# of the other four's existing behavior (see that file's header comment).
# Target names drop the old "_chain" suffix (B=1 never had it) and gain a
# "_real" suffix instead, to avoid colliding with the pre-existing mock-
# series test_acc_8x8_64_{2,4,8,16,32}_1 targets below (inst_gen/64_{B}_1
# fixtures, unrelated to this real-model family) and to read clearly as
# "real-model decode", not implying every caller uses the chain driver.
if(ENABLE_TRANSFORMER_TESTS)
foreach(_BATCH 1 2 4 8 16)
    set(_TGT test_acc_8x8_64_${_BATCH}_1_real)
    add_executable(${_TGT}
        decode/B${_BATCH}/test_acc_8x8_64_${_BATCH}_1.cpp
    )
    target_include_directories(${_TGT} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${SYSTEMC_INCLUDE_DIR}
    )
    target_compile_definitions(${_TGT} PRIVATE
        SC_INCLUDE_DYNAMIC_PROCESSES
        DRAM_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../DRAM_sim/DRAM_data"
        BATCH_SIZE=${_BATCH}
        INST_DIR_8X8_64_N_1="${CMAKE_CURRENT_SOURCE_DIR}/../../inst_gen/REAL_decode_B${_BATCH}_S1"
    )
    target_link_libraries(${_TGT}
        acc_lib
        core_lib
        noc_sim_lib
        router_lib
        dram_sim_lib
        utils_lib
        ${SYSTEMC_LIBRARY}
        Threads::Threads
    )
    set_target_properties(${_TGT} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endforeach()
endif() # ENABLE_TRANSFORMER_TESTS

# -----------------------------------------------------------------------
# Standalone dense-MLP testbench -- DISABLED
# superseded by the *_Acc.cpp
# auto-discovery mechanism below (run_scripts/gen_acc.sh / gen_acc_from_
# case.sh generate a case-specific <prefix>_Acc.cpp, picked up by the
# file(GLOB "*_Acc.cpp") loop right after this comment) -- that is the
# actual intended student workflow, not this fixed-name target. Left
# commented out rather than deleted since it's a real, working alternative
# (mlp/test_acc_8x8_mlp.cpp is still on disk, just gitignored for the
# student handout like the rest of new_code/test/ -- see new_code/test/
# CMakeLists.txt's own tracked-file note); re-enable by uncommenting if
# you want a fixed-name MLP target back instead of/alongside
# the generated ones.
# -----------------------------------------------------------------------
# add_executable(test_acc_8x8_mlp
#     mlp/test_acc_8x8_mlp.cpp
# )
# target_include_directories(test_acc_8x8_mlp PRIVATE
#     ${CMAKE_CURRENT_SOURCE_DIR}/../
#     ${CMAKE_CURRENT_SOURCE_DIR}
#     ${SYSTEMC_INCLUDE_DIR}
# )
# target_compile_definitions(test_acc_8x8_mlp PRIVATE
#     SC_INCLUDE_DYNAMIC_PROCESSES
#     DRAM_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../DRAM_sim/DRAM_data"
# )
# target_link_libraries(test_acc_8x8_mlp
#     acc_lib
#     core_lib
#     noc_sim_lib
#     router_lib
#     dram_sim_lib
#     utils_lib
#     ${SYSTEMC_LIBRARY}
#     Threads::Threads
# )
# set_target_properties(test_acc_8x8_mlp PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
# )

# ── 2-stage (request-grant) decode tests (TP1, batch=1/2/4/8/16) ─────────────
#   INST_DIR overridden to the *_2STAGE inst dirs so each binary defaults to
#   the 2-stage variant. 2-stage output verified byte-identical to
#   single-stage for all 5 batch sizes.
#   schedule_table_sop_refactor: TP2/TP4_RPC sibling targets that used to
#   live in this same foreach (test_acc_8x8_64_${_BATCH}_1_tp2_2stage /
#   _tp4_rpc_2stage / _tp4_rpc_agg2stage / _tp4_rpc_adp) are removed --
#   confirmed pure legacy (the stated cleanup target,
#   unused by any current driver script; TP2/TP4_RPC's *.cpp sources are
#   deleted alongside this).
#   Refactor: the mock-series test_acc_8x8_64_${_BATCH}_1.cpp
#   sources that B=2/4/8/16 used to borrow have been deleted along with the
#   rest of the mock series; all 5 batch sizes now share the same
#   decode/B${_BATCH}/test_acc_8x8_64_${_BATCH}_1.cpp per-batch copy that
#   the real-model decode foreach above uses, with BATCH_SIZE=${_BATCH}.
if(ENABLE_TRANSFORMER_TESTS)
foreach(_BATCH 1 2 4 8 16)
    set(_TGT test_acc_8x8_64_${_BATCH}_1_2stage)
    add_executable(${_TGT} decode/B${_BATCH}/test_acc_8x8_64_${_BATCH}_1.cpp)
    target_include_directories(${_TGT} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../ ${CMAKE_CURRENT_SOURCE_DIR} ${SYSTEMC_INCLUDE_DIR})
    target_compile_definitions(${_TGT} PRIVATE SC_INCLUDE_DYNAMIC_PROCESSES
        DRAM_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../DRAM_sim/DRAM_data"
        BATCH_SIZE=${_BATCH}
        INST_DIR_8X8_64_N_1="${CMAKE_CURRENT_SOURCE_DIR}/../../inst_gen/REAL_decode_B${_BATCH}_S1_2STAGE")
    target_link_libraries(${_TGT} acc_lib core_lib noc_sim_lib router_lib dram_sim_lib utils_lib ${SYSTEMC_LIBRARY} Threads::Threads)
    set_target_properties(${_TGT} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endforeach()
endif() # ENABLE_TRANSFORMER_TESTS

# ── 2-stage (request-grant) prefill test (B=1, S=1024) ───────────────────────
#   Prefill uses handleRPCDispatch (TP1-style, per-token); 2-stage output
#   verified byte-identical to single-stage (80640007 vs 80640753 cycles).
#   (分組 A cleanup): handleRPCDispatch/rpc_dispatch() are
#   now deleted, and gen_real_prefill_1024.py (the generator that produced
#   INST_DIR_8X8's REAL_prefill_B1_S1024_2STAGE below via --two-stage) is
#   deleted too -- this target still builds/runs against the pre-existing
#   fixture directory on disk, but that fixture can no longer be
#   regenerated. Kept rather than removed since it's still a valid
#   historical byte-identical-output regression reference.
if(ENABLE_TRANSFORMER_TESTS)
add_executable(test_acc_8x8_1_1024_2stage
    prefill/B1/test_acc_8x8_1_1024.cpp
)
target_include_directories(test_acc_8x8_1_1024_2stage PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${SYSTEMC_INCLUDE_DIR}
)
target_compile_definitions(test_acc_8x8_1_1024_2stage PRIVATE
    SC_INCLUDE_DYNAMIC_PROCESSES
    DRAM_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../DRAM_sim/DRAM_data"
    INST_DIR_8X8="${CMAKE_CURRENT_SOURCE_DIR}/../../inst_gen/map_data/REAL_prefill_B1_S1024_2STAGE"
)
target_link_libraries(test_acc_8x8_1_1024_2stage
    acc_lib core_lib noc_sim_lib router_lib dram_sim_lib utils_lib
    ${SYSTEMC_LIBRARY} Threads::Threads
)
set_target_properties(test_acc_8x8_1_1024_2stage PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
endif() # ENABLE_TRANSFORMER_TESTS

endif() # ENABLE_DEV_TESTS

# -----------------------------------------------------------------------
# Joint in-simulation prefill->decode test (B=1, S=64, layer 0 SWA, TP1)
# removed the
# 分組 A cleanup: its generator (inst_gen/gen_code/
# gen_prefill_then_decode_1_64.py) was deleted along with the last
# two_stage=False/plain non-aggregate rpc_dispatch() call site it depended
# on, and the user confirmed retiring this test entirely (not keeping it
# runnable against the frozen JOINT_prefill_then_decode_B1_S64_L0 fixture).
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# Auto-discovered Acc8x8 mesh testbenches: every source file matching
# *_Acc.cpp (e.g. run_scripts/gen_acc.sh's --prefix output, or run_scripts/
# gen_acc_from_case.sh's per-config output) becomes its own build target,
# named after the file (minus extension). The capital "Acc" suffix,
# anchored to the very end of the filename right before .cpp, is
# deliberate: this directory already has lowercase "_acc_" substrings in
# unrelated existing files (test_acc_8x8_1_128rp.cpp and other
# test_acc_8x8_*.cpp files, each with their own hand-written target above)
# that a case-insensitive or unanchored *_acc_* pattern would have
# incorrectly matched. Independent of test_acc_8x8_mlp above -- separate
# source files and targets, not a replacement.
#
# The shared Acc8x8<T> implementation (Acc/acc_8x8.h/.cpp -- real
# simulator source, not generated) lives one level up under new_code/Acc/,
# alongside the generic Acc<T>, and is compiled into acc_lib (see
# new_code/Acc/CMakeLists.txt) rather than being listed as a source file
# here -- every *_Acc.cpp target below already links acc_lib.
#
# NOTE: file(GLOB) only rescans at `cmake` configure time, not on plain
# `make`. After adding/removing a *_Acc.cpp file, re-run `cmake ..`
# before building for the new/removed target to take effect.
# -----------------------------------------------------------------------
file(GLOB ACC_TB_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*_Acc.cpp")
foreach(_acc_src ${ACC_TB_SOURCES})
    get_filename_component(_acc_tgt ${_acc_src} NAME_WE)
    add_executable(${_acc_tgt}
        ${_acc_src}
    )
    target_include_directories(${_acc_tgt} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${SYSTEMC_INCLUDE_DIR}
    )
    target_compile_definitions(${_acc_tgt} PRIVATE
        SC_INCLUDE_DYNAMIC_PROCESSES
        DRAM_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../DRAM_sim/DRAM_data"
    )
    target_link_libraries(${_acc_tgt}
        acc_lib
        core_lib
        noc_sim_lib
        router_lib
        dram_sim_lib
        utils_lib
        ${SYSTEMC_LIBRARY}
        Threads::Threads
    )
    set_target_properties(${_acc_tgt} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endforeach()
