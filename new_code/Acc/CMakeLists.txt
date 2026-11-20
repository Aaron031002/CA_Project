# Acc Library – top-level accelerator module (N×N Core + NoCSim + GlobalSyncManager)

add_library(acc_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/Acc.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gsync.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/acc_8x8.cpp
)

target_include_directories(acc_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..   # project root → Acc/Acc.h, Core/Core.h, etc.
    ${SYSTEMC_INCLUDE_DIR}
)

target_compile_definitions(acc_lib PUBLIC SC_INCLUDE_DYNAMIC_PROCESSES)

target_link_libraries(acc_lib PUBLIC
    core_lib
    noc_sim_lib
    router_lib
    dram_sim_lib
    utils_lib
    ${SYSTEMC_LIBRARY}
)

message(STATUS "Acc library 'acc_lib' created (Acc + Core + NoCSim + GlobalSyncManager)")
