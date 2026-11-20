# DRAM Simulator Library

find_package(Threads REQUIRED)

add_library(dram_sim_lib STATIC
    DataManager.cpp
    DRAMSim.cpp
    AddressMapper.cpp
    DRAMRouter.cpp
    OffchipMapper.cpp
    FallbackMapper.cpp
    OffChipDRAM.cpp
    OffChipPacketAdapter.cpp
)

# Include the project root so headers can be found as "DRAM_sim/Foo.h", "utils/Bar.h"
target_include_directories(dram_sim_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${SYSTEMC_INCLUDE_DIR}
)

target_link_libraries(dram_sim_lib PUBLIC
    utils_lib
    ${SYSTEMC_LIBRARY}
    Threads::Threads
    yaml-cpp::yaml-cpp
)

target_compile_features(dram_sim_lib PUBLIC cxx_std_17)

# Inject absolute path to the DRAM data directory so dram_params.h K/V paths
# resolve correctly regardless of the working directory when the binary runs.
# Propagated PUBLIC so targets linking dram_sim_lib (e.g. test_dram) also see it.
target_compile_definitions(dram_sim_lib PUBLIC
    DRAM_DATA_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/DRAM_data"
)
