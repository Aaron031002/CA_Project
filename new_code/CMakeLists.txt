cmake_minimum_required(VERSION 3.10)
project(CBI_Simulator)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Default to Release build
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ENABLE_DYNAMIC_DISPATCH: OFF by default for the student
# handout. Gates the DRAM-side group-remote/offchip-map/fallback-map/
# residency-chaining mechanisms, none of which the student MLP pipeline
# uses. The guarded code stays in the source tree (not deleted); this
# option is the single switch that keeps it out of both the CMake target
# list and the compiled binary.
#
# MainCtrl's own dynamic-operation mechanism (SchedulingPolicy, aggregate-
# dispatch/grant/RPC fan-out, the online/streaming-softmax Vector
# primitives) that this flag used to also gate has since been permanently
# removed from this worktree (not just disabled) -- turning this flag ON
# no longer restores that part of the pre-handout behavior, only the
# DRAM-side mechanisms above.
option(ENABLE_DYNAMIC_DISPATCH "Enable DRAM group/offchip/fallback/residency mechanisms" OFF)
if(ENABLE_DYNAMIC_DISPATCH)
    add_compile_definitions(ENABLE_DYNAMIC_DISPATCH)
endif()

# ENABLE_TRANSFORMER_TESTS: OFF by default for the student
# handout. Gates every test/CMakeLists.txt target that depends on the
# transformer/attention/MoE/RPC/streaming-softmax machinery removed from
# this worktree -- those targets fail to compile once that code is gone.
# The MLP pipeline's own target (test_acc_8x8_mlp) and the *_Acc.cpp
# auto-discovery mechanism are unaffected by this flag and always build.
option(ENABLE_TRANSFORMER_TESTS "Enable legacy transformer/attention/MoE/RPC/streaming-softmax test targets" OFF)
if(ENABLE_TRANSFORMER_TESTS)
    add_compile_definitions(ENABLE_TRANSFORMER_TESTS)
endif()

# ENABLE_DEV_TESTS: OFF by default for the student handout.
# Gates every hand-written target in test/CMakeLists.txt (unit tests and
# system tests). Their sources under test/ are gitignored, so a student
# checkout doesn't have them and `cmake ..` would fail with "Cannot find
# source file". Turn it on (-DENABLE_DEV_TESTS=ON) only in a working tree
# that has those sources on disk. The ENABLE_TRANSFORMER_TESTS and
# ENABLE_DYNAMIC_DISPATCH test targets sit inside this block too, so they
# also need it. The *_Acc.cpp testbenches are not affected and always build.
option(ENABLE_DEV_TESTS "Build the developer test targets under test/ (sources not shipped to students)" OFF)

# SystemC 3.0 marks SC_HAS_PROCESS as deprecated (IEEE 1666-2023) and warns at
# every use; the simulator still uses it. This only silences that warning.
add_compile_definitions(SC_ALLOW_DEPRECATED_IEEE_API)

# Find SystemC (supports multiple possible install locations). SYSTEMC_HOME
# is the standard SystemC environment variable convention -- set it to your
# SystemC install directory (containing include/systemc.h and
# lib/libsystemc.* or lib-linux64/libsystemc.*) before running cmake.
find_path(SYSTEMC_INCLUDE_DIR systemc.h PATHS
    $ENV{SYSTEMC_HOME}/include
    /usr/local/systemc/include
    /usr/local/include
    /usr/include
)

find_library(SYSTEMC_LIBRARY systemc PATHS
    $ENV{SYSTEMC_HOME}/lib
    $ENV{SYSTEMC_HOME}/lib-linux64
    /usr/local/systemc/lib
    /usr/local/lib
    /usr/lib
)

# Check SystemC availability
if(SYSTEMC_INCLUDE_DIR AND SYSTEMC_LIBRARY)
    message(STATUS "Found SystemC include directory: ${SYSTEMC_INCLUDE_DIR}")
    message(STATUS "Found SystemC library: ${SYSTEMC_LIBRARY}")
    set(SYSTEMC_FOUND TRUE)
else()
    message(FATAL_ERROR "SystemC not found! Set the SYSTEMC_HOME environment "
        "variable to your SystemC install directory "
        "(e.g. export SYSTEMC_HOME=/path/to/systemc) and re-run cmake, "
        "or install SystemC under /usr/local or /usr.")
endif()

# Global include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${SYSTEMC_INCLUDE_DIR}
    utils
)

# === Third-party dependencies ===
# yaml-cpp (vendored under third_party/, gitignored, fetched via git clone --depth 1)
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
set(YAML_CPP_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/yaml-cpp
    ${CMAKE_CURRENT_BINARY_DIR}/third_party/yaml-cpp
)

# === Config-driven headers ===
# new_code/config/*.yaml is the single source of truth for the parameters
# listed in tools/gen_config_headers.py; regenerate the *_params.h headers
# from it at configure time so they're always in sync (see
# new_code/config/hardware.yaml for the migration's Phase 1 scope/rationale).
find_program(PYTHON3_EXECUTABLE python3 REQUIRED)
execute_process(
    COMMAND ${PYTHON3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_config_headers.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RESULT_VARIABLE _gen_config_headers_result
)
if(NOT _gen_config_headers_result EQUAL 0)
    message(FATAL_ERROR "tools/gen_config_headers.py failed (exit ${_gen_config_headers_result})")
endif()

# === Core Components ===
# Utils library (基礎工具類)
add_subdirectory(utils)

# NoC Simulator
add_subdirectory(NoC_sim)

# Core library (如果有的話)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/Core/CMakeLists.txt")
    add_subdirectory(Core)
endif()

# DRAM Simulator (如果有的話)  
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/DRAM_sim/CMakeLists.txt")
    add_subdirectory(DRAM_sim)
endif()

# Acc top-level module
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/Acc/CMakeLists.txt")
    add_subdirectory(Acc)
endif()

# === Testing ===
add_subdirectory(test)

# === Print configuration summary ===
message(STATUS "===============================================")
message(STATUS "Configuration Summary:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  SystemC found: ${SYSTEMC_FOUND}")
message(STATUS "  SystemC include: ${SYSTEMC_INCLUDE_DIR}")
message(STATUS "  SystemC library: ${SYSTEMC_LIBRARY}")
message(STATUS "===============================================")

# === Installation (optional) ===
# Install headers for external use
install(DIRECTORY NoC_sim/ DESTINATION include/NoC_sim
    FILES_MATCHING PATTERN "*.h")
install(DIRECTORY utils/ DESTINATION include/utils  
    FILES_MATCHING PATTERN "*.h")
