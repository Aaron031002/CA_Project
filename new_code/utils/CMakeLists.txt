# Utils Library Configuration
# This CMakeLists.txt is included by the parent cmake project

# Utils is header-only library
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Create interface library for header-only utilities
add_library(utils_lib INTERFACE)

# Interface include directories
target_include_directories(utils_lib INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include/utils>
)

# Link SystemC to the interface library
target_link_libraries(utils_lib INTERFACE ${SYSTEMC_LIBRARY})

# Optional: Build test executables (only when building standalone)
if(BUILD_UTILS_TESTS)
    find_package(Threads REQUIRED)
    
    # Test executable for FIFO
    add_executable(test_fifo test_fifo.cpp)
    target_link_libraries(test_fifo utils_lib ${SYSTEMC_LIBRARY} Threads::Threads)

    # Test executable for BUFFER
    add_executable(test_buffer test_buffer.cpp)
    target_link_libraries(test_buffer utils_lib ${SYSTEMC_LIBRARY} Threads::Threads)

    # Test executable for Ready/Valid (if needed)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_rdyvld.cpp")
        add_executable(test_rdyvld test_rdyvld.cpp)
        target_link_libraries(test_rdyvld utils_lib ${SYSTEMC_LIBRARY} Threads::Threads)
    endif()
endif()


