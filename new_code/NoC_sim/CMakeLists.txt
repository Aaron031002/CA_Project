# NoCSim Library Configuration
# This CMakeLists.txt is included by the parent cmake project

# Include directories for NoC_sim
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    # ${CMAKE_CURRENT_SOURCE_DIR}/congest_model
    # ${CMAKE_CURRENT_SOURCE_DIR}/Router
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils
)

# Collect all source files
set(NOC_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/NoC_sim.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/NoCLat.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RouteAlgo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/OffChipRouteAlgo.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/Router/Router.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/congest_model/CongestionModel.cpp
)

# Collect all header files
set(NOC_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/NoC_sim.h
    ${CMAKE_CURRENT_SOURCE_DIR}/NoCLat.h
    ${CMAKE_CURRENT_SOURCE_DIR}/RouteAlgo.h
    ${CMAKE_CURRENT_SOURCE_DIR}/OffChipRouteAlgo.h
    ${CMAKE_CURRENT_SOURCE_DIR}/OffChipNoCLat.h
    # ${CMAKE_CURRENT_SOURCE_DIR}/Router/Router.h
    # ${CMAKE_CURRENT_SOURCE_DIR}/congest_model/CongestionModel.h
    ${CMAKE_CURRENT_SOURCE_DIR}/noc_param.h
    ${CMAKE_CURRENT_SOURCE_DIR}/noc_utils.h
)

# Create NoC simulation library
add_library(noc_sim_lib STATIC ${NOC_SOURCES} ${NOC_HEADERS})

# Link SystemC library
target_link_libraries(noc_sim_lib ${SYSTEMC_LIBRARY})

# Router library (compiled separately because it is a template module)
add_library(router_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/Router/Router.cpp
)
target_include_directories(router_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}          # noc_param.h, noc_utils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/..       # project root (utils/, NoC_sim/)
    ${SYSTEMC_INCLUDE_DIR}
)
target_link_libraries(router_lib ${SYSTEMC_LIBRARY})

# Note: Tests are now in the main test/ directory
# and are built by the main CMakeLists.txt
