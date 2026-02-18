cmake_minimum_required(VERSION 3.10)
project(SmartExecutionEngine VERSION 1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler flags
if(MSVC)
    add_compile_options(/W4 /O2)
else()
    add_compile_options(-Wall -Wextra -O3)
endif()

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Source files
set(SOURCES
    src/order_book/order_book.cpp
    src/venue/venue.cpp
    src/cost_model/cost_model.cpp
    src/queue_model/queue_model.cpp
    src/execution/twap_strategy.cpp
    src/execution/vwap_strategy.cpp
    src/execution/pov_strategy.cpp
    src/execution/aggressive_strategy.cpp
    src/sor/smart_order_router.cpp
    src/execution/execution_engine.cpp
    src/analytics/analytics.cpp
)

# Main executable
add_executable(execution_engine main.cpp ${SOURCES})

# Installation
install(TARGETS execution_engine DESTINATION bin)

# Print build information
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
