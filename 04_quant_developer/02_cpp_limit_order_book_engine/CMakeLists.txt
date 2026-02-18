cmake_minimum_required(VERSION 3.15)
project(ultra_low_latency_lob CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Aggressive optimization for low latency
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -flto -fno-exceptions -fno-rtti")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffast-math -funroll-loops -finline-functions")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fomit-frame-pointer -fstrict-aliasing")

set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wall -Wextra -Wpedantic -fsanitize=address")

include_directories(${PROJECT_SOURCE_DIR}/include)

add_library(lob_core STATIC
    src/order_book.cpp
    src/matching_engine.cpp
    src/memory_pool.cpp
)

add_executable(lob_engine src/main.cpp)
target_link_libraries(lob_engine lob_core)

add_executable(lob_benchmark src/benchmark.cpp)
target_link_libraries(lob_benchmark lob_core)

install(TARGETS lob_engine lob_benchmark DESTINATION bin)
