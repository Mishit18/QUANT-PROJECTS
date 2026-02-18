cmake_minimum_required(VERSION 3.20)
project(hft_feed_handler CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Aggressive optimization flags for production
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -flto -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -fsanitize=address,undefined")

# Warning flags
add_compile_options(
    -Wall -Wextra -Wpedantic
    -Wno-unused-parameter
    -fno-exceptions  # No exceptions on hot path
)

include_directories(${PROJECT_SOURCE_DIR}/include)

# Core library
add_library(feedhandler STATIC
    src/network/tcp_listener.cpp
    src/network/udp_listener.cpp
    src/network/socket_utils.cpp
    src/buffer/ring_buffer.cpp
    src/parsers/fix_parser.cpp
    src/parsers/binary_parser.cpp
    src/engine/decoder_thread.cpp
    src/engine/io_thread.cpp
    src/dispatcher/event_dispatcher.cpp
)

# Benchmark executable
add_executable(benchmark
    src/benchmark/benchmark_main.cpp
    src/benchmark/latency_measurement.cpp
)
target_link_libraries(benchmark feedhandler pthread)

# Test executable
add_executable(tests
    tests/test_ring_buffer.cpp
    tests/test_fix_parser.cpp
    tests/test_binary_parser.cpp
    tests/test_main.cpp
)
target_link_libraries(tests feedhandler pthread)

# Main application
add_executable(feed_handler
    src/main.cpp
)
target_link_libraries(feed_handler feedhandler pthread)
