cmake_minimum_required(VERSION 3.15)
project(PortfolioRiskEngine VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler-specific flags
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Od /Zi")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

include_directories(${PROJECT_SOURCE_DIR}/src)

file(GLOB_RECURSE SOURCES 
    "src/core/*.cpp"
    "src/engine/*.cpp"
    "src/pricing/*.cpp"
    "src/portfolio/*.cpp"
    "src/risk/*.cpp"
    "src/memory/*.cpp"
    "src/utils/*.cpp"
)

add_executable(risk_engine src/main.cpp ${SOURCES})
if(NOT MSVC)
    target_link_libraries(risk_engine pthread)
endif()

add_executable(benchmark benchmarks/benchmark.cpp ${SOURCES})
if(NOT MSVC)
    target_link_libraries(benchmark pthread)
endif()

enable_testing()
file(GLOB TEST_SOURCES "tests/*.cpp")
foreach(test_source ${TEST_SOURCES})
    get_filename_component(test_name ${test_source} NAME_WE)
    add_executable(${test_name} ${test_source} ${SOURCES})
    if(NOT MSVC)
        target_link_libraries(${test_name} pthread)
    endif()
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

