#pragma once

#include <cstdint>
#include <chrono>

#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace lob {

using Timestamp = uint64_t;

// Cross-platform high-resolution timestamp
// ~20-30ns latency on modern systems
inline Timestamp get_timestamp_ns() noexcept {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
}

// RDTSC for x86/x64 - faster (~5ns) but requires calibration
inline uint64_t rdtsc() noexcept {
#if defined(_MSC_VER)
    return __rdtsc();
#elif defined(__GNUC__) || defined(__clang__)
    uint32_t lo, hi;
    __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
    return (static_cast<uint64_t>(hi) << 32) | lo;
#else
    return get_timestamp_ns();
#endif
}

} // namespace lob
