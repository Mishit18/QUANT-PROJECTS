#pragma once

#include <cstdint>
#include <cstddef>

namespace hft {

// Cache line size for alignment
constexpr size_t CACHE_LINE_SIZE = 64;

// Alignment macro
#define CACHE_ALIGNED alignas(CACHE_LINE_SIZE)

// Fixed-size types for protocol parsing
using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using i8 = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;

// Price representation (fixed-point, 4 decimal places)
// Example: 12345 represents $1.2345
using Price = i64;
using Quantity = i64;
using SequenceNum = u64;

// Side enum
enum class Side : u8 {
    BUY = 0,
    SELL = 1,
    UNKNOWN = 255
};

// Message types
enum class MsgType : u8 {
    ORDER_BOOK_UPDATE = 1,
    TRADE = 2,
    EXECUTION_REPORT = 3,
    HEARTBEAT = 4,
    UNKNOWN = 255
};

// Result type for error handling without exceptions
enum class Result {
    OK = 0,
    ERROR_INVALID_MESSAGE = 1,
    ERROR_CHECKSUM_FAILED = 2,
    ERROR_INCOMPLETE_MESSAGE = 3,
    ERROR_BUFFER_FULL = 4,
    ERROR_QUEUE_FULL = 5,
    ERROR_SOCKET_ERROR = 6
};

// String view for zero-copy string handling
struct StringView {
    const char* data;
    size_t length;
    
    constexpr StringView() noexcept : data(nullptr), length(0) {}
    constexpr StringView(const char* d, size_t len) noexcept : data(d), length(len) {}
    
    // No ownership, no allocation
    bool equals(const char* str, size_t len) const noexcept {
        if (length != len) return false;
        for (size_t i = 0; i < length; ++i) {
            if (data[i] != str[i]) return false;
        }
        return true;
    }
};

} // namespace hft
