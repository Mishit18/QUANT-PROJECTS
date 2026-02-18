#pragma once

#include "common/types.hpp"

namespace hft {

// Normalized event structures
// These are cache-line aligned and sized for optimal memory access

struct CACHE_ALIGNED OrderBookUpdate {
    u64 timestamp_ns;      // RDTSC or clock_gettime
    SequenceNum sequence;
    Price price;
    Quantity quantity;
    Side side;
    u8 level;              // 0 = top of book
    char symbol[16];       // Fixed-size, no allocation
    u8 _padding[22];       // Pad to 64 bytes
    
    OrderBookUpdate() noexcept : timestamp_ns(0), sequence(0), price(0), 
                                  quantity(0), side(Side::UNKNOWN), level(0) {
        symbol[0] = '\0';
    }
};
static_assert(sizeof(OrderBookUpdate) == 64, "OrderBookUpdate must be 64 bytes");

struct CACHE_ALIGNED Trade {
    u64 timestamp_ns;
    SequenceNum sequence;
    Price price;
    Quantity quantity;
    Side aggressor_side;
    char symbol[16];
    u8 _padding[23];       // Pad to 64 bytes
    
    Trade() noexcept : timestamp_ns(0), sequence(0), price(0), 
                       quantity(0), aggressor_side(Side::UNKNOWN) {
        symbol[0] = '\0';
    }
};
static_assert(sizeof(Trade) == 64, "Trade must be 64 bytes");

struct CACHE_ALIGNED ExecutionReport {
    u64 timestamp_ns;
    SequenceNum sequence;
    Price price;
    Quantity filled_quantity;
    Quantity leaves_quantity;
    Side side;
    char order_id[16];
    char symbol[16];
    u8 exec_type;          // 0=New, 1=PartialFill, 2=Fill, 3=Canceled
    u8 _padding[6];        // Pad to 64 bytes
    
    ExecutionReport() noexcept : timestamp_ns(0), sequence(0), price(0),
                                  filled_quantity(0), leaves_quantity(0),
                                  side(Side::UNKNOWN), exec_type(0) {
        order_id[0] = '\0';
        symbol[0] = '\0';
    }
};
static_assert(sizeof(ExecutionReport) == 64, "ExecutionReport must be 64 bytes");

// Union for type-safe event passing
struct CACHE_ALIGNED Event {
    MsgType type;
    u8 _padding[7];
    union {
        OrderBookUpdate order_book;
        Trade trade;
        ExecutionReport execution;
    };
    
    Event() noexcept : type(MsgType::UNKNOWN) {}
};

} // namespace hft
