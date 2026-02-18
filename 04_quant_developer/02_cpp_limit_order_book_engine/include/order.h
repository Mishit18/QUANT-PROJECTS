#pragma once

#include "timestamp.h"
#include <cstdint>

namespace lob {

using Price = int64_t;
using Quantity = uint32_t;
using OrderId = uint64_t;

constexpr Price PRICE_MULTIPLIER = 100;  // 2 decimal places
constexpr Price INVALID_PRICE = -1;

enum class Side : uint8_t {
    BUY = 0,
    SELL = 1
};

enum class OrderStatus : uint8_t {
    NEW = 0,
    PARTIAL_FILL = 1,
    FULL_FILL = 2,
    CANCELLED = 3
};

// 64 bytes - fits in single cache line
struct alignas(64) Order {
    OrderId order_id;
    Price price;
    Quantity quantity;
    Quantity filled_quantity;
    Timestamp timestamp;
    Side side;
    OrderStatus status;
    uint16_t queue_position;
    uint32_t _padding;
    
    // Intrusive list pointers for FIFO queue
    Order* next;
    Order* prev;
    
    Order() noexcept 
        : order_id(0), price(0), quantity(0), filled_quantity(0),
          timestamp(0), side(Side::BUY), status(OrderStatus::NEW),
          queue_position(0), _padding(0), next(nullptr), prev(nullptr) {}
    
    [[nodiscard]] inline Quantity remaining_quantity() const noexcept {
        return quantity - filled_quantity;
    }
    
    [[nodiscard]] inline bool is_buy() const noexcept {
        return side == Side::BUY;
    }
    
    [[nodiscard]] inline bool is_sell() const noexcept {
        return side == Side::SELL;
    }
    
    [[nodiscard]] inline bool is_filled() const noexcept {
        return filled_quantity >= quantity;
    }
};

static_assert(sizeof(Order) == 64, "Order must be exactly 64 bytes");
static_assert(alignof(Order) == 64, "Order must be 64-byte aligned");

} // namespace lob
