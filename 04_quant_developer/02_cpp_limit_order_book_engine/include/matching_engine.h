#pragma once

#include "order_book.h"
#include "memory_pool.h"
#include <unordered_map>

namespace lob {

// Single-threaded matching engine
// No locks, no atomics - deterministic execution
class MatchingEngine {
public:
    MatchingEngine() noexcept;
    
    Order* submit_order(OrderId order_id, Side side, Price price, Quantity quantity) noexcept;
    bool cancel_order(OrderId order_id) noexcept;
    bool modify_order(OrderId order_id, Quantity new_quantity) noexcept;
    
    [[nodiscard]] uint64_t total_orders() const noexcept { return total_orders_; }
    [[nodiscard]] uint64_t total_matches() const noexcept { return total_matches_; }
    [[nodiscard]] uint64_t total_cancels() const noexcept { return total_cancels_; }
    
    void reset_stats() noexcept {
        total_orders_ = 0;
        total_matches_ = 0;
        total_cancels_ = 0;
    }

private:
    void match_order(Order* order) noexcept;
    void execute_match(Order* aggressive, Order* passive, Quantity match_qty) noexcept;
    
    OrderBook book_;
    MemoryPool<Order, MAX_ORDERS> order_pool_;
    std::unordered_map<OrderId, Order*> order_map_;
    
    uint64_t total_orders_;
    uint64_t total_matches_;
    uint64_t total_cancels_;
};

} // namespace lob
