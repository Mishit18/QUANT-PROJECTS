#pragma once

#include "price_level.h"
#include <array>
#include <cstddef>

namespace lob {

constexpr size_t MAX_PRICE_LEVELS = 10'000;

// Flat array-based order book for cache efficiency
// Linear search acceptable for typical depth (50-200 levels)
// Production would use skip list for wider ranges
class OrderBook {
public:
    OrderBook() noexcept;
    
    void add_order(Order* order) noexcept;
    void remove_order(Order* order) noexcept;
    void update_order(Order* order, Quantity old_remaining) noexcept;
    
    [[nodiscard]] PriceLevel* best_bid() noexcept;
    [[nodiscard]] PriceLevel* best_ask() noexcept;
    
    [[nodiscard]] bool can_match_buy(Price price) const noexcept;
    [[nodiscard]] bool can_match_sell(Price price) const noexcept;
    
    [[nodiscard]] Price best_bid_price() const noexcept { return best_bid_price_; }
    [[nodiscard]] Price best_ask_price() const noexcept { return best_ask_price_; }

private:
    PriceLevel* get_or_create_level(Price price, Side side) noexcept;
    void update_best_bid() noexcept;
    void update_best_ask() noexcept;
    
    // Separate storage for bids and asks
    std::array<PriceLevel, MAX_PRICE_LEVELS> bid_levels_;
    std::array<PriceLevel, MAX_PRICE_LEVELS> ask_levels_;
    
    // Active level tracking for fast iteration
    PriceLevel* active_bids_[MAX_PRICE_LEVELS];
    PriceLevel* active_asks_[MAX_PRICE_LEVELS];
    size_t num_active_bids_;
    size_t num_active_asks_;
    
    Price best_bid_price_;
    Price best_ask_price_;
    PriceLevel* best_bid_level_;
    PriceLevel* best_ask_level_;
};

} // namespace lob
