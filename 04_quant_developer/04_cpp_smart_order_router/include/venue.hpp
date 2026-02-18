#pragma once

#include "types.hpp"
#include "order_book.hpp"
#include <memory>
#include <random>

namespace execution {

// Venue configuration
struct VenueConfig {
    VenueId id;
    double base_latency_us;      // Base latency in microseconds
    double latency_std_dev_us;   // Latency standard deviation
    double maker_fee_bps;        // Maker fee in basis points
    double taker_fee_bps;        // Taker fee in basis points
    Price initial_mid_price;     // Starting mid price
    Price initial_spread_bps;    // Initial spread in basis points
    Quantity initial_depth;      // Depth at each level
    int num_levels;              // Number of price levels to initialize
    
    VenueConfig() : base_latency_us(100.0), latency_std_dev_us(20.0),
                    maker_fee_bps(0.0), taker_fee_bps(5.0),
                    initial_mid_price(100.0), initial_spread_bps(5.0),
                    initial_depth(10000), num_levels(5) {}
};

// Simulated trading venue
class Venue {
public:
    explicit Venue(const VenueConfig& config, std::mt19937& rng);
    
    // Submit order to venue
    void submit_order(std::shared_ptr<Order> order, Timestamp current_time);
    
    // Process orders (simulate matching with latency)
    std::vector<Fill> process_orders(Timestamp current_time);
    
    // Get current quote
    Quote get_quote(Timestamp current_time) const;
    
    // Get order book
    const OrderBook& get_order_book() const { return order_book_; }
    
    // Get venue configuration
    const VenueConfig& get_config() const { return config_; }
    
    // Simulate market activity (random orders from other participants)
    void simulate_market_activity(Timestamp current_time, double activity_rate);
    
    // Get simulated latency for this venue
    Timestamp get_latency() const;

private:
    VenueConfig config_;
    OrderBook order_book_;
    std::mt19937& rng_;
    
    // Pending orders (simulating network latency)
    struct PendingOrder {
        std::shared_ptr<Order> order;
        Timestamp arrival_time;
    };
    std::vector<PendingOrder> pending_orders_;
    
    // Initialize order book with liquidity
    void initialize_liquidity();
    
    // Order ID counter
    OrderId next_order_id_;
};

} // namespace execution
