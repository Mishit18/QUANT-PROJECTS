#pragma once

#include "execution_strategy.hpp"

namespace execution {

// Aggressive Liquidity Seeker
// Crosses spread and sweeps liquidity when urgency is high
class AggressiveStrategy : public ExecutionStrategy {
public:
    struct Config {
        double urgency_threshold;     // Urgency level to trigger aggressive behavior
        double sweep_percentage;      // Percentage of available liquidity to take
        double order_interval_us;     // Minimum time between orders
        bool allow_market_orders;     // Allow market orders
        int max_price_levels;         // Maximum price levels to sweep
        
        Config() : urgency_threshold(0.7), sweep_percentage(0.5),
                   order_interval_us(1000000.0), allow_market_orders(true),
                   max_price_levels(3) {}
    };
    
    explicit AggressiveStrategy(const Config& config);
    
    void initialize(const ParentOrder& parent_order) override;
    
    std::vector<std::shared_ptr<Order>> generate_orders(
        Timestamp current_time,
        const std::vector<Quote>& venue_quotes,
        Quantity remaining_quantity
    ) override;
    
    void on_fill(const Fill& fill, Timestamp current_time) override;
    
    bool is_complete(Timestamp current_time) const override;
    
    StrategyType get_type() const override { return StrategyType::AGGRESSIVE_LIQUIDITY_SEEKER; }
    
    std::string get_name() const override { return "Aggressive Liquidity Seeker"; }

private:
    Config config_;
    Timestamp last_order_time_;
    
    // Calculate urgency based on time remaining and quantity remaining
    double calculate_urgency(Timestamp current_time, Quantity remaining) const;
    
    // Determine if we should cross the spread
    bool should_cross_spread(double urgency) const;
    
    // Calculate order size based on available liquidity
    Quantity calculate_sweep_size(const Quote& quote, Quantity remaining) const;
};

} // namespace execution
