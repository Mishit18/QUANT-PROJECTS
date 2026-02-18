#pragma once

#include "execution_strategy.hpp"

namespace execution {

// Time-Weighted Average Price strategy
// Slices parent order evenly across time buckets
class TWAPStrategy : public ExecutionStrategy {
public:
    struct Config {
        int num_slices;           // Number of time slices
        double slice_interval_us; // Time between slices in microseconds
        double aggression;        // 0.0 = passive (limit), 1.0 = aggressive (market)
        
        Config() : num_slices(10), slice_interval_us(60000000.0), aggression(0.3) {}
    };
    
    explicit TWAPStrategy(const Config& config);
    
    void initialize(const ParentOrder& parent_order) override;
    
    std::vector<std::shared_ptr<Order>> generate_orders(
        Timestamp current_time,
        const std::vector<Quote>& venue_quotes,
        Quantity remaining_quantity
    ) override;
    
    void on_fill(const Fill& fill, Timestamp current_time) override;
    
    bool is_complete(Timestamp current_time) const override;
    
    StrategyType get_type() const override { return StrategyType::TWAP; }
    
    std::string get_name() const override { return "TWAP"; }

private:
    Config config_;
    int current_slice_;
    Timestamp last_slice_time_;
    Quantity slice_size_;
    Quantity unfilled_from_previous_slices_;
    
    // Calculate slice size
    void calculate_slice_size();
    
    // Determine order price based on aggression level
    Price determine_order_price(const Quote& quote, Side side) const;
};

} // namespace execution
