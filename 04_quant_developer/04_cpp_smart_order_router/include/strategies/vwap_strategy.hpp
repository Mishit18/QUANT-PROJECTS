#pragma once

#include "execution_strategy.hpp"
#include <vector>

namespace execution {

// Volume-Weighted Average Price strategy
// Adjusts execution pace to match historical volume profile
class VWAPStrategy : public ExecutionStrategy {
public:
    struct Config {
        int num_intervals;        // Number of time intervals
        double interval_us;       // Interval duration in microseconds
        double aggression;        // Order aggression level
        std::vector<double> volume_profile;  // Historical volume distribution (sums to 1.0)
        
        Config() : num_intervals(10), interval_us(60000000.0), aggression(0.4) {
            // Default U-shaped volume profile (higher at open/close)
            volume_profile = {0.15, 0.10, 0.08, 0.07, 0.06, 0.06, 0.07, 0.08, 0.10, 0.23};
        }
    };
    
    explicit VWAPStrategy(const Config& config);
    
    void initialize(const ParentOrder& parent_order) override;
    
    std::vector<std::shared_ptr<Order>> generate_orders(
        Timestamp current_time,
        const std::vector<Quote>& venue_quotes,
        Quantity remaining_quantity
    ) override;
    
    void on_fill(const Fill& fill, Timestamp current_time) override;
    
    bool is_complete(Timestamp current_time) const override;
    
    StrategyType get_type() const override { return StrategyType::VWAP; }
    
    std::string get_name() const override { return "Adaptive VWAP"; }
    
    // Get deviation reasons for analysis
    struct DeviationReason {
        bool spread_widened;
        bool impact_increased;
        bool volume_deviated;
        double spread_threshold_bps;
        double impact_threshold;
        double volume_deviation_pct;
        
        DeviationReason() : spread_widened(false), impact_increased(false),
                           volume_deviated(false), spread_threshold_bps(10.0),
                           impact_threshold(0.01), volume_deviation_pct(0.3) {}
    };
    
    const DeviationReason& get_last_deviation() const { return last_deviation_; }

private:
    Config config_;
    int current_interval_;
    Timestamp last_interval_time_;
    std::vector<Quantity> target_quantities_;  // Target quantity per interval
    Quantity cumulative_target_;
    Quantity cumulative_filled_;
    DeviationReason last_deviation_;
    
    // Track market conditions
    double initial_spread_;
    double expected_volume_;
    Quantity observed_volume_;
    
    // Calculate target quantities based on volume profile
    void calculate_targets();
    
    // Determine order price
    Price determine_order_price(const Quote& quote, Side side) const;
    
    // Get current interval index
    int get_current_interval(Timestamp current_time) const;
    
    // Check if should deviate from VWAP schedule
    bool should_deviate(const std::vector<Quote>& quotes, Timestamp current_time);
    
    // Evaluate spread condition
    bool is_spread_widened(const std::vector<Quote>& quotes) const;
    
    // Evaluate impact condition
    bool is_impact_increased(Quantity order_size, const std::vector<Quote>& quotes) const;
    
    // Evaluate volume condition
    bool is_volume_deviated(Timestamp current_time) const;
};

} // namespace execution
