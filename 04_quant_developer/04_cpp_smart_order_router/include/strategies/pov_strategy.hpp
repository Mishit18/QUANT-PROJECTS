#pragma once

#include "execution_strategy.hpp"

namespace execution {

// Percentage of Volume strategy
// Maintains target participation rate relative to market volume
class POVStrategy : public ExecutionStrategy {
public:
    struct Config {
        double target_participation;  // Target participation rate (0.0 to 1.0)
        double max_participation;     // Maximum allowed participation
        double measurement_window_us; // Window for measuring market volume
        double order_interval_us;     // Time between order submissions
        double aggression;            // Order aggression level
        
        Config() : target_participation(0.10), max_participation(0.30),
                   measurement_window_us(10000000.0), order_interval_us(5000000.0),
                   aggression(0.5) {}
    };
    
    explicit POVStrategy(const Config& config);
    
    void initialize(const ParentOrder& parent_order) override;
    
    std::vector<std::shared_ptr<Order>> generate_orders(
        Timestamp current_time,
        const std::vector<Quote>& venue_quotes,
        Quantity remaining_quantity
    ) override;
    
    void on_fill(const Fill& fill, Timestamp current_time) override;
    
    bool is_complete(Timestamp current_time) const override;
    
    StrategyType get_type() const override { return StrategyType::POV; }
    
    std::string get_name() const override { return "POV"; }
    
    // Update observed market volume
    void update_market_volume(Quantity volume, Timestamp timestamp);

private:
    Config config_;
    Timestamp last_order_time_;
    
    // Market volume tracking
    struct VolumeObservation {
        Quantity volume;
        Timestamp timestamp;
    };
    std::vector<VolumeObservation> volume_history_;
    
    Quantity our_filled_in_window_;
    
    // Calculate recent market volume
    Quantity calculate_market_volume(Timestamp current_time) const;
    
    // Calculate our participation rate
    double calculate_participation_rate(Timestamp current_time) const;
    
    // Determine order size based on market volume
    Quantity calculate_order_size(Quantity market_volume, Quantity remaining) const;
    
    // Determine order price
    Price determine_order_price(const Quote& quote, Side side) const;
    
    // Clean old volume observations
    void clean_old_observations(Timestamp current_time);
};

} // namespace execution
