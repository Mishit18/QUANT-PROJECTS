#pragma once

#include "types.hpp"
#include "venue.hpp"
#include <random>

namespace execution {

// Stochastic queue position model
// Models queue depletion as a stochastic process
class QueueModel {
public:
    struct Config {
        double market_order_arrival_rate;  // Lambda for Poisson process (orders/sec)
        double cancellation_rate;          // Probability of cancellation per time unit
        double queue_decay_rate;           // Rate of queue depletion
        
        Config() : market_order_arrival_rate(10.0),
                   cancellation_rate(0.05),
                   queue_decay_rate(0.1) {}
    };
    
    explicit QueueModel(const Config& config);
    
    // Estimate fill probability for order at given queue position
    // Accounts for:
    // - Market order arrivals (deplete queue)
    // - Cancellations (improve position)
    // - Latency (queue changes during transmission)
    double estimate_fill_probability(
        int queue_position,
        Quantity order_size,
        Quantity available_depth,
        double latency_us,
        double time_horizon_us
    ) const;
    
    // Estimate expected queue position at arrival
    // Accounts for latency-induced queue changes
    int estimate_arrival_queue_position(
        int current_position,
        double latency_us,
        double market_activity_rate
    ) const;
    
    // Calculate expected time to fill
    double estimate_time_to_fill(
        int queue_position,
        Quantity order_size,
        double market_order_rate
    ) const;

private:
    Config config_;
    
    // Calculate queue depletion over time
    double calculate_queue_depletion(
        double time_us,
        double market_order_rate
    ) const;
    
    // Calculate probability of being filled before cancellation
    double calculate_fill_before_cancel_prob(
        int queue_position,
        double time_horizon_us
    ) const;
};

} // namespace execution
