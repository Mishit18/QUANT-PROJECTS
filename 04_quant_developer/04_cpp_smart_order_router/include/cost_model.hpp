#pragma once

#include "types.hpp"
#include "venue.hpp"
#include <cmath>

namespace execution {

// Cost model for venue selection
// Estimates expected execution cost per venue
class CostModel {
public:
    struct Config {
        double temporary_impact_coeff;  // k in sqrt model
        double latency_penalty_bps;     // Cost per microsecond of latency
        double queue_risk_coeff;        // Penalty for queue uncertainty
        
        Config() : temporary_impact_coeff(0.1), 
                   latency_penalty_bps(0.001),
                   queue_risk_coeff(0.5) {}
    };
    
    explicit CostModel(const Config& config);
    
    // Calculate expected cost for executing order at venue
    struct CostBreakdown {
        double spread_cost;           // Bid-ask spread cost
        double temporary_impact;      // Market impact (temporary)
        double latency_penalty;       // Cost of latency
        double queue_risk_penalty;    // Queue position uncertainty
        double total_cost;            // Sum of all components
        
        CostBreakdown() : spread_cost(0), temporary_impact(0), 
                         latency_penalty(0), queue_risk_penalty(0), 
                         total_cost(0) {}
    };
    
    CostBreakdown calculate_expected_cost(
        const Order& order,
        const Venue* venue,
        const Quote& quote,
        Timestamp current_time
    ) const;
    
private:
    Config config_;
    
    // Calculate spread cost based on order type
    double calculate_spread_cost(
        const Order& order,
        const Quote& quote
    ) const;
    
    // Calculate temporary market impact
    // Uses square-root model: impact = k * sigma * sqrt(Q / V)
    double calculate_temporary_impact(
        Quantity order_size,
        const Quote& quote,
        double venue_liquidity
    ) const;
    
    // Calculate latency penalty
    // Higher latency increases adverse selection risk
    double calculate_latency_penalty(
        const Venue* venue,
        const Quote& quote
    ) const;
    
    // Calculate queue risk penalty
    // Based on queue position and fill probability uncertainty
    double calculate_queue_risk_penalty(
        const Order& order,
        const Venue* venue,
        const Quote& quote
    ) const;
    
    // Estimate venue liquidity (depth-based)
    double estimate_venue_liquidity(const Quote& quote) const;
};

} // namespace execution
