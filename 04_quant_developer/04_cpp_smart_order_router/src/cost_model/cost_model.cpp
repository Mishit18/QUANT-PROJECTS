#include "cost_model.hpp"
#include <algorithm>
#include <cmath>

namespace execution {

CostModel::CostModel(const Config& config) : config_(config) {}

CostModel::CostBreakdown CostModel::calculate_expected_cost(
    const Order& order,
    const Venue* venue,
    const Quote& quote,
    Timestamp current_time
) const {
    CostBreakdown breakdown;
    
    // 1. Spread cost
    breakdown.spread_cost = calculate_spread_cost(order, quote);
    
    // 2. Temporary impact
    double liquidity = estimate_venue_liquidity(quote);
    breakdown.temporary_impact = calculate_temporary_impact(
        order.quantity, quote, liquidity
    );
    
    // 3. Latency penalty
    breakdown.latency_penalty = calculate_latency_penalty(venue, quote);
    
    // 4. Queue risk penalty
    breakdown.queue_risk_penalty = calculate_queue_risk_penalty(
        order, venue, quote
    );
    
    // Total expected cost
    breakdown.total_cost = breakdown.spread_cost + 
                          breakdown.temporary_impact + 
                          breakdown.latency_penalty + 
                          breakdown.queue_risk_penalty;
    
    return breakdown;
}

double CostModel::calculate_spread_cost(
    const Order& order,
    const Quote& quote
) const {
    if (quote.bid_price == 0.0 || quote.ask_price == 0.0) {
        return 1e9;  // Invalid quote, very high cost
    }
    
    double spread = quote.spread();
    
    if (order.type == OrderType::MARKET) {
        // Market order pays full spread
        return spread * order.quantity;
    } else {
        // Limit order pays partial spread depending on aggressiveness
        // Assume limit orders pay half spread on average
        return 0.5 * spread * order.quantity;
    }
}

double CostModel::calculate_temporary_impact(
    Quantity order_size,
    const Quote& quote,
    double venue_liquidity
) const {
    if (venue_liquidity == 0.0 || quote.mid_price() == 0.0) {
        return 1e9;  // Invalid state
    }
    
    // Square-root impact model: impact = k * sigma * sqrt(Q / V)
    // where:
    //   k = impact coefficient
    //   sigma = volatility (estimated from spread)
    //   Q = order size
    //   V = venue liquidity
    
    double sigma = quote.spread();  // Spread as volatility proxy
    double participation = static_cast<double>(order_size) / venue_liquidity;
    
    // Impact in price units
    double impact_price = config_.temporary_impact_coeff * sigma * std::sqrt(participation);
    
    // Total cost = impact * quantity
    return impact_price * order_size;
}

double CostModel::calculate_latency_penalty(
    const Venue* venue,
    const Quote& quote
) const {
    // Higher latency increases adverse selection risk
    // Penalty scales with latency and quote volatility
    
    double latency_us = venue->get_config().base_latency_us;
    double mid_price = quote.mid_price();
    
    if (mid_price == 0.0) {
        return 1e9;
    }
    
    // Penalty in basis points per microsecond
    double penalty_bps = config_.latency_penalty_bps * latency_us;
    
    // Convert to dollar cost (assuming unit quantity for scaling)
    return (penalty_bps / 10000.0) * mid_price;
}

double CostModel::calculate_queue_risk_penalty(
    const Order& order,
    const Venue* venue,
    const Quote& quote
) const {
    // Queue risk: uncertainty about fill probability
    // Higher when:
    // - Queue is long (more orders ahead)
    // - Latency is high (stale information)
    // - Depth is low (less liquidity)
    
    if (order.type == OrderType::MARKET) {
        return 0.0;  // Market orders don't queue
    }
    
    // Estimate queue position
    Quantity depth = (order.side == Side::BUY) ? quote.bid_size : quote.ask_size;
    
    if (depth == 0) {
        return 1e9;  // No liquidity
    }
    
    // Queue uncertainty increases with:
    // 1. Lower depth (less liquidity)
    // 2. Higher latency (stale information)
    
    double depth_factor = 1.0 / (1.0 + std::log(1.0 + depth / 1000.0));
    double latency_factor = venue->get_config().base_latency_us / 100.0;
    
    double queue_risk = config_.queue_risk_coeff * depth_factor * latency_factor;
    
    // Scale by order size and price
    return queue_risk * order.quantity * quote.mid_price() / 10000.0;
}

double CostModel::estimate_venue_liquidity(const Quote& quote) const {
    // Estimate total available liquidity
    // Use both bid and ask depth as proxy
    return static_cast<double>(quote.bid_size + quote.ask_size);
}

} // namespace execution
