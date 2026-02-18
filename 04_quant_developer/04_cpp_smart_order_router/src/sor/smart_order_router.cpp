#include "smart_order_router.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace execution {

SmartOrderRouter::SmartOrderRouter(
    const Config& config,
    const CostModel::Config& cost_config,
    const QueueModel::Config& queue_config
) : config_(config), 
    cost_model_(cost_config),
    queue_model_(queue_config) {}

VenueId SmartOrderRouter::route_order(
    const Order& order,
    const std::vector<Venue*>& venues,
    Timestamp current_time
) {
    if (venues.empty()) {
        return "";
    }
    
    // Calculate expected cost for each venue
    std::vector<VenueCost> venue_costs;
    for (const auto* venue : venues) {
        VenueCost cost = calculate_venue_cost(order, venue, current_time);
        venue_costs.push_back(cost);
    }
    
    // Find venue with minimum expected cost
    auto best = std::min_element(venue_costs.begin(), venue_costs.end(),
        [](const VenueCost& a, const VenueCost& b) {
            return a.cost_breakdown.total_cost < b.cost_breakdown.total_cost;
        });
    
    if (best != venue_costs.end()) {
        // Update statistics
        stats_.order_count[best->venue_id]++;
        stats_.total_quantity[best->venue_id] += order.quantity;
        
        // Update average cost
        double& avg = stats_.avg_cost[best->venue_id];
        int count = stats_.order_count[best->venue_id];
        avg = (avg * (count - 1) + best->cost_breakdown.total_cost) / count;
        
        // Store cost breakdown for analysis
        last_cost_breakdowns_[best->venue_id] = best->cost_breakdown;
        
        return best->venue_id;
    }
    
    return venues[0]->get_config().id;  // Fallback
}

std::map<VenueId, Quantity> SmartOrderRouter::route_and_split(
    const Order& order,
    const std::vector<Venue*>& venues,
    Timestamp current_time
) {
    std::map<VenueId, Quantity> allocation;
    
    if (venues.empty()) {
        return allocation;
    }
    
    // Calculate costs for all venues
    std::vector<VenueCost> venue_costs;
    for (const auto* venue : venues) {
        VenueCost cost = calculate_venue_cost(order, venue, current_time);
        
        // Only consider venues with sufficient depth
        Quote quote = venue->get_quote(current_time);
        Quantity available = (order.side == Side::BUY) ? quote.ask_size : quote.bid_size;
        
        if (available >= config_.min_depth_threshold) {
            venue_costs.push_back(cost);
        }
    }
    
    if (venue_costs.empty()) {
        // Fallback: route entire order to first venue
        allocation[venues[0]->get_config().id] = order.quantity;
        return allocation;
    }
    
    // Sort by cost (ascending - lower cost is better)
    std::sort(venue_costs.begin(), venue_costs.end(),
        [](const VenueCost& a, const VenueCost& b) {
            return a.cost_breakdown.total_cost < b.cost_breakdown.total_cost;
        });
    
    // Allocate to lowest cost venues first
    // Weight by inverse cost (lower cost gets more allocation)
    double total_inv_cost = 0.0;
    for (const auto& vc : venue_costs) {
        if (vc.cost_breakdown.total_cost > 0) {
            total_inv_cost += 1.0 / vc.cost_breakdown.total_cost;
        }
    }
    
    Quantity remaining = order.quantity;
    for (size_t i = 0; i < venue_costs.size() && remaining > 0; ++i) {
        double inv_cost = (venue_costs[i].cost_breakdown.total_cost > 0) ?
            1.0 / venue_costs[i].cost_breakdown.total_cost : 1.0;
        double proportion = inv_cost / total_inv_cost;
        Quantity alloc = static_cast<Quantity>(order.quantity * proportion);
        
        // Last venue gets remainder
        if (i == venue_costs.size() - 1) {
            alloc = remaining;
        } else {
            alloc = std::min(alloc, remaining);
        }
        
        if (alloc > 0) {
            allocation[venue_costs[i].venue_id] = alloc;
            remaining -= alloc;
        }
    }
    
    return allocation;
}

const CostModel::CostBreakdown& SmartOrderRouter::get_last_cost_breakdown(
    const VenueId& venue_id
) const {
    static CostModel::CostBreakdown empty;
    auto it = last_cost_breakdowns_.find(venue_id);
    return (it != last_cost_breakdowns_.end()) ? it->second : empty;
}

SmartOrderRouter::VenueCost SmartOrderRouter::calculate_venue_cost(
    const Order& order,
    const Venue* venue,
    Timestamp current_time
) const {
    VenueCost vc;
    vc.venue_id = venue->get_config().id;
    
    Quote quote = venue->get_quote(current_time);
    
    // Calculate expected cost using cost model
    vc.cost_breakdown = cost_model_.calculate_expected_cost(
        order, venue, quote, current_time
    );
    
    // Calculate fill probability using queue model
    int queue_pos = venue->get_order_book().estimate_queue_position(
        order.side, order.price
    );
    Quantity depth = (order.side == Side::BUY) ? quote.ask_size : quote.bid_size;
    double latency = venue->get_config().base_latency_us;
    
    vc.fill_probability = queue_model_.estimate_fill_probability(
        queue_pos,
        order.quantity,
        depth,
        latency,
        10000000.0  // 10 second time horizon
    );
    
    // Adjust cost by fill probability
    // If fill probability is low, effective cost is higher
    if (vc.fill_probability > 0.01) {
        vc.cost_breakdown.total_cost /= vc.fill_probability;
    } else {
        vc.cost_breakdown.total_cost = 1e9;  // Very high cost for low fill prob
    }
    
    return vc;
}

} // namespace execution
