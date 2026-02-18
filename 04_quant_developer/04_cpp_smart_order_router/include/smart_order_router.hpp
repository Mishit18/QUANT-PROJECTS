#pragma once

#include "types.hpp"
#include "venue.hpp"
#include "cost_model.hpp"
#include "queue_model.hpp"
#include <vector>
#include <memory>
#include <map>

namespace execution {

// Smart Order Router
// Routes child orders to optimal venues based on expected cost minimization
class SmartOrderRouter {
public:
    struct Config {
        double min_depth_threshold;   // Minimum depth to consider venue
        
        Config() : min_depth_threshold(100) {}
    };
    
    explicit SmartOrderRouter(
        const Config& config,
        const CostModel::Config& cost_config,
        const QueueModel::Config& queue_config
    );
    
    // Route a single order to best venue
    VenueId route_order(
        const Order& order,
        const std::vector<Venue*>& venues,
        Timestamp current_time
    );
    
    // Route and split order across multiple venues if beneficial
    std::map<VenueId, Quantity> route_and_split(
        const Order& order,
        const std::vector<Venue*>& venues,
        Timestamp current_time
    );
    
    // Get routing statistics
    struct RoutingStats {
        std::map<VenueId, int> order_count;
        std::map<VenueId, Quantity> total_quantity;
        std::map<VenueId, double> avg_cost;
    };
    
    const RoutingStats& get_stats() const { return stats_; }
    
    // Get cost breakdown for last routing decision
    const CostModel::CostBreakdown& get_last_cost_breakdown(const VenueId& venue_id) const;

private:
    Config config_;
    RoutingStats stats_;
    CostModel cost_model_;
    QueueModel queue_model_;
    
    // Store last cost breakdowns for analysis
    std::map<VenueId, CostModel::CostBreakdown> last_cost_breakdowns_;
    
    // Venue cost evaluation
    struct VenueCost {
        VenueId venue_id;
        CostModel::CostBreakdown cost_breakdown;
        double fill_probability;
        
        VenueCost() : fill_probability(0.0) {}
    };
    
    // Calculate expected cost for venue
    VenueCost calculate_venue_cost(
        const Order& order,
        const Venue* venue,
        Timestamp current_time
    ) const;
};

} // namespace execution
