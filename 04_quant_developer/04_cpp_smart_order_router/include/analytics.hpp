#pragma once

#include "types.hpp"
#include "execution_engine.hpp"
#include <vector>
#include <string>

namespace execution {

// Execution analytics and performance measurement
class Analytics {
public:
    struct PerformanceMetrics {
        // Price metrics
        double arrival_price;
        double vwap;
        double avg_fill_price;
        double implementation_shortfall;
        double implementation_shortfall_bps;
        
        // Slippage breakdown
        double spread_cost;
        double impact_cost;
        double delay_cost;
        double total_slippage_bps;
        
        // Execution metrics
        double fill_rate;
        Quantity total_filled;
        Quantity total_target;
        double time_to_complete_seconds;
        int num_fills;
        int num_venues_used;
        
        // Venue breakdown
        std::map<VenueId, Quantity> venue_quantities;
        std::map<VenueId, double> venue_avg_prices;
        std::map<VenueId, double> venue_fill_rates;
    };
    
    // Calculate comprehensive performance metrics
    static PerformanceMetrics calculate_metrics(
        const ParentOrder& parent_order,
        const std::vector<Fill>& fills,
        const ExecutionEngine::ExecutionSummary& summary,
        double arrival_price
    );
    
    // Print detailed report
    static void print_report(
        const PerformanceMetrics& metrics,
        const std::string& strategy_name
    );
    
    // Export to CSV
    static void export_fills_csv(
        const std::vector<Fill>& fills,
        const std::string& filename
    );
    
private:
    // Calculate spread cost
    static double calculate_spread_cost(
        const std::vector<Fill>& fills,
        double arrival_price
    );
    
    // Estimate market impact
    static double estimate_impact_cost(
        const ParentOrder& parent_order,
        const std::vector<Fill>& fills,
        double arrival_price
    );
    
    // Calculate delay cost
    static double calculate_delay_cost(
        const ParentOrder& parent_order,
        const std::vector<Fill>& fills,
        double arrival_price
    );
};

} // namespace execution
