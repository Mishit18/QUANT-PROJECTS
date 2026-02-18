#include "analytics.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <algorithm>

namespace execution {

Analytics::PerformanceMetrics Analytics::calculate_metrics(
    const ParentOrder& parent_order,
    const std::vector<Fill>& fills,
    const ExecutionEngine::ExecutionSummary& summary,
    double arrival_price
) {
    PerformanceMetrics metrics;
    
    metrics.arrival_price = arrival_price;
    metrics.vwap = summary.vwap;
    metrics.avg_fill_price = summary.avg_fill_price;
    metrics.total_filled = summary.total_filled;
    metrics.total_target = parent_order.total_quantity;
    metrics.num_fills = static_cast<int>(fills.size());
    metrics.num_venues_used = static_cast<int>(summary.venue_distribution.size());
    
    // Fill rate
    metrics.fill_rate = static_cast<double>(summary.total_filled) / 
                        static_cast<double>(parent_order.total_quantity);
    
    // Time to complete
    if (!fills.empty()) {
        metrics.time_to_complete_seconds = 
            (fills.back().timestamp - fills.front().timestamp).count() / 1000000.0;
    } else {
        metrics.time_to_complete_seconds = 0.0;
    }
    
    // Implementation shortfall
    double sign = (parent_order.side == Side::BUY) ? 1.0 : -1.0;
    metrics.implementation_shortfall = 
        sign * (metrics.avg_fill_price - arrival_price) * summary.total_filled;
    metrics.implementation_shortfall_bps = 
        (metrics.implementation_shortfall / (arrival_price * summary.total_filled)) * 10000.0;
    
    // Slippage components
    metrics.spread_cost = calculate_spread_cost(fills, arrival_price);
    metrics.impact_cost = estimate_impact_cost(parent_order, fills, arrival_price);
    metrics.delay_cost = calculate_delay_cost(parent_order, fills, arrival_price);
    
    double total_slippage = metrics.spread_cost + metrics.impact_cost + metrics.delay_cost;
    metrics.total_slippage_bps = 
        (total_slippage / (arrival_price * summary.total_filled)) * 10000.0;
    
    // Venue breakdown
    metrics.venue_quantities = summary.venue_distribution;
    metrics.venue_avg_prices = summary.venue_avg_price;
    
    for (const auto& [venue_id, qty] : summary.venue_distribution) {
        metrics.venue_fill_rates[venue_id] = 
            static_cast<double>(qty) / static_cast<double>(parent_order.total_quantity);
    }
    
    return metrics;
}

void Analytics::print_report(
    const PerformanceMetrics& metrics,
    const std::string& strategy_name
) {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "Execution Report - " << strategy_name << "\n";
    std::cout << std::string(70, '=') << "\n\n";
    
    // Price metrics
    std::cout << "Price Metrics:\n";
    std::cout << std::string(50, '-') << "\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  Arrival Price:              $" << metrics.arrival_price << "\n";
    std::cout << "  Average Fill Price:         $" << metrics.avg_fill_price << "\n";
    std::cout << "  VWAP:                       $" << metrics.vwap << "\n";
    std::cout << "  Implementation Shortfall:   $" << metrics.implementation_shortfall 
              << " (" << std::setprecision(2) << metrics.implementation_shortfall_bps << " bps)\n\n";
    
    // Slippage breakdown
    std::cout << "Slippage Analysis:\n";
    std::cout << std::string(50, '-') << "\n";
    std::cout << "  Spread Cost:                $" << std::setprecision(4) << metrics.spread_cost << "\n";
    std::cout << "  Impact Cost:                $" << metrics.impact_cost << "\n";
    std::cout << "  Delay Cost:                 $" << metrics.delay_cost << "\n";
    std::cout << "  Total Slippage:             " << std::setprecision(2) 
              << metrics.total_slippage_bps << " bps\n\n";
    
    // Execution metrics
    std::cout << "Execution Metrics:\n";
    std::cout << std::string(50, '-') << "\n";
    std::cout << "  Target Quantity:            " << metrics.total_target << "\n";
    std::cout << "  Filled Quantity:            " << metrics.total_filled << "\n";
    std::cout << "  Fill Rate:                  " << std::setprecision(2) 
              << (metrics.fill_rate * 100.0) << "%\n";
    std::cout << "  Number of Fills:            " << metrics.num_fills << "\n";
    std::cout << "  Time to Complete:           " << std::setprecision(2) 
              << metrics.time_to_complete_seconds << " seconds\n";
    std::cout << "  Venues Used:                " << metrics.num_venues_used << "\n\n";
    
    // Venue breakdown
    std::cout << "Venue Breakdown:\n";
    std::cout << std::string(50, '-') << "\n";
    for (const auto& [venue_id, qty] : metrics.venue_quantities) {
        double pct = metrics.venue_fill_rates.at(venue_id) * 100.0;
        double avg_price = metrics.venue_avg_prices.at(venue_id);
        
        std::cout << "  " << std::left << std::setw(15) << venue_id << ": "
                  << std::right << std::setw(8) << qty << " shares ("
                  << std::setw(5) << std::setprecision(1) << pct << "%) @ $"
                  << std::setprecision(4) << avg_price << "\n";
    }
    
    std::cout << "\n" << std::string(70, '=') << "\n\n";
}

void Analytics::export_fills_csv(
    const std::vector<Fill>& fills,
    const std::string& filename
) {
    std::ofstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << "\n";
        return;
    }
    
    // Header
    file << "timestamp_us,order_id,venue_id,side,price,quantity\n";
    
    // Data
    for (const auto& fill : fills) {
        file << fill.timestamp.count() << ","
             << fill.order_id << ","
             << fill.venue_id << ","
             << (fill.side == Side::BUY ? "BUY" : "SELL") << ","
             << std::fixed << std::setprecision(4) << fill.price << ","
             << fill.quantity << "\n";
    }
    
    file.close();
    std::cout << "Fills exported to: " << filename << "\n";
}

double Analytics::calculate_spread_cost(
    const std::vector<Fill>& fills,
    double arrival_price
) {
    // Simplified: assume half-spread cost per fill
    double total_cost = 0.0;
    
    for (const auto& fill : fills) {
        // Estimate spread as percentage of price
        double estimated_spread = arrival_price * 0.0005;  // 5 bps spread assumption
        total_cost += estimated_spread * fill.quantity / 2.0;
    }
    
    return total_cost;
}

double Analytics::estimate_impact_cost(
    const ParentOrder& parent_order,
    const std::vector<Fill>& fills,
    double arrival_price
) {
    // Simplified square-root impact model
    // Impact = k * sigma * sqrt(Q / V)
    // Where Q = order size, V = daily volume
    
    double k = 0.1;  // Market impact coefficient
    double sigma = arrival_price * 0.01;  // 1% volatility assumption
    double daily_volume = 10000000.0;  // Assumed daily volume
    
    double participation = static_cast<double>(parent_order.total_quantity) / daily_volume;
    double impact = k * sigma * std::sqrt(participation);
    
    return impact * parent_order.total_quantity;
}

double Analytics::calculate_delay_cost(
    const ParentOrder& parent_order,
    const std::vector<Fill>& fills,
    double arrival_price
) {
    // Cost of price drift during execution
    // Simplified: assume linear drift
    
    if (fills.empty()) {
        return 0.0;
    }
    
    double total_cost = 0.0;
    double drift_rate = 0.0001;  // 1 bp per second drift assumption
    
    for (const auto& fill : fills) {
        double time_elapsed = (fill.timestamp - parent_order.start_time).count() / 1000000.0;
        double drift = arrival_price * drift_rate * time_elapsed;
        total_cost += drift * fill.quantity;
    }
    
    return total_cost;
}

} // namespace execution
