#include "execution_engine.hpp"
#include <algorithm>
#include <numeric>
#include <iostream>
#include <iomanip>

namespace execution {

ExecutionEngine::ExecutionEngine(
    const Config& config,
    std::vector<std::unique_ptr<Venue>> venues,
    std::unique_ptr<ExecutionStrategy> strategy,
    std::unique_ptr<SmartOrderRouter> sor
) : config_(config),
    venues_(std::move(venues)),
    strategy_(std::move(strategy)),
    sor_(std::move(sor)),
    current_time_(0),
    rng_(12345) {}  // Seeded for determinism

void ExecutionEngine::execute(const ParentOrder& parent_order) {
    // Initialize strategy
    strategy_->initialize(parent_order);
    current_time_ = parent_order.start_time;
    
    std::cout << "Executing " << parent_order.total_quantity 
              << " shares using " << strategy_->get_name() << "\n";
    std::cout << "Horizon: " 
              << (parent_order.end_time - parent_order.start_time).count() / 1000000.0 
              << " seconds\n\n";
    
    ParentOrder mutable_parent = parent_order;
    
    // Main execution loop
    while (current_time_ < parent_order.end_time && 
           mutable_parent.remaining_quantity() > 0 &&
           !strategy_->is_complete(current_time_)) {
        
        // Simulate market activity
        if (config_.enable_market_simulation) {
            for (auto& venue : venues_) {
                venue->simulate_market_activity(current_time_, config_.market_activity_rate);
            }
        }
        
        // Process venue matching
        process_fills();
        
        // Update parent order filled quantity
        Quantity total_filled = 0;
        for (const auto& fill : fills_) {
            total_filled += fill.quantity;
        }
        mutable_parent.filled_quantity = total_filled;
        
        // Generate new orders if strategy dictates
        generate_orders(mutable_parent);
        
        // Advance time
        step();
    }
    
    // Final processing
    process_fills();
    
    std::cout << "\nExecution complete\n";
    std::cout << "Total fills: " << fills_.size() << "\n";
    std::cout << "Total filled: " << mutable_parent.filled_quantity << "\n";
}

void ExecutionEngine::step() {
    current_time_ += config_.simulation_step_us;
}

void ExecutionEngine::process_fills() {
    for (auto& venue : venues_) {
        auto venue_fills = venue->process_orders(current_time_);
        
        // Add to our fills
        fills_.insert(fills_.end(), venue_fills.begin(), venue_fills.end());
        
        // Notify strategy of fills
        for (const auto& fill : venue_fills) {
            strategy_->on_fill(fill, current_time_);
            
            // Log fill
            std::cout << "FILL: " << fill.quantity << " @ " << std::fixed 
                      << std::setprecision(2) << fill.price 
                      << " on " << fill.venue_id 
                      << " at t=" << fill.timestamp.count() / 1000000.0 << "s\n";
        }
    }
}

void ExecutionEngine::generate_orders(const ParentOrder& parent_order) {
    Quantity remaining = parent_order.remaining_quantity();
    
    if (remaining == 0) {
        return;
    }
    
    // Get current market quotes
    auto quotes = get_all_quotes();
    
    // Ask strategy to generate orders
    auto new_orders = strategy_->generate_orders(current_time_, quotes, remaining);
    
    // Route each order to appropriate venue
    for (auto& order : new_orders) {
        VenueId venue_id = sor_->route_order(*order, get_venue_pointers(), current_time_);
        
        // Find venue and submit order
        for (auto& venue : venues_) {
            if (venue->get_config().id == venue_id) {
                venue->submit_order(order, current_time_);
                active_orders_.push_back(order);
                
                std::cout << "ORDER: " << order->quantity << " " 
                          << (order->side == Side::BUY ? "BUY" : "SELL")
                          << " @ " << (order->type == OrderType::MARKET ? "MKT" : std::to_string(order->price))
                          << " -> " << venue_id 
                          << " at t=" << current_time_.count() / 1000000.0 << "s\n";
                break;
            }
        }
    }
}

std::vector<Quote> ExecutionEngine::get_all_quotes() const {
    std::vector<Quote> quotes;
    for (const auto& venue : venues_) {
        quotes.push_back(venue->get_quote(current_time_));
    }
    return quotes;
}

std::vector<Venue*> ExecutionEngine::get_venue_pointers() const {
    std::vector<Venue*> ptrs;
    for (const auto& venue : venues_) {
        ptrs.push_back(venue.get());
    }
    return ptrs;
}

ExecutionEngine::ExecutionSummary ExecutionEngine::get_summary() const {
    ExecutionSummary summary;
    
    if (fills_.empty()) {
        summary.total_filled = 0;
        summary.avg_fill_price = 0.0;
        summary.vwap = 0.0;
        summary.total_time = Timestamp(0);
        summary.num_child_orders = 0;
        summary.fill_rate = 0.0;
        return summary;
    }
    
    // Calculate total filled
    summary.total_filled = 0;
    double total_value = 0.0;
    
    for (const auto& fill : fills_) {
        summary.total_filled += fill.quantity;
        total_value += fill.price * fill.quantity;
        
        // Venue distribution
        summary.venue_distribution[fill.venue_id] += fill.quantity;
        
        // Venue average price
        if (summary.venue_avg_price.find(fill.venue_id) == summary.venue_avg_price.end()) {
            summary.venue_avg_price[fill.venue_id] = 0.0;
        }
    }
    
    // Calculate VWAP
    summary.vwap = total_value / summary.total_filled;
    summary.avg_fill_price = summary.vwap;
    
    // Calculate venue average prices
    for (auto& [venue_id, qty] : summary.venue_distribution) {
        double venue_value = 0.0;
        for (const auto& fill : fills_) {
            if (fill.venue_id == venue_id) {
                venue_value += fill.price * fill.quantity;
            }
        }
        summary.venue_avg_price[venue_id] = venue_value / qty;
    }
    
    // Time metrics
    if (!fills_.empty()) {
        summary.total_time = fills_.back().timestamp - fills_.front().timestamp;
    }
    
    summary.num_child_orders = static_cast<int>(active_orders_.size());
    
    return summary;
}

} // namespace execution
