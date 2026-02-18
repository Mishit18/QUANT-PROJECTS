#pragma once

#include "types.hpp"
#include "venue.hpp"
#include "execution_strategy.hpp"
#include "smart_order_router.hpp"
#include <vector>
#include <memory>
#include <map>

namespace execution {

// Main execution engine
class ExecutionEngine {
public:
    struct Config {
        Timestamp simulation_step_us;  // Time step for simulation
        double market_activity_rate;   // Rate of simulated market activity
        bool enable_market_simulation; // Enable background market activity
        
        Config() : simulation_step_us(100000), market_activity_rate(2.0),
                   enable_market_simulation(true) {}
    };
    
    ExecutionEngine(
        const Config& config,
        std::vector<std::unique_ptr<Venue>> venues,
        std::unique_ptr<ExecutionStrategy> strategy,
        std::unique_ptr<SmartOrderRouter> sor
    );
    
    // Execute parent order
    void execute(const ParentOrder& parent_order);
    
    // Get all fills
    const std::vector<Fill>& get_fills() const { return fills_; }
    
    // Get execution summary
    struct ExecutionSummary {
        Quantity total_filled;
        double avg_fill_price;
        double vwap;
        Timestamp total_time;
        std::map<VenueId, Quantity> venue_distribution;
        std::map<VenueId, double> venue_avg_price;
        int num_child_orders;
        double fill_rate;
    };
    
    ExecutionSummary get_summary() const;
    
    // Get current simulation time
    Timestamp get_current_time() const { return current_time_; }

private:
    Config config_;
    std::vector<std::unique_ptr<Venue>> venues_;
    std::unique_ptr<ExecutionStrategy> strategy_;
    std::unique_ptr<SmartOrderRouter> sor_;
    
    Timestamp current_time_;
    std::vector<Fill> fills_;
    std::vector<std::shared_ptr<Order>> active_orders_;
    
    std::mt19937 rng_;
    
    // Simulation step
    void step();
    
    // Process fills from venues
    void process_fills();
    
    // Generate and route new orders
    void generate_orders(const ParentOrder& parent_order);
    
    // Get quotes from all venues
    std::vector<Quote> get_all_quotes() const;
    
    // Get raw venue pointers for SOR
    std::vector<Venue*> get_venue_pointers() const;
};

} // namespace execution
