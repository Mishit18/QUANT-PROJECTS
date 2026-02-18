#include "types.hpp"
#include "venue.hpp"
#include "execution_engine.hpp"
#include "smart_order_router.hpp"
#include "cost_model.hpp"
#include "queue_model.hpp"
#include "analytics.hpp"
#include "strategies/twap_strategy.hpp"
#include "strategies/vwap_strategy.hpp"
#include "strategies/pov_strategy.hpp"
#include "strategies/aggressive_strategy.hpp"

#include <iostream>
#include <memory>
#include <random>
#include <string>

using namespace execution;

// Create venues with different characteristics
std::vector<std::unique_ptr<Venue>> create_venues(std::mt19937& rng) {
    std::vector<std::unique_ptr<Venue>> venues;
    
    // Venue A: Low latency, tight spread, moderate depth
    VenueConfig config_a;
    config_a.id = "VENUE_A";
    config_a.base_latency_us = 50.0;
    config_a.latency_std_dev_us = 10.0;
    config_a.initial_mid_price = 100.0;
    config_a.initial_spread_bps = 3.0;
    config_a.initial_depth = 5000;
    config_a.num_levels = 5;
    venues.push_back(std::make_unique<Venue>(config_a, rng));
    
    // Venue B: Medium latency, wider spread, high depth
    VenueConfig config_b;
    config_b.id = "VENUE_B";
    config_b.base_latency_us = 150.0;
    config_b.latency_std_dev_us = 30.0;
    config_b.initial_mid_price = 100.0;
    config_b.initial_spread_bps = 5.0;
    config_b.initial_depth = 10000;
    config_b.num_levels = 5;
    venues.push_back(std::make_unique<Venue>(config_b, rng));
    
    // Venue C: High latency, moderate spread, low depth
    VenueConfig config_c;
    config_c.id = "VENUE_C";
    config_c.base_latency_us = 250.0;
    config_c.latency_std_dev_us = 50.0;
    config_c.initial_mid_price = 100.0;
    config_c.initial_spread_bps = 4.0;
    config_c.initial_depth = 3000;
    config_c.num_levels = 5;
    venues.push_back(std::make_unique<Venue>(config_c, rng));
    
    return venues;
}

// Run TWAP strategy
void run_twap_example() {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "Example 1: TWAP\n";
    std::cout << std::string(70, '=') << "\n\n";
    
    std::mt19937 rng(42);
    auto venues = create_venues(rng);
    
    // Configure TWAP
    TWAPStrategy::Config twap_config;
    twap_config.num_slices = 10;
    twap_config.slice_interval_us = 5000000.0;  // 5 seconds
    twap_config.aggression = 0.3;
    
    auto strategy = std::make_unique<TWAPStrategy>(twap_config);
    
    // Configure SOR with cost and queue models
    SmartOrderRouter::Config sor_config;
    CostModel::Config cost_config;
    QueueModel::Config queue_config;
    
    auto sor = std::make_unique<SmartOrderRouter>(sor_config, cost_config, queue_config);
    
    // Configure engine
    ExecutionEngine::Config engine_config;
    engine_config.simulation_step_us = Timestamp(100000);  // 100ms steps
    engine_config.market_activity_rate = 1.0;
    
    ExecutionEngine engine(
        engine_config,
        std::move(venues),
        std::move(strategy),
        std::move(sor)
    );
    
    // Create parent order
    ParentOrder parent;
    parent.id = 1;
    parent.side = Side::BUY;
    parent.total_quantity = 50000;
    parent.filled_quantity = 0;
    parent.strategy = StrategyType::TWAP;
    parent.start_time = Timestamp(0);
    parent.end_time = Timestamp(60000000);  // 60 seconds
    parent.urgency = 0.3;
    
    double arrival_price = 100.0;
    
    // Execute
    engine.execute(parent);
    
    // Analyze
    auto summary = engine.get_summary();
    auto metrics = Analytics::calculate_metrics(parent, engine.get_fills(), summary, arrival_price);
    Analytics::print_report(metrics, "TWAP");
    Analytics::export_fills_csv(engine.get_fills(), "twap_fills.csv");
}

// Run VWAP strategy
void run_vwap_example() {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "Example 2: VWAP\n";
    std::cout << std::string(70, '=') << "\n\n";
    
    std::mt19937 rng(43);
    auto venues = create_venues(rng);
    
    // Configure VWAP
    VWAPStrategy::Config vwap_config;
    vwap_config.num_intervals = 10;
    vwap_config.interval_us = 6000000.0;  // 6 seconds
    vwap_config.aggression = 0.4;
    // U-shaped volume profile
    vwap_config.volume_profile = {0.15, 0.10, 0.08, 0.07, 0.06, 0.06, 0.07, 0.08, 0.10, 0.23};
    
    auto strategy = std::make_unique<VWAPStrategy>(vwap_config);
    
    // Configure SOR with cost and queue models
    SmartOrderRouter::Config sor_config;
    CostModel::Config cost_config;
    QueueModel::Config queue_config;
    
    auto sor = std::make_unique<SmartOrderRouter>(sor_config, cost_config, queue_config);
    
    // Configure engine
    ExecutionEngine::Config engine_config;
    engine_config.simulation_step_us = Timestamp(100000);
    engine_config.market_activity_rate = 1.5;
    
    ExecutionEngine engine(
        engine_config,
        std::move(venues),
        std::move(strategy),
        std::move(sor)
    );
    
    // Create parent order
    ParentOrder parent;
    parent.id = 2;
    parent.side = Side::BUY;
    parent.total_quantity = 75000;
    parent.filled_quantity = 0;
    parent.strategy = StrategyType::VWAP;
    parent.start_time = Timestamp(0);
    parent.end_time = Timestamp(60000000);
    parent.urgency = 0.4;
    
    double arrival_price = 100.0;
    
    // Execute
    engine.execute(parent);
    
    // Analyze
    auto summary = engine.get_summary();
    auto metrics = Analytics::calculate_metrics(parent, engine.get_fills(), summary, arrival_price);
    Analytics::print_report(metrics, "VWAP");
    Analytics::export_fills_csv(engine.get_fills(), "vwap_fills.csv");
}

// Run Aggressive strategy
void run_aggressive_example() {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "Example 3: Aggressive Liquidity Seeker\n";
    std::cout << std::string(70, '=') << "\n\n";
    
    std::mt19937 rng(44);
    auto venues = create_venues(rng);
    
    // Configure Aggressive strategy
    AggressiveStrategy::Config agg_config;
    agg_config.urgency_threshold = 0.6;
    agg_config.sweep_percentage = 0.4;
    agg_config.order_interval_us = 2000000.0;  // 2 seconds
    agg_config.allow_market_orders = true;
    
    auto strategy = std::make_unique<AggressiveStrategy>(agg_config);
    
    // Configure SOR with cost and queue models
    SmartOrderRouter::Config sor_config;
    CostModel::Config cost_config;
    QueueModel::Config queue_config;
    
    auto sor = std::make_unique<SmartOrderRouter>(sor_config, cost_config, queue_config);
    
    // Configure engine
    ExecutionEngine::Config engine_config;
    engine_config.simulation_step_us = Timestamp(100000);
    engine_config.market_activity_rate = 2.0;
    
    ExecutionEngine engine(
        engine_config,
        std::move(venues),
        std::move(strategy),
        std::move(sor)
    );
    
    // Create parent order with high urgency
    ParentOrder parent;
    parent.id = 3;
    parent.side = Side::BUY;
    parent.total_quantity = 30000;
    parent.filled_quantity = 0;
    parent.strategy = StrategyType::AGGRESSIVE_LIQUIDITY_SEEKER;
    parent.start_time = Timestamp(0);
    parent.end_time = Timestamp(30000000);  // 30 seconds (shorter horizon)
    parent.urgency = 0.8;  // High urgency
    
    double arrival_price = 100.0;
    
    // Execute
    engine.execute(parent);
    
    // Analyze
    auto summary = engine.get_summary();
    auto metrics = Analytics::calculate_metrics(parent, engine.get_fills(), summary, arrival_price);
    Analytics::print_report(metrics, "Aggressive Liquidity Seeker");
    Analytics::export_fills_csv(engine.get_fills(), "aggressive_fills.csv");
}

int main(int argc, char* argv[]) {
    std::cout << "\n";
    std::cout << "Execution Engine - Multi-Venue Simulation\n";
    std::cout << "==========================================\n\n";
    
    try {
        // Run all examples
        run_twap_example();
        run_vwap_example();
        run_aggressive_example();
        
        std::cout << "\n" << std::string(70, '=') << "\n";
        std::cout << "All simulations completed\n";
        std::cout << std::string(70, '=') << "\n\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
