#include "core/Simulator.h"
#include "core/Event.h"
#include "latency/LatencyModel.h"
#include "replay/MarketDataReplayer.h"
#include <iostream>
#include <memory>

using namespace simulator;

int main() {
    std::cout << "Event-Driven Market Simulator\n";
    std::cout << "==============================\n\n";
    
    // Create latency model (1 microsecond fixed)
    auto latency_model = std::make_shared<FixedLatencyModel>(1000);
    
    // Create simulator
    Simulator sim(latency_model);
    
    // Register fill callback
    sim.onFill([](const OrderFillEvent& fill) {
        std::cout << "FILL: Order " << fill.getOrderId()
                  << " @ " << fill.getFillPrice()
                  << " x " << fill.getFillQuantity()
                  << " at t=" << fill.getTimestamp() << "ns\n";
    });
    
    // Generate synthetic market data
    MarketDataReplayer replayer;
    replayer.generateSyntheticOrders(100, "AAPL", 150.0, 1000000);  // 1ms apart
    
    std::cout << "Scheduling " << replayer.getEvents().size() << " events...\n\n";
    
    // Schedule events
    for (const auto& event : replayer.getEvents()) {
        sim.scheduleEvent(event);
    }
    
    // Run simulation
    std::cout << "Running simulation...\n\n";
    sim.run();
    
    std::cout << "\nSimulation complete.\n";
    std::cout << "Final time: " << sim.getCurrentTime() << " ns\n";
    
    return 0;
}
