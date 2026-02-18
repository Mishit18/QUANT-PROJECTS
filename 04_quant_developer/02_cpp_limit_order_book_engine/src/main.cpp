#include "matching_engine.h"
#include "cpu_affinity.h"
#include <iostream>

using namespace lob;

int main() {
    if (!pin_to_core(0)) {
        std::cerr << "Warning: Failed to pin to core 0\n";
    }
    
    MatchingEngine engine;
    
    std::cout << "Limit Order Book - Test Scenario\n";
    std::cout << "=================================\n\n";
    
    // Build order book
    engine.submit_order(1, Side::BUY, 10000, 100);   // Buy 100 @ $100.00
    engine.submit_order(2, Side::BUY, 9950, 200);    // Buy 200 @ $99.50
    engine.submit_order(3, Side::BUY, 9900, 150);    // Buy 150 @ $99.00
    
    engine.submit_order(4, Side::SELL, 10100, 100);  // Sell 100 @ $101.00
    engine.submit_order(5, Side::SELL, 10150, 200);  // Sell 200 @ $101.50
    
    std::cout << "Order book built (3 bids, 2 asks)\n";
    std::cout << "Best bid: $100.00, Best ask: $101.00\n\n";
    
    // Aggressive buy order that matches
    std::cout << "Submitting aggressive buy: 50 @ $101.00\n";
    engine.submit_order(6, Side::BUY, 10100, 50);
    std::cout << "  -> Matched with order 4 (partial fill)\n\n";
    
    // Aggressive sell order that matches
    std::cout << "Submitting aggressive sell: 75 @ $100.00\n";
    engine.submit_order(7, Side::SELL, 10000, 75);
    std::cout << "  -> Matched with order 1 (partial fill)\n\n";
    
    // Cancel order
    std::cout << "Cancelling order 2\n";
    engine.cancel_order(2);
    std::cout << "  -> Order removed from book\n\n";
    
    std::cout << "Final stats:\n";
    std::cout << "  Total orders: " << engine.total_orders() << "\n";
    std::cout << "  Total matches: " << engine.total_matches() << "\n";
    std::cout << "  Total cancels: " << engine.total_cancels() << "\n";
    
    return 0;
}
