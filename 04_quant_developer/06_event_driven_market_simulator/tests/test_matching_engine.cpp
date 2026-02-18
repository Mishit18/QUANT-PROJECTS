// Matching engine test template

#include "../src/orderbook/MatchingEngine.h"
#include <cassert>
#include <iostream>

using namespace simulator;

void test_simple_match() {
    MatchingEngine engine;
    
    // Add resting sell order
    Order sell_order(1, "AAPL", false, 150.0, 100, 1000);
    auto fills1 = engine.submitOrder(sell_order);
    assert(fills1.empty());  // No match yet
    
    // Add aggressive buy order
    Order buy_order(2, "AAPL", true, 150.0, 100, 2000);
    auto fills2 = engine.submitOrder(buy_order);
    
    // Should generate 2 fills (one for each side)
    assert(fills2.size() == 2);
    assert(fills2[0].quantity == 100);
    assert(fills2[0].price == 150.0);
    
    std::cout << "test_simple_match: PASSED\n";
}

void test_partial_fill() {
    MatchingEngine engine;
    
    // Add resting sell order (100 shares)
    Order sell_order(1, "AAPL", false, 150.0, 100, 1000);
    engine.submitOrder(sell_order);
    
    // Add aggressive buy order (200 shares)
    Order buy_order(2, "AAPL", true, 150.0, 200, 2000);
    auto fills = engine.submitOrder(buy_order);
    
    // Should match 100, leave 100 resting
    assert(fills.size() == 2);
    assert(fills[0].quantity == 100);
    
    // Check remaining order in book
    auto book = engine.getBook("AAPL");
    auto best_bid = book->getBestBid();
    assert(best_bid != nullptr);
    assert(best_bid->getTotalQuantity() == 100);
    
    std::cout << "test_partial_fill: PASSED\n";
}

void test_no_match() {
    MatchingEngine engine;
    
    // Add buy order at 149.0
    Order buy_order(1, "AAPL", true, 149.0, 100, 1000);
    auto fills1 = engine.submitOrder(buy_order);
    assert(fills1.empty());
    
    // Add sell order at 151.0 (no cross)
    Order sell_order(2, "AAPL", false, 151.0, 100, 2000);
    auto fills2 = engine.submitOrder(sell_order);
    assert(fills2.empty());
    
    // Both should be resting
    auto book = engine.getBook("AAPL");
    assert(book->getBidDepth() == 1);
    assert(book->getAskDepth() == 1);
    
    std::cout << "test_no_match: PASSED\n";
}

void test_cancel_order() {
    MatchingEngine engine;
    
    Order order(1, "AAPL", true, 150.0, 100, 1000);
    engine.submitOrder(order);
    
    bool cancelled = engine.cancelOrder(1);
    assert(cancelled == true);
    
    auto book = engine.getBook("AAPL");
    assert(book->getBidDepth() == 0);
    
    std::cout << "test_cancel_order: PASSED\n";
}

int main() {
    std::cout << "Running Matching Engine Tests\n";
    std::cout << "==============================\n\n";
    
    test_simple_match();
    test_partial_fill();
    test_no_match();
    test_cancel_order();
    
    std::cout << "\nAll tests passed!\n";
    return 0;
}
