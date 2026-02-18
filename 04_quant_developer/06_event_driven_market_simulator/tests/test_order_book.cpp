// Basic test structure - requires test framework (Google Test, Catch2, etc.)
// This is a template showing what tests should cover

#include "../src/orderbook/OrderBook.h"
#include "../src/orderbook/Order.h"
#include <cassert>
#include <iostream>

using namespace simulator;

void test_add_order() {
    OrderBook book("AAPL");
    
    auto order = std::make_shared<Order>(1, "AAPL", true, 150.0, 100, 1000);
    book.addOrder(order);
    
    assert(book.getBidDepth() == 1);
    assert(book.getAskDepth() == 0);
    
    auto best_bid = book.getBestBid();
    assert(best_bid != nullptr);
    assert(best_bid->getPrice() == 150.0);
    
    std::cout << "test_add_order: PASSED\n";
}

void test_remove_order() {
    OrderBook book("AAPL");
    
    auto order = std::make_shared<Order>(1, "AAPL", true, 150.0, 100, 1000);
    book.addOrder(order);
    
    bool removed = book.removeOrder(1);
    assert(removed == true);
    assert(book.getBidDepth() == 0);
    
    std::cout << "test_remove_order: PASSED\n";
}

void test_price_time_priority() {
    OrderBook book("AAPL");
    
    // Add orders at same price, different times
    auto order1 = std::make_shared<Order>(1, "AAPL", true, 150.0, 100, 1000);
    auto order2 = std::make_shared<Order>(2, "AAPL", true, 150.0, 100, 2000);
    
    book.addOrder(order1);
    book.addOrder(order2);
    
    auto best_bid = book.getBestBid();
    auto front_order = best_bid->getFrontOrder();
    
    // First order should be at front (time priority)
    assert(front_order->getId() == 1);
    
    std::cout << "test_price_time_priority: PASSED\n";
}

void test_best_bid_ask() {
    OrderBook book("AAPL");
    
    auto bid1 = std::make_shared<Order>(1, "AAPL", true, 150.0, 100, 1000);
    auto bid2 = std::make_shared<Order>(2, "AAPL", true, 149.0, 100, 2000);
    auto ask1 = std::make_shared<Order>(3, "AAPL", false, 151.0, 100, 3000);
    auto ask2 = std::make_shared<Order>(4, "AAPL", false, 152.0, 100, 4000);
    
    book.addOrder(bid1);
    book.addOrder(bid2);
    book.addOrder(ask1);
    book.addOrder(ask2);
    
    // Best bid should be highest price
    auto best_bid = book.getBestBid();
    assert(best_bid->getPrice() == 150.0);
    
    // Best ask should be lowest price
    auto best_ask = book.getBestAsk();
    assert(best_ask->getPrice() == 151.0);
    
    std::cout << "test_best_bid_ask: PASSED\n";
}

int main() {
    std::cout << "Running Order Book Tests\n";
    std::cout << "========================\n\n";
    
    test_add_order();
    test_remove_order();
    test_price_time_priority();
    test_best_bid_ask();
    
    std::cout << "\nAll tests passed!\n";
    return 0;
}
