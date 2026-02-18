#include "order_book.hpp"
#include <iostream>
#include <cassert>
#include <memory>

using namespace execution;

void test_order_book_basic() {
    std::cout << "Testing basic order book operations...\n";
    
    OrderBook book("TEST_VENUE");
    
    // Add buy order
    auto buy_order = std::make_shared<Order>();
    buy_order->id = 1;
    buy_order->side = Side::BUY;
    buy_order->type = OrderType::LIMIT;
    buy_order->price = 100.0;
    buy_order->quantity = 1000;
    buy_order->filled_quantity = 0;
    
    book.add_order(buy_order);
    
    // Add sell order
    auto sell_order = std::make_shared<Order>();
    sell_order->id = 2;
    sell_order->side = Side::SELL;
    sell_order->type = OrderType::LIMIT;
    sell_order->price = 101.0;
    sell_order->quantity = 1000;
    sell_order->filled_quantity = 0;
    
    book.add_order(sell_order);
    
    // Get quote
    Quote quote = book.get_quote(Timestamp(0));
    
    assert(quote.bid_price == 100.0);
    assert(quote.ask_price == 101.0);
    assert(quote.bid_size == 1000);
    assert(quote.ask_size == 1000);
    
    std::cout << "✓ Basic order book test passed\n";
}

void test_order_matching() {
    std::cout << "Testing order matching...\n";
    
    OrderBook book("TEST_VENUE");
    
    // Add resting sell order
    auto sell_order = std::make_shared<Order>();
    sell_order->id = 1;
    sell_order->side = Side::SELL;
    sell_order->type = OrderType::LIMIT;
    sell_order->price = 100.0;
    sell_order->quantity = 1000;
    sell_order->filled_quantity = 0;
    
    book.add_order(sell_order);
    
    // Match with aggressive buy
    auto buy_order = std::make_shared<Order>();
    buy_order->id = 2;
    buy_order->side = Side::BUY;
    buy_order->type = OrderType::LIMIT;
    buy_order->price = 100.0;
    buy_order->quantity = 500;
    buy_order->filled_quantity = 0;
    
    auto fills = book.match_order(buy_order, Timestamp(1000));
    
    assert(fills.size() == 1);
    assert(fills[0].quantity == 500);
    assert(fills[0].price == 100.0);
    assert(buy_order->filled_quantity == 500);
    assert(buy_order->status == OrderStatus::FILLED);
    
    std::cout << "✓ Order matching test passed\n";
}

void test_partial_fills() {
    std::cout << "Testing partial fills...\n";
    
    OrderBook book("TEST_VENUE");
    
    // Add small resting order
    auto sell_order = std::make_shared<Order>();
    sell_order->id = 1;
    sell_order->side = Side::SELL;
    sell_order->type = OrderType::LIMIT;
    sell_order->price = 100.0;
    sell_order->quantity = 300;
    sell_order->filled_quantity = 0;
    
    book.add_order(sell_order);
    
    // Large aggressive buy
    auto buy_order = std::make_shared<Order>();
    buy_order->id = 2;
    buy_order->side = Side::BUY;
    buy_order->type = OrderType::LIMIT;
    buy_order->price = 100.0;
    buy_order->quantity = 1000;
    buy_order->filled_quantity = 0;
    
    auto fills = book.match_order(buy_order, Timestamp(1000));
    
    assert(fills.size() == 1);
    assert(fills[0].quantity == 300);
    assert(buy_order->filled_quantity == 300);
    assert(buy_order->remaining_quantity() == 700);
    assert(buy_order->status == OrderStatus::PARTIALLY_FILLED);
    
    std::cout << "✓ Partial fills test passed\n";
}

void test_price_time_priority() {
    std::cout << "Testing price-time priority...\n";
    
    OrderBook book("TEST_VENUE");
    
    // Add first sell order at 100
    auto sell1 = std::make_shared<Order>();
    sell1->id = 1;
    sell1->side = Side::SELL;
    sell1->type = OrderType::LIMIT;
    sell1->price = 100.0;
    sell1->quantity = 500;
    sell1->filled_quantity = 0;
    book.add_order(sell1);
    
    // Add second sell order at 100 (later time)
    auto sell2 = std::make_shared<Order>();
    sell2->id = 2;
    sell2->side = Side::SELL;
    sell2->type = OrderType::LIMIT;
    sell2->price = 100.0;
    sell2->quantity = 500;
    sell2->filled_quantity = 0;
    book.add_order(sell2);
    
    // Buy should match first order first
    auto buy = std::make_shared<Order>();
    buy->id = 3;
    buy->side = Side::BUY;
    buy->type = OrderType::LIMIT;
    buy->price = 100.0;
    buy->quantity = 600;
    buy->filled_quantity = 0;
    
    auto fills = book.match_order(buy, Timestamp(2000));
    
    // Should fill 500 from first order, 100 from second
    assert(fills.size() == 2);
    assert(sell1->status == OrderStatus::FILLED);
    assert(sell2->filled_quantity == 100);
    
    std::cout << "✓ Price-time priority test passed\n";
}

int main() {
    std::cout << "\n=== Order Book Unit Tests ===\n\n";
    
    try {
        test_order_book_basic();
        test_order_matching();
        test_partial_fills();
        test_price_time_priority();
        
        std::cout << "\n✓ All tests passed!\n\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n✗ Test failed: " << e.what() << "\n\n";
        return 1;
    }
}
