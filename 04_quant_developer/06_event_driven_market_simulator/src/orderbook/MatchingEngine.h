#pragma once

#include "Order.h"
#include "OrderBook.h"
#include <vector>
#include <unordered_map>
#include <memory>

namespace simulator {

// Exchange-grade matching engine
// Implements continuous limit order book with price-time priority
class MatchingEngine {
public:
    MatchingEngine() = default;
    
    // Submit order and return fills
    std::vector<Fill> submitOrder(const Order& order_data) {
        std::vector<Fill> fills;
        
        // Get or create order book for symbol
        auto& book = getOrCreateBook(order_data.getSymbol());
        
        // Create order object
        auto order = std::make_shared<Order>(order_data);
        
        // Try to match against opposite side
        if (order->isBuy()) {
            matchBuyOrder(order, book, fills);
        } else {
            matchSellOrder(order, book, fills);
        }
        
        // If order still has remaining quantity, add to book
        if (order->isActive() && order->getRemainingQuantity() > 0) {
            book.addOrder(order);
        }
        
        return fills;
    }
    
    // Cancel order
    bool cancelOrder(uint64_t order_id) {
        // Find which book contains this order
        for (auto& [symbol, book] : books_) {
            auto order = book->getOrder(order_id);
            if (order) {
                order->cancel();
                book->removeOrder(order_id);
                return true;
            }
        }
        return false;
    }
    
    // Amend order (loses time priority)
    bool amendOrder(uint64_t order_id, double new_price, uint64_t new_quantity) {
        // Find and remove order
        for (auto& [symbol, book] : books_) {
            auto order = book->getOrder(order_id);
            if (order) {
                // Remove from book
                book->removeOrder(order_id);
                
                // Create amended order (loses time priority)
                Order amended(order_id, order->getSymbol(), order->isBuy(),
                            new_price, new_quantity, order->getTimestamp());
                
                // Resubmit (will go to back of queue at new price)
                submitOrder(amended);
                return true;
            }
        }
        return false;
    }
    
    // Get order book for symbol
    std::shared_ptr<OrderBook> getBook(const std::string& symbol) {
        auto it = books_.find(symbol);
        return (it != books_.end()) ? it->second : nullptr;
    }

private:
    OrderBook& getOrCreateBook(const std::string& symbol) {
        auto it = books_.find(symbol);
        if (it == books_.end()) {
            auto book = std::make_shared<OrderBook>(symbol);
            books_[symbol] = book;
            return *book;
        }
        return *it->second;
    }
    
    void matchBuyOrder(std::shared_ptr<Order> order, OrderBook& book,
                      std::vector<Fill>& fills) {
        // Match against asks (sell orders)
        while (order->getRemainingQuantity() > 0) {
            auto best_ask = book.getBestAsk();
            if (!best_ask) break;
            
            // Check price cross
            if (order->getPrice() < best_ask->getPrice()) {
                break;  // No match possible
            }
            
            // Match with front order at this level (time priority)
            auto ask_order = best_ask->getFrontOrder();
            if (!ask_order) break;
            
            // Calculate fill quantity
            uint64_t fill_qty = std::min(order->getRemainingQuantity(),
                                        ask_order->getRemainingQuantity());
            
            // Execute trade at ask price (passive order price)
            double fill_price = best_ask->getPrice();
            
            // Update orders
            order->fill(fill_qty);
            ask_order->fill(fill_qty);
            best_ask->updateQuantity(fill_qty);
            
            // Record fills for both sides
            fills.push_back({order->getId(), fill_price, fill_qty,
                           order->getTimestamp()});
            fills.push_back({ask_order->getId(), fill_price, fill_qty,
                           ask_order->getTimestamp()});
            
            // Remove fully filled ask order
            if (ask_order->getRemainingQuantity() == 0) {
                best_ask->popFront();
                if (best_ask->isEmpty()) {
                    book.removeOrder(ask_order->getId());
                }
            }
        }
    }
    
    void matchSellOrder(std::shared_ptr<Order> order, OrderBook& book,
                       std::vector<Fill>& fills) {
        // Match against bids (buy orders)
        while (order->getRemainingQuantity() > 0) {
            auto best_bid = book.getBestBid();
            if (!best_bid) break;
            
            // Check price cross
            if (order->getPrice() > best_bid->getPrice()) {
                break;  // No match possible
            }
            
            // Match with front order at this level (time priority)
            auto bid_order = best_bid->getFrontOrder();
            if (!bid_order) break;
            
            // Calculate fill quantity
            uint64_t fill_qty = std::min(order->getRemainingQuantity(),
                                        bid_order->getRemainingQuantity());
            
            // Execute trade at bid price (passive order price)
            double fill_price = best_bid->getPrice();
            
            // Update orders
            order->fill(fill_qty);
            bid_order->fill(fill_qty);
            best_bid->updateQuantity(fill_qty);
            
            // Record fills for both sides
            fills.push_back({order->getId(), fill_price, fill_qty,
                           order->getTimestamp()});
            fills.push_back({bid_order->getId(), fill_price, fill_qty,
                           bid_order->getTimestamp()});
            
            // Remove fully filled bid order
            if (bid_order->getRemainingQuantity() == 0) {
                best_bid->popFront();
                if (best_bid->isEmpty()) {
                    book.removeOrder(bid_order->getId());
                }
            }
        }
    }

private:
    std::unordered_map<std::string, std::shared_ptr<OrderBook>> books_;
};

} // namespace simulator
