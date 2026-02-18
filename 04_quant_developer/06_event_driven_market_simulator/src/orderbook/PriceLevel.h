#pragma once

#include "Order.h"
#include <deque>
#include <memory>

namespace simulator {

// FIFO queue at a single price level
// Maintains strict time priority
class PriceLevel {
public:
    PriceLevel(double price) : price_(price), total_quantity_(0) {}
    
    double getPrice() const { return price_; }
    uint64_t getTotalQuantity() const { return total_quantity_; }
    bool isEmpty() const { return orders_.empty(); }
    size_t size() const { return orders_.size(); }
    
    // Add order to back of queue (time priority)
    void addOrder(std::shared_ptr<Order> order) {
        orders_.push_back(order);
        total_quantity_ += order->getRemainingQuantity();
    }
    
    // Remove specific order (for cancels)
    bool removeOrder(uint64_t order_id) {
        for (auto it = orders_.begin(); it != orders_.end(); ++it) {
            if ((*it)->getId() == order_id) {
                total_quantity_ -= (*it)->getRemainingQuantity();
                orders_.erase(it);
                return true;
            }
        }
        return false;
    }
    
    // Get front order (best time priority)
    std::shared_ptr<Order> getFrontOrder() {
        if (orders_.empty()) {
            return nullptr;
        }
        return orders_.front();
    }
    
    // Remove front order after full fill
    void popFront() {
        if (!orders_.empty()) {
            total_quantity_ -= orders_.front()->getRemainingQuantity();
            orders_.pop_front();
        }
    }
    
    // Update total quantity after partial fill
    void updateQuantity(uint64_t delta) {
        total_quantity_ -= delta;
    }
    
    // Get all orders (for inspection)
    const std::deque<std::shared_ptr<Order>>& getOrders() const {
        return orders_;
    }

private:
    double price_;
    uint64_t total_quantity_;
    std::deque<std::shared_ptr<Order>> orders_;  // FIFO queue
};

} // namespace simulator
