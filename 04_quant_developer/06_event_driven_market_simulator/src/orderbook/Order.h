#pragma once

#include <cstdint>
#include <string>

namespace simulator {

enum class OrderState {
    PENDING,
    ACTIVE,
    PARTIALLY_FILLED,
    FILLED,
    CANCELLED,
    REJECTED
};

// Limit order representation
class Order {
public:
    Order(uint64_t id, const std::string& symbol, bool is_buy,
          double price, uint64_t quantity, uint64_t timestamp)
        : id_(id), symbol_(symbol), is_buy_(is_buy),
          price_(price), quantity_(quantity),
          remaining_quantity_(quantity),
          timestamp_(timestamp),
          state_(OrderState::ACTIVE) {}
    
    // Getters
    uint64_t getId() const { return id_; }
    const std::string& getSymbol() const { return symbol_; }
    bool isBuy() const { return is_buy_; }
    double getPrice() const { return price_; }
    uint64_t getQuantity() const { return quantity_; }
    uint64_t getRemainingQuantity() const { return remaining_quantity_; }
    uint64_t getTimestamp() const { return timestamp_; }
    OrderState getState() const { return state_; }
    
    // Modify order
    void fill(uint64_t quantity) {
        if (quantity > remaining_quantity_) {
            quantity = remaining_quantity_;
        }
        remaining_quantity_ -= quantity;
        
        if (remaining_quantity_ == 0) {
            state_ = OrderState::FILLED;
        } else {
            state_ = OrderState::PARTIALLY_FILLED;
        }
    }
    
    void cancel() {
        state_ = OrderState::CANCELLED;
    }
    
    void amend(double new_price, uint64_t new_quantity) {
        price_ = new_price;
        quantity_ = new_quantity;
        // Note: In real exchange, amend loses time priority
        // This is simplified; production would re-insert at back of queue
    }
    
    bool isActive() const {
        return state_ == OrderState::ACTIVE || 
               state_ == OrderState::PARTIALLY_FILLED;
    }

private:
    uint64_t id_;
    std::string symbol_;
    bool is_buy_;
    double price_;
    uint64_t quantity_;
    uint64_t remaining_quantity_;
    uint64_t timestamp_;
    OrderState state_;
};

// Fill result
struct Fill {
    uint64_t order_id;
    double price;
    uint64_t quantity;
    uint64_t timestamp;
};

} // namespace simulator
