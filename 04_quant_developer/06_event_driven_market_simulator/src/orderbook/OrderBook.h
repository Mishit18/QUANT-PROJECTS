#pragma once

#include "Order.h"
#include "PriceLevel.h"
#include <map>
#include <memory>
#include <unordered_map>

namespace simulator {

// Limit order book for a single symbol
// Maintains bid and ask sides with price-time priority
class OrderBook {
public:
    OrderBook(const std::string& symbol) : symbol_(symbol) {}
    
    const std::string& getSymbol() const { return symbol_; }
    
    // Add order to book
    void addOrder(std::shared_ptr<Order> order) {
        double price = order->getPrice();
        bool is_buy = order->isBuy();
        
        // Get or create price level
        auto& side = is_buy ? bids_ : asks_;
        auto it = side.find(price);
        
        if (it == side.end()) {
            auto level = std::make_shared<PriceLevel>(price);
            level->addOrder(order);
            side[price] = level;
        } else {
            it->second->addOrder(order);
        }
        
        // Track order location
        order_map_[order->getId()] = order;
    }
    
    // Remove order from book
    bool removeOrder(uint64_t order_id) {
        auto it = order_map_.find(order_id);
        if (it == order_map_.end()) {
            return false;
        }
        
        auto order = it->second;
        double price = order->getPrice();
        bool is_buy = order->isBuy();
        
        auto& side = is_buy ? bids_ : asks_;
        auto level_it = side.find(price);
        
        if (level_it != side.end()) {
            level_it->second->removeOrder(order_id);
            
            // Remove empty price level
            if (level_it->second->isEmpty()) {
                side.erase(level_it);
            }
        }
        
        order_map_.erase(it);
        return true;
    }
    
    // Get best bid price level
    std::shared_ptr<PriceLevel> getBestBid() {
        if (bids_.empty()) {
            return nullptr;
        }
        return bids_.rbegin()->second;  // Highest price
    }
    
    // Get best ask price level
    std::shared_ptr<PriceLevel> getBestAsk() {
        if (asks_.empty()) {
            return nullptr;
        }
        return asks_.begin()->second;  // Lowest price
    }
    
    // Get order by ID
    std::shared_ptr<Order> getOrder(uint64_t order_id) {
        auto it = order_map_.find(order_id);
        return (it != order_map_.end()) ? it->second : nullptr;
    }
    
    // Get bid/ask depth
    size_t getBidDepth() const { return bids_.size(); }
    size_t getAskDepth() const { return asks_.size(); }
    
    // Get all price levels (for inspection)
    const std::map<double, std::shared_ptr<PriceLevel>>& getBids() const {
        return bids_;
    }
    const std::map<double, std::shared_ptr<PriceLevel>>& getAsks() const {
        return asks_;
    }

private:
    std::string symbol_;
    
    // Price levels: bids descending, asks ascending
    std::map<double, std::shared_ptr<PriceLevel>> bids_;
    std::map<double, std::shared_ptr<PriceLevel>> asks_;
    
    // Fast order lookup
    std::unordered_map<uint64_t, std::shared_ptr<Order>> order_map_;
};

} // namespace simulator
