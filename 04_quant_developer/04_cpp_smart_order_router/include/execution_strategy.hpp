#pragma once

#include "types.hpp"
#include "venue.hpp"
#include <vector>
#include <memory>

namespace execution {

// Base class for execution strategies
class ExecutionStrategy {
public:
    virtual ~ExecutionStrategy() = default;
    
    // Initialize strategy with parent order
    virtual void initialize(const ParentOrder& parent_order) = 0;
    
    // Generate child orders based on current market state
    virtual std::vector<std::shared_ptr<Order>> generate_orders(
        Timestamp current_time,
        const std::vector<Quote>& venue_quotes,
        Quantity remaining_quantity
    ) = 0;
    
    // Handle fill feedback
    virtual void on_fill(const Fill& fill, Timestamp current_time) = 0;
    
    // Check if strategy is complete
    virtual bool is_complete(Timestamp current_time) const = 0;
    
    // Get strategy type
    virtual StrategyType get_type() const = 0;
    
    // Get strategy name
    virtual std::string get_name() const = 0;

protected:
    ParentOrder parent_order_;
    OrderId next_child_order_id_;
    
    ExecutionStrategy() : next_child_order_id_(1) {}
    
    // Helper to create child order
    std::shared_ptr<Order> create_child_order(Side side, Quantity quantity, Price price, OrderType type) {
        auto order = std::make_shared<Order>();
        order->id = next_child_order_id_++;
        order->side = side;
        order->quantity = quantity;
        order->filled_quantity = 0;
        order->price = price;
        order->type = type;
        order->status = OrderStatus::PENDING;
        return order;
    }
};

} // namespace execution
