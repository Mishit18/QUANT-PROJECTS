#include "matching_engine.h"

namespace lob {

MatchingEngine::MatchingEngine() noexcept
    : total_orders_(0), total_matches_(0), total_cancels_(0) {
}

Order* MatchingEngine::submit_order(OrderId order_id, Side side, Price price, Quantity quantity) noexcept {
    Order* order = order_pool_.allocate();
    if (!order) [[unlikely]] {
        return nullptr;
    }
    
    order->order_id = order_id;
    order->price = price;
    order->quantity = quantity;
    order->filled_quantity = 0;
    order->side = side;
    order->status = OrderStatus::NEW;
    order->timestamp = get_timestamp_ns();
    
    order_map_[order_id] = order;
    ++total_orders_;
    
    match_order(order);
    
    if (!order->is_filled()) {
        book_.add_order(order);
    } else {
        order_map_.erase(order_id);
        order_pool_.deallocate(order);
        return nullptr;
    }
    
    return order;
}

bool MatchingEngine::cancel_order(OrderId order_id) noexcept {
    auto it = order_map_.find(order_id);
    if (it == order_map_.end()) {
        return false;
    }
    
    Order* order = it->second;
    book_.remove_order(order);
    order->status = OrderStatus::CANCELLED;
    
    order_map_.erase(it);
    order_pool_.deallocate(order);
    ++total_cancels_;
    
    return true;
}

bool MatchingEngine::modify_order(OrderId order_id, Quantity new_quantity) noexcept {
    auto it = order_map_.find(order_id);
    if (it == order_map_.end()) {
        return false;
    }
    
    Order* order = it->second;
    
    if (new_quantity <= order->quantity) {
        return false;
    }
    
    Quantity old_remaining = order->remaining_quantity();
    order->quantity = new_quantity;
    book_.update_order(order, old_remaining);
    
    return true;
}

void MatchingEngine::match_order(Order* order) noexcept {
    if (order->is_buy()) {
        while (!order->is_filled() && book_.can_match_buy(order->price)) {
            PriceLevel* level = book_.best_ask();
            if (!level || level->empty()) [[unlikely]] break;
            
            Order* passive = level->front();
            Quantity match_qty = std::min(order->remaining_quantity(), 
                                         passive->remaining_quantity());
            
            execute_match(order, passive, match_qty);
        }
    } else {
        while (!order->is_filled() && book_.can_match_sell(order->price)) {
            PriceLevel* level = book_.best_bid();
            if (!level || level->empty()) [[unlikely]] break;
            
            Order* passive = level->front();
            Quantity match_qty = std::min(order->remaining_quantity(), 
                                         passive->remaining_quantity());
            
            execute_match(order, passive, match_qty);
        }
    }
}

void MatchingEngine::execute_match(Order* aggressive, Order* passive, Quantity match_qty) noexcept {
    Quantity passive_old_remaining = passive->remaining_quantity();
    
    aggressive->filled_quantity += match_qty;
    passive->filled_quantity += match_qty;
    
    if (aggressive->filled_quantity > 0) {
        aggressive->status = aggressive->is_filled() ? 
            OrderStatus::FULL_FILL : OrderStatus::PARTIAL_FILL;
    }
    
    if (passive->filled_quantity > 0) {
        passive->status = passive->is_filled() ? 
            OrderStatus::FULL_FILL : OrderStatus::PARTIAL_FILL;
    }
    
    book_.update_order(passive, passive_old_remaining);
    
    if (passive->is_filled()) {
        order_map_.erase(passive->order_id);
        order_pool_.deallocate(passive);
    }
    
    ++total_matches_;
}

} // namespace lob
