#pragma once

#include "order.h"

namespace lob {

// FIFO queue at single price level using intrusive doubly-linked list
// All operations O(1) except iteration
class PriceLevel {
public:
    PriceLevel() noexcept 
        : price_(INVALID_PRICE), total_quantity_(0), 
          order_count_(0), head_(nullptr), tail_(nullptr) {}
    
    void initialize(Price price) noexcept {
        price_ = price;
        total_quantity_ = 0;
        order_count_ = 0;
        head_ = nullptr;
        tail_ = nullptr;
    }
    
    // Add order to back of queue (new passive order)
    void add_order(Order* order) noexcept {
        order->queue_position = order_count_;
        order->next = nullptr;
        order->prev = tail_;
        
        if (tail_) [[likely]] {
            tail_->next = order;
        } else {
            head_ = order;
        }
        tail_ = order;
        
        total_quantity_ += order->remaining_quantity();
        ++order_count_;
    }
    
    // Remove order from queue (cancel or full fill)
    void remove_order(Order* order) noexcept {
        if (order->prev) [[likely]] {
            order->prev->next = order->next;
        } else {
            head_ = order->next;
        }
        
        if (order->next) [[likely]] {
            order->next->prev = order->prev;
        } else {
            tail_ = order->prev;
        }
        
        total_quantity_ -= order->remaining_quantity();
        --order_count_;
        
        // Update queue positions for orders behind this one
        Order* curr = order->next;
        while (curr) {
            --curr->queue_position;
            curr = curr->next;
        }
    }
    
    // Update quantity after partial fill
    void update_quantity(Order* order, Quantity old_remaining) noexcept {
        total_quantity_ -= old_remaining;
        total_quantity_ += order->remaining_quantity();
    }
    
    [[nodiscard]] Order* front() const noexcept { return head_; }
    [[nodiscard]] Price price() const noexcept { return price_; }
    [[nodiscard]] Quantity total_quantity() const noexcept { return total_quantity_; }
    [[nodiscard]] uint32_t order_count() const noexcept { return order_count_; }
    [[nodiscard]] bool empty() const noexcept { return head_ == nullptr; }

private:
    Price price_;
    Quantity total_quantity_;
    uint32_t order_count_;
    Order* head_;
    Order* tail_;
};

} // namespace lob
