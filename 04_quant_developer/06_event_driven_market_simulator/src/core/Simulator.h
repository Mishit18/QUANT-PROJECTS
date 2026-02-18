#pragma once

#include "Event.h"
#include "EventQueue.h"
#include "Clock.h"
#include "../orderbook/MatchingEngine.h"
#include "../latency/LatencyModel.h"
#include <memory>
#include <functional>
#include <unordered_map>

namespace simulator {

// Main event-driven simulator
class Simulator {
public:
    Simulator(std::shared_ptr<LatencyModel> latency_model)
        : latency_model_(latency_model),
          clock_(),
          event_queue_(),
          matching_engine_(),
          running_(false) {}
    
    // Add event to queue
    void scheduleEvent(std::shared_ptr<Event> event) {
        event_queue_.push(event);
    }
    
    // Run simulation until queue is empty
    void run() {
        running_ = true;
        
        while (running_ && !event_queue_.empty()) {
            auto event = event_queue_.pop();
            if (!event) break;
            
            // Advance logical clock to event time
            clock_.advanceTo(event->getTimestamp());
            
            // Process event
            processEvent(event);
        }
        
        running_ = false;
    }
    
    // Stop simulation
    void stop() {
        running_ = false;
    }
    
    // Get current simulation time
    uint64_t getCurrentTime() const {
        return clock_.now();
    }
    
    // Get matching engine for inspection
    MatchingEngine& getMatchingEngine() {
        return matching_engine_;
    }
    
    const MatchingEngine& getMatchingEngine() const {
        return matching_engine_;
    }
    
    // Register callback for fill events
    void onFill(std::function<void(const OrderFillEvent&)> callback) {
        fill_callback_ = callback;
    }

private:
    void processEvent(std::shared_ptr<Event> event) {
        switch (event->getType()) {
            case EventType::ORDER_SUBMIT:
                handleOrderSubmit(
                    std::static_pointer_cast<OrderSubmitEvent>(event));
                break;
            
            case EventType::ORDER_CANCEL:
                handleOrderCancel(
                    std::static_pointer_cast<OrderCancelEvent>(event));
                break;
            
            case EventType::ORDER_AMEND:
                handleOrderAmend(
                    std::static_pointer_cast<OrderAmendEvent>(event));
                break;
            
            case EventType::MARKET_DATA:
                // Market data events can be logged or used for reference
                break;
            
            default:
                break;
        }
    }
    
    void handleOrderSubmit(std::shared_ptr<OrderSubmitEvent> event) {
        // Apply latency model
        uint64_t latency_ns = latency_model_->getLatency();
        uint64_t execution_time = event->getTimestamp() + latency_ns;
        
        // Create order
        Order order(event->getOrderId(), event->getSymbol(),
                   event->isBuy(), event->getPrice(),
                   event->getQuantity(), execution_time);
        
        // Submit to matching engine
        auto fills = matching_engine_.submitOrder(order);
        
        // Generate fill events
        for (const auto& fill : fills) {
            auto fill_event = std::make_shared<OrderFillEvent>(
                execution_time, fill.order_id, fill.price, fill.quantity);
            
            if (fill_callback_) {
                fill_callback_(*fill_event);
            }
        }
    }
    
    void handleOrderCancel(std::shared_ptr<OrderCancelEvent> event) {
        uint64_t latency_ns = latency_model_->getLatency();
        uint64_t execution_time = event->getTimestamp() + latency_ns;
        
        matching_engine_.cancelOrder(event->getOrderId());
    }
    
    void handleOrderAmend(std::shared_ptr<OrderAmendEvent> event) {
        uint64_t latency_ns = latency_model_->getLatency();
        uint64_t execution_time = event->getTimestamp() + latency_ns;
        
        matching_engine_.amendOrder(event->getOrderId(),
                                   event->getNewPrice(),
                                   event->getNewQuantity());
    }

private:
    std::shared_ptr<LatencyModel> latency_model_;
    SimulationClock clock_;
    EventQueue event_queue_;
    MatchingEngine matching_engine_;
    bool running_;
    std::function<void(const OrderFillEvent&)> fill_callback_;
};

} // namespace simulator
