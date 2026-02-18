#pragma once

#include <cstdint>
#include <string>
#include <memory>

namespace simulator {

enum class EventType {
    MARKET_DATA,
    ORDER_SUBMIT,
    ORDER_CANCEL,
    ORDER_AMEND,
    ORDER_FILL,
    ORDER_REJECT
};

// Base event class with logical timestamp
class Event {
public:
    Event(EventType type, uint64_t timestamp) 
        : type_(type), timestamp_(timestamp) {}
    
    virtual ~Event() = default;
    
    EventType getType() const { return type_; }
    uint64_t getTimestamp() const { return timestamp_; }
    
    // For deterministic ordering: timestamp first, then type
    bool operator<(const Event& other) const {
        if (timestamp_ != other.timestamp_) {
            return timestamp_ < other.timestamp_;
        }
        return static_cast<int>(type_) < static_cast<int>(other.type_);
    }

protected:
    EventType type_;
    uint64_t timestamp_;  // Nanoseconds since simulation start
};

// Market data event (e.g., external price update)
class MarketDataEvent : public Event {
public:
    MarketDataEvent(uint64_t timestamp, const std::string& symbol, 
                    double price, uint64_t volume)
        : Event(EventType::MARKET_DATA, timestamp),
          symbol_(symbol), price_(price), volume_(volume) {}
    
    const std::string& getSymbol() const { return symbol_; }
    double getPrice() const { return price_; }
    uint64_t getVolume() const { return volume_; }

private:
    std::string symbol_;
    double price_;
    uint64_t volume_;
};

// Order submission event
class OrderSubmitEvent : public Event {
public:
    OrderSubmitEvent(uint64_t timestamp, uint64_t order_id,
                     const std::string& symbol, bool is_buy,
                     double price, uint64_t quantity)
        : Event(EventType::ORDER_SUBMIT, timestamp),
          order_id_(order_id), symbol_(symbol), is_buy_(is_buy),
          price_(price), quantity_(quantity) {}
    
    uint64_t getOrderId() const { return order_id_; }
    const std::string& getSymbol() const { return symbol_; }
    bool isBuy() const { return is_buy_; }
    double getPrice() const { return price_; }
    uint64_t getQuantity() const { return quantity_; }

private:
    uint64_t order_id_;
    std::string symbol_;
    bool is_buy_;
    double price_;
    uint64_t quantity_;
};

// Order cancel event
class OrderCancelEvent : public Event {
public:
    OrderCancelEvent(uint64_t timestamp, uint64_t order_id)
        : Event(EventType::ORDER_CANCEL, timestamp),
          order_id_(order_id) {}
    
    uint64_t getOrderId() const { return order_id_; }

private:
    uint64_t order_id_;
};

// Order amend event (price and/or quantity change)
class OrderAmendEvent : public Event {
public:
    OrderAmendEvent(uint64_t timestamp, uint64_t order_id,
                    double new_price, uint64_t new_quantity)
        : Event(EventType::ORDER_AMEND, timestamp),
          order_id_(order_id), new_price_(new_price),
          new_quantity_(new_quantity) {}
    
    uint64_t getOrderId() const { return order_id_; }
    double getNewPrice() const { return new_price_; }
    uint64_t getNewQuantity() const { return new_quantity_; }

private:
    uint64_t order_id_;
    double new_price_;
    uint64_t new_quantity_;
};

// Order fill event (match notification)
class OrderFillEvent : public Event {
public:
    OrderFillEvent(uint64_t timestamp, uint64_t order_id,
                   double fill_price, uint64_t fill_quantity)
        : Event(EventType::ORDER_FILL, timestamp),
          order_id_(order_id), fill_price_(fill_price),
          fill_quantity_(fill_quantity) {}
    
    uint64_t getOrderId() const { return order_id_; }
    double getFillPrice() const { return fill_price_; }
    uint64_t getFillQuantity() const { return fill_quantity_; }

private:
    uint64_t order_id_;
    double fill_price_;
    uint64_t fill_quantity_;
};

} // namespace simulator
