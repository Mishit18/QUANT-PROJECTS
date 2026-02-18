#pragma once

#include "Event.h"
#include <queue>
#include <vector>
#include <memory>
#include <functional>

namespace simulator {

// Priority queue for deterministic event ordering
// Events are ordered by timestamp, then by type for determinism
class EventQueue {
public:
    EventQueue() = default;
    
    void push(std::shared_ptr<Event> event) {
        queue_.push(event);
    }
    
    std::shared_ptr<Event> pop() {
        if (queue_.empty()) {
            return nullptr;
        }
        auto event = queue_.top();
        queue_.pop();
        return event;
    }
    
    bool empty() const {
        return queue_.empty();
    }
    
    size_t size() const {
        return queue_.size();
    }
    
    // Peek at next event without removing
    std::shared_ptr<Event> peek() const {
        if (queue_.empty()) {
            return nullptr;
        }
        return queue_.top();
    }

private:
    // Comparator for priority queue (min-heap by timestamp)
    struct EventComparator {
        bool operator()(const std::shared_ptr<Event>& a,
                       const std::shared_ptr<Event>& b) const {
            // Reverse comparison for min-heap
            return *b < *a;
        }
    };
    
    std::priority_queue<std::shared_ptr<Event>,
                       std::vector<std::shared_ptr<Event>>,
                       EventComparator> queue_;
};

} // namespace simulator
