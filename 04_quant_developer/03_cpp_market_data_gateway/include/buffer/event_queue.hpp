#pragma once

#include "common/types.hpp"
#include "common/events.hpp"
#include <atomic>

namespace hft {

// Lock-free SPSC queue for events
// Fixed capacity, no dynamic allocation
template<size_t Capacity>
class CACHE_ALIGNED EventQueue {
public:
    EventQueue() noexcept : write_idx_(0), read_idx_(0) {}
    
    // Producer: try to push event
    bool try_push(const Event& event) noexcept {
        const size_t current_write = write_idx_.load(std::memory_order_relaxed);
        const size_t next_write = (current_write + 1) % Capacity;
        
        if (next_write == read_idx_.load(std::memory_order_acquire)) {
            return false; // Queue full
        }
        
        buffer_[current_write] = event;
        write_idx_.store(next_write, std::memory_order_release);
        return true;
    }
    
    // Consumer: try to pop event
    bool try_pop(Event& event) noexcept {
        const size_t current_read = read_idx_.load(std::memory_order_relaxed);
        
        if (current_read == write_idx_.load(std::memory_order_acquire)) {
            return false; // Queue empty
        }
        
        event = buffer_[current_read];
        const size_t next_read = (current_read + 1) % Capacity;
        read_idx_.store(next_read, std::memory_order_release);
        return true;
    }
    
    size_t size() const noexcept {
        const size_t w = write_idx_.load(std::memory_order_acquire);
        const size_t r = read_idx_.load(std::memory_order_acquire);
        return (w >= r) ? (w - r) : (Capacity - r + w);
    }
    
    bool empty() const noexcept {
        return read_idx_.load(std::memory_order_acquire) == 
               write_idx_.load(std::memory_order_acquire);
    }
    
private:
    Event buffer_[Capacity];
    CACHE_ALIGNED std::atomic<size_t> write_idx_;
    CACHE_ALIGNED std::atomic<size_t> read_idx_;
};

} // namespace hft
