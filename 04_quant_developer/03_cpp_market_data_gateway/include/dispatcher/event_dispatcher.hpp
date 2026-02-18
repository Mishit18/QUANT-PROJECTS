#pragma once

#include "common/types.hpp"
#include "common/events.hpp"
#include "buffer/event_queue.hpp"
#include <atomic>
#include <thread>
#include <functional>

namespace hft {

// Event dispatcher: reads from event queue and publishes to consumers
// Uses function pointers to avoid std::function allocation on hot path
class EventDispatcher {
public:
    using ConsumerCallback = void (*)(const Event&, void* context);
    
    explicit EventDispatcher(EventQueue<65536>& input);
    ~EventDispatcher();
    
    // Register consumer callback with optional context pointer
    // NOTE: Callbacks are invoked on dispatcher thread
    // Keep callbacks fast to avoid backpressure
    void register_consumer(ConsumerCallback callback, void* context = nullptr);
    
    // Start dispatcher thread
    void start(int cpu_affinity = -1);
    
    // Stop dispatcher thread gracefully
    void stop();
    
    // Check if running
    bool is_running() const noexcept { return running_.load(std::memory_order_acquire); }
    
    // Statistics
    u64 get_events_dispatched() const noexcept { return events_dispatched_.load(std::memory_order_relaxed); }
    
private:
    void run();
    
    EventQueue<65536>& input_queue_;
    
    // Consumer callbacks (registered at startup, not modified during runtime)
    static constexpr size_t MAX_CONSUMERS = 8;
    struct Consumer {
        ConsumerCallback callback;
        void* context;
    };
    Consumer consumers_[MAX_CONSUMERS];
    size_t num_consumers_;
    
    std::thread thread_;
    std::atomic<bool> running_;
    
    // Statistics
    std::atomic<u64> events_dispatched_;
};

} // namespace hft
