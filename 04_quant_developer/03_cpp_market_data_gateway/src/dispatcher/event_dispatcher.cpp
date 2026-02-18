#include "dispatcher/event_dispatcher.hpp"
#include <pthread.h>
#include <sched.h>

namespace hft {

EventDispatcher::EventDispatcher(EventQueue<65536>& input)
    : input_queue_(input), num_consumers_(0), running_(false), events_dispatched_(0) {}

EventDispatcher::~EventDispatcher() {
    stop();
}

void EventDispatcher::register_consumer(ConsumerCallback callback, void* context) {
    if (num_consumers_ < MAX_CONSUMERS) {
        consumers_[num_consumers_].callback = callback;
        consumers_[num_consumers_].context = context;
        ++num_consumers_;
    }
}

void EventDispatcher::start(int cpu_affinity) {
    running_.store(true, std::memory_order_release);
    
    thread_ = std::thread([this, cpu_affinity]() {
        // Set CPU affinity
        if (cpu_affinity >= 0) {
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(cpu_affinity, &cpuset);
            pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
        }
        
        // Set thread priority
        sched_param param;
        param.sched_priority = sched_get_priority_max(SCHED_FIFO);
        pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
        
        run();
    });
}

void EventDispatcher::stop() {
    running_.store(false, std::memory_order_release);
    if (thread_.joinable()) {
        thread_.join();
    }
}

void EventDispatcher::run() {
    while (running_.load(std::memory_order_acquire)) {
        Event event;
        
        if (input_queue_.try_pop(event)) {
            // Dispatch to all consumers (no allocation, direct function call)
            for (size_t i = 0; i < num_consumers_; ++i) {
                consumers_[i].callback(event, consumers_[i].context);
            }
            
            events_dispatched_.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

} // namespace hft
