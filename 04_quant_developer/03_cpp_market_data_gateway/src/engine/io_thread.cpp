#include "engine/io_thread.hpp"
#include <pthread.h>
#include <sched.h>

namespace hft {

IOThread::IOThread(TCPListener& tcp, UDPListener& udp, RingBuffer& output)
    : tcp_listener_(tcp), udp_listener_(udp), output_buffer_(output),
      running_(false), bytes_received_(0), packets_received_(0) {}

IOThread::~IOThread() {
    stop();
}

void IOThread::start(int cpu_affinity) {
    running_.store(true, std::memory_order_release);
    
    thread_ = std::thread([this, cpu_affinity]() {
        // Set CPU affinity if specified
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

void IOThread::stop() {
    running_.store(false, std::memory_order_release);
    if (thread_.joinable()) {
        thread_.join();
    }
}

void IOThread::run() {
    while (running_.load(std::memory_order_acquire)) {
        // Read from UDP (non-blocking)
        Result udp_result = udp_listener_.read_datagrams();
        if (udp_result == Result::OK) {
            packets_received_.fetch_add(1, std::memory_order_relaxed);
        }
        
        // Read from TCP (with short timeout)
        Result tcp_result = tcp_listener_.poll_and_read(0);
        if (tcp_result == Result::OK) {
            packets_received_.fetch_add(1, std::memory_order_relaxed);
        }
        
        // Yield to avoid busy-wait
        // In production, use epoll to wait on both sockets
        // For now, tight loop for minimal latency
    }
}

} // namespace hft
