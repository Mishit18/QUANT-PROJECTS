#pragma once

#include "common/types.hpp"
#include "network/tcp_listener.hpp"
#include "network/udp_listener.hpp"
#include "buffer/ring_buffer.hpp"
#include <atomic>
#include <thread>

namespace hft {

// IO thread: reads from sockets and writes to ring buffer
// Pinned to dedicated CPU core for deterministic latency
class IOThread {
public:
    IOThread(TCPListener& tcp, UDPListener& udp, RingBuffer& output);
    ~IOThread();
    
    // Start IO thread
    void start(int cpu_affinity = -1);
    
    // Stop IO thread gracefully
    void stop();
    
    // Check if running
    bool is_running() const noexcept { return running_.load(std::memory_order_acquire); }
    
    // Statistics
    u64 get_bytes_received() const noexcept { return bytes_received_.load(std::memory_order_relaxed); }
    u64 get_packets_received() const noexcept { return packets_received_.load(std::memory_order_relaxed); }
    
private:
    void run();
    
    TCPListener& tcp_listener_;
    UDPListener& udp_listener_;
    RingBuffer& output_buffer_;
    
    std::thread thread_;
    std::atomic<bool> running_;
    
    // Statistics (relaxed ordering, not critical)
    std::atomic<u64> bytes_received_;
    std::atomic<u64> packets_received_;
};

} // namespace hft
