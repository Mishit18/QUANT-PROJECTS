#pragma once

#include "common/types.hpp"
#include "buffer/ring_buffer.hpp"

namespace hft {

// Non-blocking UDP listener
// Used for market data multicast feeds
class UDPListener {
public:
    explicit UDPListener(uint16_t port, RingBuffer& output_buffer);
    ~UDPListener();
    
    // Non-copyable
    UDPListener(const UDPListener&) = delete;
    UDPListener& operator=(const UDPListener&) = delete;
    
    // Initialize socket
    Result initialize() noexcept;
    
    // Join multicast group (optional)
    Result join_multicast(const char* group_ip) noexcept;
    
    // Read datagrams into ring buffer
    // Returns number of bytes read, or error
    Result read_datagrams() noexcept;
    
    // Get file descriptor for external epoll management
    int get_fd() const noexcept { return socket_fd_; }
    
private:
    uint16_t port_;
    int socket_fd_;
    RingBuffer& output_buffer_;
    
    // Pre-allocated buffer for recvfrom
    static constexpr size_t RECV_BUFFER_SIZE = 65536;
    char recv_buffer_[RECV_BUFFER_SIZE];
};

} // namespace hft
