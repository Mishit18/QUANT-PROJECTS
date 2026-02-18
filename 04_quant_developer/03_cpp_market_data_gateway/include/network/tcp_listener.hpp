#pragma once

#include "common/types.hpp"
#include "buffer/ring_buffer.hpp"

namespace hft {

// Non-blocking TCP listener using epoll
// Handles partial reads and message reassembly
class TCPListener {
public:
    explicit TCPListener(uint16_t port, RingBuffer& output_buffer);
    ~TCPListener();
    
    // Non-copyable
    TCPListener(const TCPListener&) = delete;
    TCPListener& operator=(const TCPListener&) = delete;
    
    // Initialize socket and epoll
    Result initialize() noexcept;
    
    // Poll for events and read data into ring buffer
    // Returns number of bytes read, or error
    Result poll_and_read(int timeout_ms) noexcept;
    
    // Get file descriptor for external epoll management
    int get_fd() const noexcept { return listen_fd_; }
    
private:
    uint16_t port_;
    int listen_fd_;
    int epoll_fd_;
    RingBuffer& output_buffer_;
    
    // Pre-allocated buffer for recv
    static constexpr size_t RECV_BUFFER_SIZE = 65536;
    char recv_buffer_[RECV_BUFFER_SIZE];
    
    Result accept_connection() noexcept;
    Result read_from_client(int client_fd) noexcept;
};

} // namespace hft
