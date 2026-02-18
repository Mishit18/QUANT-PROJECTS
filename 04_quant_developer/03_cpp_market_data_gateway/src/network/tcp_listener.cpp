#include "network/tcp_listener.hpp"
#include <sys/socket.h>
#include <sys/epoll.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>

namespace hft {

TCPListener::TCPListener(uint16_t port, RingBuffer& output_buffer)
    : port_(port), listen_fd_(-1), epoll_fd_(-1), output_buffer_(output_buffer) {}

TCPListener::~TCPListener() {
    if (epoll_fd_ >= 0) close(epoll_fd_);
    if (listen_fd_ >= 0) close(listen_fd_);
}

Result TCPListener::initialize() noexcept {
    // Create socket
    listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) {
        return Result::ERROR_SOCKET_ERROR;
    }
    
    // Set socket options
    int opt = 1;
    setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt));
    
    // Disable Nagle's algorithm for low latency
    setsockopt(listen_fd_, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));
    
    // Set non-blocking
    int flags = fcntl(listen_fd_, F_GETFL, 0);
    fcntl(listen_fd_, F_SETFL, flags | O_NONBLOCK);
    
    // Bind
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port_);
    
    if (bind(listen_fd_, (sockaddr*)&addr, sizeof(addr)) < 0) {
        return Result::ERROR_SOCKET_ERROR;
    }
    
    // Listen
    if (listen(listen_fd_, 128) < 0) {
        return Result::ERROR_SOCKET_ERROR;
    }
    
    // Create epoll
    epoll_fd_ = epoll_create1(0);
    if (epoll_fd_ < 0) {
        return Result::ERROR_SOCKET_ERROR;
    }
    
    // Add listen socket to epoll
    epoll_event ev{};
    ev.events = EPOLLIN | EPOLLET; // Edge-triggered
    ev.data.fd = listen_fd_;
    if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, listen_fd_, &ev) < 0) {
        return Result::ERROR_SOCKET_ERROR;
    }
    
    return Result::OK;
}

Result TCPListener::poll_and_read(int timeout_ms) noexcept {
    epoll_event events[32];
    int nfds = epoll_wait(epoll_fd_, events, 32, timeout_ms);
    
    if (nfds < 0) {
        return Result::ERROR_SOCKET_ERROR;
    }
    
    for (int i = 0; i < nfds; ++i) {
        if (events[i].data.fd == listen_fd_) {
            // New connection
            accept_connection();
        } else {
            // Data from client
            read_from_client(events[i].data.fd);
        }
    }
    
    return Result::OK;
}

Result TCPListener::accept_connection() noexcept {
    sockaddr_in client_addr{};
    socklen_t addr_len = sizeof(client_addr);
    
    int client_fd = accept(listen_fd_, (sockaddr*)&client_addr, &addr_len);
    if (client_fd < 0) {
        return Result::ERROR_SOCKET_ERROR;
    }
    
    // Set non-blocking
    int flags = fcntl(client_fd, F_GETFL, 0);
    fcntl(client_fd, F_SETFL, flags | O_NONBLOCK);
    
    // Disable Nagle
    int opt = 1;
    setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));
    
    // Add to epoll
    epoll_event ev{};
    ev.events = EPOLLIN | EPOLLET;
    ev.data.fd = client_fd;
    epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, client_fd, &ev);
    
    return Result::OK;
}

Result TCPListener::read_from_client(int client_fd) noexcept {
    while (true) {
        ssize_t n = recv(client_fd, recv_buffer_, RECV_BUFFER_SIZE, 0);
        
        if (n > 0) {
            // Write to ring buffer
            char* write_ptr = output_buffer_.reserve_write(n);
            if (write_ptr) {
                memcpy(write_ptr, recv_buffer_, n);
                output_buffer_.commit_write(n);
            } else {
                // Buffer full, backpressure
                return Result::ERROR_BUFFER_FULL;
            }
        } else if (n == 0) {
            // Connection closed
            epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, client_fd, nullptr);
            close(client_fd);
            break;
        } else {
            // Error
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                // No more data
                break;
            } else if (errno == EINTR) {
                // Interrupted, retry
                continue;
            } else {
                // Real error
                epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, client_fd, nullptr);
                close(client_fd);
                return Result::ERROR_SOCKET_ERROR;
            }
        }
    }
    
    return Result::OK;
}

} // namespace hft
