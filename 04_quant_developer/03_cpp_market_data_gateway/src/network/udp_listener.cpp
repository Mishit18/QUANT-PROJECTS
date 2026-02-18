#include "network/udp_listener.hpp"
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>

namespace hft {

UDPListener::UDPListener(uint16_t port, RingBuffer& output_buffer)
    : port_(port), socket_fd_(-1), output_buffer_(output_buffer) {}

UDPListener::~UDPListener() {
    if (socket_fd_ >= 0) close(socket_fd_);
}

Result UDPListener::initialize() noexcept {
    // Create socket
    socket_fd_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (socket_fd_ < 0) {
        return Result::ERROR_SOCKET_ERROR;
    }
    
    // Set socket options
    int opt = 1;
    setsockopt(socket_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    setsockopt(socket_fd_, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt));
    
    // Increase receive buffer size
    int rcvbuf = 16 * 1024 * 1024; // 16MB
    setsockopt(socket_fd_, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf));
    
    // Set non-blocking
    int flags = fcntl(socket_fd_, F_GETFL, 0);
    fcntl(socket_fd_, F_SETFL, flags | O_NONBLOCK);
    
    // Bind
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port_);
    
    if (bind(socket_fd_, (sockaddr*)&addr, sizeof(addr)) < 0) {
        return Result::ERROR_SOCKET_ERROR;
    }
    
    return Result::OK;
}

Result UDPListener::join_multicast(const char* group_ip) noexcept {
    ip_mreq mreq{};
    mreq.imr_multiaddr.s_addr = inet_addr(group_ip);
    mreq.imr_interface.s_addr = INADDR_ANY;
    
    if (setsockopt(socket_fd_, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq)) < 0) {
        return Result::ERROR_SOCKET_ERROR;
    }
    
    return Result::OK;
}

Result UDPListener::read_datagrams() noexcept {
    while (true) {
        sockaddr_in src_addr{};
        socklen_t addr_len = sizeof(src_addr);
        
        ssize_t n = recvfrom(socket_fd_, recv_buffer_, RECV_BUFFER_SIZE, 0,
                            (sockaddr*)&src_addr, &addr_len);
        
        if (n > 0) {
            // Write to ring buffer
            char* write_ptr = output_buffer_.reserve_write(n);
            if (write_ptr) {
                memcpy(write_ptr, recv_buffer_, n);
                output_buffer_.commit_write(n);
            } else {
                // Buffer full, drop packet (UDP semantics)
                return Result::ERROR_BUFFER_FULL;
            }
        } else {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                // No more data
                break;
            } else {
                // Error
                return Result::ERROR_SOCKET_ERROR;
            }
        }
    }
    
    return Result::OK;
}

} // namespace hft
