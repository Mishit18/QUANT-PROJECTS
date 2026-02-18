#include "buffer/ring_buffer.hpp"
#include <cstdlib>
#include <cstring>

namespace hft {

RingBuffer::RingBuffer(size_t capacity) 
    : capacity_(capacity), write_pos_(0), read_pos_(0) {
    // Allocate aligned buffer
    if (posix_memalign((void**)&buffer_, CACHE_LINE_SIZE, capacity_) != 0) {
        buffer_ = nullptr;
    }
}

RingBuffer::~RingBuffer() {
    if (buffer_) {
        free(buffer_);
    }
}

char* RingBuffer::reserve_write(size_t size) noexcept {
    const size_t w = write_pos_.load(std::memory_order_relaxed);
    const size_t r = read_pos_.load(std::memory_order_acquire);
    
    // Calculate available space (always leave 1 byte gap to distinguish full from empty)
    size_t available;
    if (w >= r) {
        available = capacity_ - w + r - 1;
    } else {
        available = r - w - 1;
    }
    
    if (size > available) {
        return nullptr; // Not enough space
    }
    
    // Handle wrap-around: if write would exceed capacity, wrap to beginning
    // This requires checking if there's space at the beginning
    if (w + size > capacity_) {
        // Would wrap - check if there's space at beginning
        if (size > r - 1) {
            return nullptr; // Not enough space at beginning
        }
        // Mark end of buffer as unusable by advancing write_pos to 0
        // Reader will see this and wrap around
        write_pos_.store(0, std::memory_order_release);
        return buffer_; // Write at beginning
    }
    
    return buffer_ + w;
}

void RingBuffer::commit_write(size_t size) noexcept {
    const size_t w = write_pos_.load(std::memory_order_relaxed);
    size_t new_w = w + size;
    
    // Note: wrap-around is handled in reserve_write by setting write_pos to 0
    // So new_w should never exceed capacity here
    if (new_w >= capacity_) {
        new_w = size; // We wrapped, so new position is just the size written
    }
    
    write_pos_.store(new_w, std::memory_order_release);
}

const char* RingBuffer::peek_read(size_t& out_size) noexcept {
    const size_t r = read_pos_.load(std::memory_order_relaxed);
    const size_t w = write_pos_.load(std::memory_order_acquire);
    
    if (r == w) {
        out_size = 0;
        return nullptr; // Empty
    }
    
    // Calculate available data
    if (w > r) {
        out_size = w - r;
    } else {
        // Wrapped: read until end of buffer
        out_size = capacity_ - r;
    }
    
    return buffer_ + r;
}

void RingBuffer::commit_read(size_t size) noexcept {
    const size_t r = read_pos_.load(std::memory_order_relaxed);
    size_t new_r = r + size;
    
    // Handle wrap-around
    if (new_r >= capacity_) {
        new_r = 0; // Wrap to beginning
    }
    
    read_pos_.store(new_r, std::memory_order_release);
}

size_t RingBuffer::available_write() const noexcept {
    const size_t w = write_pos_.load(std::memory_order_acquire);
    const size_t r = read_pos_.load(std::memory_order_acquire);
    
    if (w >= r) {
        return capacity_ - w + r - 1;
    } else {
        return r - w - 1;
    }
}

size_t RingBuffer::available_read() const noexcept {
    const size_t w = write_pos_.load(std::memory_order_acquire);
    const size_t r = read_pos_.load(std::memory_order_acquire);
    
    if (w >= r) {
        return w - r;
    } else {
        return capacity_ - r + w;
    }
}

} // namespace hft
