#pragma once

#include "common/types.hpp"
#include <atomic>

namespace hft {

// Lock-free SPSC ring buffer
// Single producer, single consumer only
// No dynamic allocation after construction
class CACHE_ALIGNED RingBuffer {
public:
    explicit RingBuffer(size_t capacity);
    ~RingBuffer();
    
    // Non-copyable, non-movable
    RingBuffer(const RingBuffer&) = delete;
    RingBuffer& operator=(const RingBuffer&) = delete;
    
    // Producer API (called by IO thread)
    // Returns pointer to write location, or nullptr if full
    // Caller must call commit_write() after writing
    char* reserve_write(size_t size) noexcept;
    void commit_write(size_t size) noexcept;
    
    // Consumer API (called by decoder thread)
    // Returns pointer to read location and size, or nullptr if empty
    const char* peek_read(size_t& out_size) noexcept;
    void commit_read(size_t size) noexcept;
    
    // Statistics
    size_t capacity() const noexcept { return capacity_; }
    size_t available_write() const noexcept;
    size_t available_read() const noexcept;
    
private:
    char* buffer_;
    size_t capacity_;
    
    // Cache-line separated to avoid false sharing
    CACHE_ALIGNED std::atomic<size_t> write_pos_;
    CACHE_ALIGNED std::atomic<size_t> read_pos_;
};

} // namespace hft
