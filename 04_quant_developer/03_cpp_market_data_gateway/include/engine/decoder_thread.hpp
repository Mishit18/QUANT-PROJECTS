#pragma once

#include "common/types.hpp"
#include "buffer/ring_buffer.hpp"
#include "buffer/event_queue.hpp"
#include "parsers/fix_parser.hpp"
#include "parsers/binary_parser.hpp"
#include <atomic>
#include <thread>

namespace hft {

// Decoder thread: reads from ring buffer, parses messages, writes to event queue
// Pinned to dedicated CPU core
class DecoderThread {
public:
    DecoderThread(RingBuffer& input, EventQueue<65536>& output);
    ~DecoderThread();
    
    // Start decoder thread
    void start(int cpu_affinity = -1);
    
    // Stop decoder thread gracefully
    void stop();
    
    // Check if running
    bool is_running() const noexcept { return running_.load(std::memory_order_acquire); }
    
    // Statistics
    u64 get_messages_parsed() const noexcept { return messages_parsed_.load(std::memory_order_relaxed); }
    u64 get_parse_errors() const noexcept { return parse_errors_.load(std::memory_order_relaxed); }
    
private:
    void run();
    
    // Detect protocol (FIX vs binary) from first byte
    enum class Protocol { FIX, BINARY, UNKNOWN };
    Protocol detect_protocol(const char* data, size_t length) noexcept;
    
    RingBuffer& input_buffer_;
    EventQueue<65536>& output_queue_;
    
    FIXParser fix_parser_;
    BinaryParser binary_parser_;
    
    std::thread thread_;
    std::atomic<bool> running_;
    
    // Statistics
    std::atomic<u64> messages_parsed_;
    std::atomic<u64> parse_errors_;
};

} // namespace hft
