#pragma once

#include "common/types.hpp"
#include "common/events.hpp"

namespace hft {

// Zero-copy FIX parser
// Parses FIX 4.2/4.4 tag-value format without allocation
// Handles partial messages across packet boundaries
class FIXParser {
public:
    FIXParser() noexcept;
    
    // Parse FIX message from buffer
    // Returns Result::OK if complete message parsed
    // Returns Result::ERROR_INCOMPLETE_MESSAGE if need more data
    // out_event is populated on success
    // consumed_bytes indicates how many bytes were processed
    Result parse(const char* buffer, size_t length, 
                 Event& out_event, size_t& consumed_bytes) noexcept;
    
    // Reset parser state (for new connection)
    void reset() noexcept;
    
private:
    // FIX message starts with "8=FIX.4.x|9=length|..."
    // SOH (0x01) is field delimiter
    static constexpr char SOH = 0x01;
    
    // Find complete FIX message in buffer
    // Returns pointer to start of message and length, or nullptr
    const char* find_complete_message(const char* buffer, size_t length, 
                                      size_t& out_msg_length) noexcept;
    
    // Extract tag value without allocation
    bool extract_tag(const char* msg, size_t msg_len, 
                     int tag, StringView& out_value) noexcept;
    
    // Validate checksum (tag 10)
    bool validate_checksum(const char* msg, size_t msg_len) noexcept;
    
    // Parse execution report (MsgType=8)
    Result parse_execution_report(const char* msg, size_t msg_len, 
                                  ExecutionReport& out) noexcept;
    
    // Convert string to integer (no std::stoi)
    i64 parse_int(const char* str, size_t len) noexcept;
    
    // Convert string to price (fixed-point)
    Price parse_price(const char* str, size_t len) noexcept;
    
    // Partial message buffer for reassembly
    static constexpr size_t PARTIAL_BUFFER_SIZE = 4096;
    char partial_buffer_[PARTIAL_BUFFER_SIZE];
    size_t partial_length_;
};

} // namespace hft
