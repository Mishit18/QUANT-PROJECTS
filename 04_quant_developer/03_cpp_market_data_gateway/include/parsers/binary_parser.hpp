#pragma once

#include "common/types.hpp"
#include "common/events.hpp"

namespace hft {

// Binary market data protocol parser
// Wire format:
//   Header: [type:u8][length:u16][sequence:u64][reserved:u8][checksum:u32]
//   Payload: varies by message type
//
// OrderBookUpdate payload: [price:i64][quantity:i64][side:u8][level:u8][symbol:16bytes]
// Trade payload: [price:i64][quantity:i64][side:u8][symbol:16bytes]
//
// All multi-byte integers are little-endian
class BinaryParser {
public:
    BinaryParser() noexcept;
    
    // Parse binary message from buffer
    // Returns Result::OK if complete message parsed
    // Returns Result::ERROR_INCOMPLETE_MESSAGE if need more data
    // out_event is populated on success
    // consumed_bytes indicates how many bytes were processed
    Result parse(const char* buffer, size_t length,
                 Event& out_event, size_t& consumed_bytes) noexcept;
    
    // Reset parser state
    void reset() noexcept;
    
private:
    // Wire format header (16 bytes, aligned)
    struct CACHE_ALIGNED Header {
        u8 msg_type;
        u8 reserved1;
        u16 payload_length;
        u32 reserved2;
        u64 sequence;
    } __attribute__((packed));
    static_assert(sizeof(Header) == 16, "Header must be 16 bytes");
    
    // Payload structures (must match wire format exactly)
    struct OrderBookPayload {
        i64 price;
        i64 quantity;
        u8 side;
        u8 level;
        char symbol[16];
        u8 _padding[6];
    } __attribute__((packed));
    
    struct TradePayload {
        i64 price;
        i64 quantity;
        u8 side;
        char symbol[16];
        u8 _padding[7];
    } __attribute__((packed));
    
    // Endianness conversion (assume little-endian wire format)
    static u16 le16_to_host(u16 val) noexcept;
    static u32 le32_to_host(u32 val) noexcept;
    static u64 le64_to_host(u64 val) noexcept;
    static i64 le64_to_host_signed(i64 val) noexcept;
    
    // Validate header checksum (simple XOR for now)
    bool validate_header(const Header* hdr) noexcept;
    
    // Parse specific message types
    Result parse_order_book_update(const char* payload, size_t len,
                                   const Header* hdr, OrderBookUpdate& out) noexcept;
    Result parse_trade(const char* payload, size_t len,
                      const Header* hdr, Trade& out) noexcept;
    
    // Partial message buffer for reassembly
    static constexpr size_t PARTIAL_BUFFER_SIZE = 4096;
    char partial_buffer_[PARTIAL_BUFFER_SIZE];
    size_t partial_length_;
};

} // namespace hft
