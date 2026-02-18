#include "parsers/binary_parser.hpp"
#include <cstring>

namespace hft {

BinaryParser::BinaryParser() noexcept : partial_length_(0) {}

void BinaryParser::reset() noexcept {
    partial_length_ = 0;
}

Result BinaryParser::parse(const char* buffer, size_t length,
                           Event& out_event, size_t& consumed_bytes) noexcept {
    consumed_bytes = 0;
    
    // Combine partial buffer with new data if needed
    const char* parse_buffer = buffer;
    size_t parse_length = length;
    
    if (partial_length_ > 0) {
        size_t to_copy = std::min(length, PARTIAL_BUFFER_SIZE - partial_length_);
        memcpy(partial_buffer_ + partial_length_, buffer, to_copy);
        partial_length_ += to_copy;
        parse_buffer = partial_buffer_;
        parse_length = partial_length_;
    }
    
    // Need at least header
    if (parse_length < sizeof(Header)) {
        if (parse_buffer != partial_buffer_) {
            memcpy(partial_buffer_, parse_buffer, parse_length);
            partial_length_ = parse_length;
        }
        return Result::ERROR_INCOMPLETE_MESSAGE;
    }
    
    // Parse header - use memcpy to avoid alignment issues
    Header hdr;
    memcpy(&hdr, parse_buffer, sizeof(Header));
    
    // Convert endianness
    u16 payload_length = le16_to_host(hdr.payload_length);
    u64 sequence = le64_to_host(hdr.sequence);
    
    // Validate header
    if (!validate_header(&hdr)) {
        consumed_bytes = sizeof(Header);
        partial_length_ = 0;
        return Result::ERROR_INVALID_MESSAGE;
    }
    
    // Check if we have complete message
    size_t total_length = sizeof(Header) + payload_length;
    if (parse_length < total_length) {
        if (parse_buffer != partial_buffer_) {
            memcpy(partial_buffer_, parse_buffer, parse_length);
            partial_length_ = parse_length;
        }
        return Result::ERROR_INCOMPLETE_MESSAGE;
    }
    
    // Parse payload based on message type
    const char* payload = parse_buffer + sizeof(Header);
    Result result = Result::ERROR_INVALID_MESSAGE;
    
    switch (hdr.msg_type) {
        case 1: // OrderBookUpdate
            out_event.type = MsgType::ORDER_BOOK_UPDATE;
            result = parse_order_book_update(payload, payload_length, &hdr, out_event.order_book);
            break;
        case 2: // Trade
            out_event.type = MsgType::TRADE;
            result = parse_trade(payload, payload_length, &hdr, out_event.trade);
            break;
        default:
            result = Result::ERROR_INVALID_MESSAGE;
    }
    
    consumed_bytes = total_length;
    partial_length_ = 0;
    return result;
}

u16 BinaryParser::le16_to_host(u16 val) noexcept {
    // Portable little-endian to host conversion
    #if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        return val;
    #else
        return ((val & 0xFF) << 8) | ((val >> 8) & 0xFF);
    #endif
}

u32 BinaryParser::le32_to_host(u32 val) noexcept {
    #if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        return val;
    #else
        return ((val & 0xFF) << 24) | ((val & 0xFF00) << 8) |
               ((val >> 8) & 0xFF00) | ((val >> 24) & 0xFF);
    #endif
}

u64 BinaryParser::le64_to_host(u64 val) noexcept {
    #if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        return val;
    #else
        return ((val & 0xFFULL) << 56) | ((val & 0xFF00ULL) << 40) |
               ((val & 0xFF0000ULL) << 24) | ((val & 0xFF000000ULL) << 8) |
               ((val >> 8) & 0xFF000000ULL) | ((val >> 24) & 0xFF0000ULL) |
               ((val >> 40) & 0xFF00ULL) | ((val >> 56) & 0xFFULL);
    #endif
}

i64 BinaryParser::le64_to_host_signed(i64 val) noexcept {
    return (i64)le64_to_host((u64)val);
}

bool BinaryParser::validate_header(const Header* hdr) noexcept {
    // Basic validation
    if (hdr->msg_type == 0 || hdr->msg_type > 10) {
        return false;
    }
    
    u16 payload_length = le16_to_host(hdr->payload_length);
    if (payload_length > 4096) {
        return false;
    }
    
    return true;
}

Result BinaryParser::parse_order_book_update(const char* payload, size_t len,
                                             const Header* hdr, OrderBookUpdate& out) noexcept {
    if (len < sizeof(OrderBookPayload)) {
        return Result::ERROR_INVALID_MESSAGE;
    }
    
    // Use memcpy to avoid alignment issues
    OrderBookPayload data;
    memcpy(&data, payload, sizeof(OrderBookPayload));
    
    out.sequence = le64_to_host(hdr->sequence);
    out.price = le64_to_host_signed(data.price);
    out.quantity = le64_to_host_signed(data.quantity);
    out.side = static_cast<Side>(data.side);
    out.level = data.level;
    
    // Copy symbol (fixed size)
    memcpy(out.symbol, data.symbol, sizeof(out.symbol));
    out.symbol[sizeof(out.symbol) - 1] = '\0';
    
    return Result::OK;
}

Result BinaryParser::parse_trade(const char* payload, size_t len,
                                 const Header* hdr, Trade& out) noexcept {
    if (len < sizeof(TradePayload)) {
        return Result::ERROR_INVALID_MESSAGE;
    }
    
    // Use memcpy to avoid alignment issues
    TradePayload data;
    memcpy(&data, payload, sizeof(TradePayload));
    
    out.sequence = le64_to_host(hdr->sequence);
    out.price = le64_to_host_signed(data.price);
    out.quantity = le64_to_host_signed(data.quantity);
    out.aggressor_side = static_cast<Side>(data.side);
    
    // Copy symbol
    memcpy(out.symbol, data.symbol, sizeof(out.symbol));
    out.symbol[sizeof(out.symbol) - 1] = '\0';
    
    return Result::OK;
}

} // namespace hft
