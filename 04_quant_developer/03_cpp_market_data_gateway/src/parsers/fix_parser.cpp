#include "parsers/fix_parser.hpp"
#include <cstring>

namespace hft {

FIXParser::FIXParser() noexcept : partial_length_(0) {}

void FIXParser::reset() noexcept {
    partial_length_ = 0;
}

Result FIXParser::parse(const char* buffer, size_t length,
                        Event& out_event, size_t& consumed_bytes) noexcept {
    consumed_bytes = 0;
    
    // Combine partial buffer with new data if needed
    const char* parse_buffer = buffer;
    size_t parse_length = length;
    
    if (partial_length_ > 0) {
        // Append to partial buffer
        size_t to_copy = std::min(length, PARTIAL_BUFFER_SIZE - partial_length_);
        memcpy(partial_buffer_ + partial_length_, buffer, to_copy);
        partial_length_ += to_copy;
        parse_buffer = partial_buffer_;
        parse_length = partial_length_;
    }
    
    // Find complete message
    size_t msg_length = 0;
    const char* msg = find_complete_message(parse_buffer, parse_length, msg_length);
    
    if (!msg) {
        // Incomplete message, save to partial buffer
        if (parse_buffer != partial_buffer_) {
            size_t to_save = std::min(parse_length, PARTIAL_BUFFER_SIZE);
            memcpy(partial_buffer_, parse_buffer, to_save);
            partial_length_ = to_save;
        }
        return Result::ERROR_INCOMPLETE_MESSAGE;
    }
    
    // Validate checksum
    if (!validate_checksum(msg, msg_length)) {
        consumed_bytes = msg_length;
        partial_length_ = 0;
        return Result::ERROR_CHECKSUM_FAILED;
    }
    
    // Extract MsgType (tag 35)
    StringView msg_type;
    if (!extract_tag(msg, msg_length, 35, msg_type)) {
        consumed_bytes = msg_length;
        partial_length_ = 0;
        return Result::ERROR_INVALID_MESSAGE;
    }
    
    // Parse based on message type
    Result result = Result::ERROR_INVALID_MESSAGE;
    
    if (msg_type.equals("8", 1)) {
        // Execution Report
        out_event.type = MsgType::EXECUTION_REPORT;
        result = parse_execution_report(msg, msg_length, out_event.execution);
    }
    // Add other message types as needed
    
    consumed_bytes = msg_length;
    partial_length_ = 0;
    return result;
}

const char* FIXParser::find_complete_message(const char* buffer, size_t length,
                                             size_t& out_msg_length) noexcept {
    // FIX message format: 8=FIX.4.x|9=length|...
    // Find start: "8=FIX"
    const char* start = nullptr;
    for (size_t i = 0; i + 5 < length; ++i) {
        if (buffer[i] == '8' && buffer[i+1] == '=' && 
            buffer[i+2] == 'F' && buffer[i+3] == 'I' && buffer[i+4] == 'X') {
            start = buffer + i;
            break;
        }
    }
    
    if (!start) return nullptr;
    
    // Extract body length (tag 9)
    StringView body_length_str;
    if (!extract_tag(start, length - (start - buffer), 9, body_length_str)) {
        return nullptr;
    }
    
    i64 body_length = parse_int(body_length_str.data, body_length_str.length);
    if (body_length <= 0 || body_length > 65536) {
        return nullptr;
    }
    
    // Calculate total message length
    // Find end of tag 9 value
    const char* body_start = body_length_str.data + body_length_str.length + 1; // +1 for SOH
    size_t total_length = (body_start - start) + body_length + 7; // +7 for checksum (10=XXX|)
    
    if (start + total_length > buffer + length) {
        return nullptr; // Incomplete
    }
    
    out_msg_length = total_length;
    return start;
}

bool FIXParser::extract_tag(const char* msg, size_t msg_len,
                            int tag, StringView& out_value) noexcept {
    // Convert tag to string
    char tag_str[16];
    int tag_len = 0;
    int t = tag;
    do {
        tag_str[tag_len++] = '0' + (t % 10);
        t /= 10;
    } while (t > 0);
    
    // Reverse
    for (int i = 0; i < tag_len / 2; ++i) {
        char tmp = tag_str[i];
        tag_str[i] = tag_str[tag_len - 1 - i];
        tag_str[tag_len - 1 - i] = tmp;
    }
    
    // Search for tag=value
    for (size_t i = 0; i + tag_len + 1 < msg_len; ++i) {
        bool match = true;
        for (int j = 0; j < tag_len; ++j) {
            if (msg[i + j] != tag_str[j]) {
                match = false;
                break;
            }
        }
        
        if (match && msg[i + tag_len] == '=') {
            // Found tag, extract value until SOH
            const char* value_start = msg + i + tag_len + 1;
            size_t value_len = 0;
            while (value_start + value_len < msg + msg_len && 
                   value_start[value_len] != SOH) {
                ++value_len;
            }
            
            out_value.data = value_start;
            out_value.length = value_len;
            return true;
        }
    }
    
    return false;
}

bool FIXParser::validate_checksum(const char* msg, size_t msg_len) noexcept {
    // Find checksum tag (10=) - must be at end of message
    // Search backwards for "10=" to avoid scanning entire message
    if (msg_len < 7) return false; // Minimum: "10=XXX|"
    
    // Find last occurrence of "10="
    const char* checksum_tag = nullptr;
    for (size_t i = msg_len - 7; i > 0; --i) {
        if (msg[i] == '1' && msg[i+1] == '0' && msg[i+2] == '=') {
            checksum_tag = msg + i;
            break;
        }
    }
    
    if (!checksum_tag) return false;
    
    // Extract checksum value (3 digits after "10=")
    const char* checksum_value = checksum_tag + 3;
    if (checksum_value + 3 > msg + msg_len) return false;
    
    // Parse checksum (3 digits)
    u32 expected = 0;
    for (int i = 0; i < 3; ++i) {
        if (checksum_value[i] < '0' || checksum_value[i] > '9') return false;
        expected = expected * 10 + (checksum_value[i] - '0');
    }
    
    // Calculate checksum (sum of bytes before "10=")
    u32 calculated = 0;
    for (const char* p = msg; p < checksum_tag; ++p) {
        calculated += (u8)*p;
    }
    calculated %= 256;
    
    return calculated == expected;
}

Result FIXParser::parse_execution_report(const char* msg, size_t msg_len,
                                        ExecutionReport& out) noexcept {
    // Extract required fields
    StringView order_id, symbol, side_str, price_str, qty_str, leaves_str, exec_type_str, seq_str;
    
    if (!extract_tag(msg, msg_len, 11, order_id)) return Result::ERROR_INVALID_MESSAGE;
    if (!extract_tag(msg, msg_len, 55, symbol)) return Result::ERROR_INVALID_MESSAGE;
    if (!extract_tag(msg, msg_len, 54, side_str)) return Result::ERROR_INVALID_MESSAGE;
    if (!extract_tag(msg, msg_len, 44, price_str)) return Result::ERROR_INVALID_MESSAGE;
    if (!extract_tag(msg, msg_len, 32, qty_str)) return Result::ERROR_INVALID_MESSAGE;
    if (!extract_tag(msg, msg_len, 151, leaves_str)) return Result::ERROR_INVALID_MESSAGE;
    if (!extract_tag(msg, msg_len, 150, exec_type_str)) return Result::ERROR_INVALID_MESSAGE;
    
    // Extract sequence number (tag 34) - optional but important for gap detection
    if (extract_tag(msg, msg_len, 34, seq_str)) {
        out.sequence = parse_int(seq_str.data, seq_str.length);
    } else {
        out.sequence = 0; // No sequence number
    }
    
    // Populate event
    size_t copy_len = std::min(order_id.length, sizeof(out.order_id) - 1);
    memcpy(out.order_id, order_id.data, copy_len);
    out.order_id[copy_len] = '\0';
    
    copy_len = std::min(symbol.length, sizeof(out.symbol) - 1);
    memcpy(out.symbol, symbol.data, copy_len);
    out.symbol[copy_len] = '\0';
    
    out.side = (side_str.data[0] == '1') ? Side::BUY : Side::SELL;
    out.price = parse_price(price_str.data, price_str.length);
    out.filled_quantity = parse_int(qty_str.data, qty_str.length);
    out.leaves_quantity = parse_int(leaves_str.data, leaves_str.length);
    out.exec_type = exec_type_str.data[0] - '0';
    
    return Result::OK;
}

i64 FIXParser::parse_int(const char* str, size_t len) noexcept {
    i64 result = 0;
    bool negative = false;
    size_t i = 0;
    
    if (len > 0 && str[0] == '-') {
        negative = true;
        i = 1;
    }
    
    for (; i < len; ++i) {
        if (str[i] >= '0' && str[i] <= '9') {
            result = result * 10 + (str[i] - '0');
        }
    }
    
    return negative ? -result : result;
}

Price FIXParser::parse_price(const char* str, size_t len) noexcept {
    // Parse fixed-point price (4 decimal places)
    // Example: "123.4567" -> 1234567
    i64 integer_part = 0;
    i64 fractional_part = 0;
    int fractional_digits = 0;
    bool in_fractional = false;
    bool negative = false;
    
    for (size_t i = 0; i < len; ++i) {
        if (str[i] == '-') {
            negative = true;
        } else if (str[i] == '.') {
            in_fractional = true;
        } else if (str[i] >= '0' && str[i] <= '9') {
            if (in_fractional) {
                if (fractional_digits < 4) {
                    fractional_part = fractional_part * 10 + (str[i] - '0');
                    ++fractional_digits;
                }
            } else {
                integer_part = integer_part * 10 + (str[i] - '0');
            }
        }
    }
    
    // Pad fractional part to 4 digits
    while (fractional_digits < 4) {
        fractional_part *= 10;
        ++fractional_digits;
    }
    
    Price result = integer_part * 10000 + fractional_part;
    return negative ? -result : result;
}

} // namespace hft
