# FIX Protocol Parser

## Overview

The FIX (Financial Information eXchange) parser implements zero-copy parsing of FIX 4.2/4.4 tag-value messages without heap allocation.

## FIX Protocol Basics

### Message Format

```
8=FIX.4.2|9=178|35=8|49=SENDER|56=TARGET|34=1|52=20240101-12:00:00|...10=123|
```

- **SOH delimiter**: 0x01 (shown as `|` above)
- **Tag=Value**: Each field is `tag=value` separated by SOH
- **Required tags**:
  - 8: BeginString (FIX version)
  - 9: BodyLength (bytes between tag 9 and tag 10)
  - 35: MsgType (message type)
  - 10: CheckSum (3-digit checksum)

### Message Types

- **8**: Execution Report
- **D**: New Order Single
- **G**: Order Cancel Request
- **0**: Heartbeat
- **A**: Logon

## Parser Design

### Zero-Copy Approach

**Problem**: Traditional parsers allocate strings for each tag value

**Solution**: Use `StringView` to point into buffer
```cpp
struct StringView {
    const char* data;
    size_t length;
};
```

**Benefit**: No allocation, no memcpy

### Partial Message Handling

**Problem**: TCP may deliver partial messages

**Solution**: Maintain partial buffer
```cpp
char partial_buffer_[4096];
size_t partial_length_;
```

**Algorithm**:
1. If previous partial data exists, prepend to new data
2. Search for complete message (check BodyLength)
3. If incomplete, save to partial buffer
4. If complete, parse and return

### Tag Extraction

**Naive approach**: Use std::map<int, std::string>
- Allocates for each tag
- Slow lookup

**Our approach**: Linear search with pointer traversal
```cpp
bool extract_tag(const char* msg, size_t msg_len, int tag, StringView& out);
```

**Why linear search?**
- FIX messages are small (~100-500 bytes)
- Linear search is cache-friendly
- No allocation overhead

**Optimization**: For hot tags (35, 55, 44), check first

### Checksum Validation

**Algorithm**:
```cpp
u32 checksum = 0;
for (const char* p = msg; p < checksum_tag; ++p) {
    checksum += (u8)*p;
}
checksum %= 256;
```

**Tradeoff**: Validation adds ~50ns
**Decision**: Enable in debug, disable in production (trust exchange)

## Parsing Execution Reports

### Required Tags

- **11**: ClOrdID (client order ID)
- **55**: Symbol
- **54**: Side (1=Buy, 2=Sell)
- **44**: Price
- **32**: LastQty (filled quantity)
- **151**: LeavesQty (remaining quantity)
- **150**: ExecType (0=New, 1=PartialFill, 2=Fill, 3=Canceled)

### Parsing Algorithm

```cpp
Result parse_execution_report(const char* msg, size_t msg_len, ExecutionReport& out) {
    // Extract required tags
    StringView order_id, symbol, side_str, price_str, qty_str, leaves_str, exec_type_str;
    
    if (!extract_tag(msg, msg_len, 11, order_id)) return ERROR;
    if (!extract_tag(msg, msg_len, 55, symbol)) return ERROR;
    // ... extract other tags
    
    // Populate event struct (fixed-size, no allocation)
    memcpy(out.order_id, order_id.data, min(order_id.length, 16));
    out.side = (side_str.data[0] == '1') ? Side::BUY : Side::SELL;
    out.price = parse_price(price_str.data, price_str.length);
    // ...
    
    return OK;
}
```

### Price Parsing

**Problem**: FIX prices are decimal strings ("150.25")

**Solution**: Fixed-point representation
```cpp
Price parse_price(const char* str, size_t len) {
    // Parse "150.25" -> 1502500 (4 decimal places)
    i64 integer_part = 0;
    i64 fractional_part = 0;
    // ... parse digits
    return integer_part * 10000 + fractional_part;
}
```

**Why fixed-point?**
- No floating-point rounding errors
- Fast integer arithmetic
- Deterministic

**Precision**: 4 decimal places (0.0001)

## Performance Characteristics

### Latency

- **Tag extraction**: 20-50ns per tag
- **Price parsing**: 30-50ns
- **Checksum validation**: 50-100ns
- **Total**: 200-500ns per message

### Throughput

- **Single core**: 2-5M messages/sec
- **Bottleneck**: Tag extraction (linear search)

### Optimization Opportunities

1. **SIMD tag search**: Use AVX2 to search for '=' and SOH
2. **Lookup table**: For common tags, use array instead of search
3. **Skip checksum**: Trust exchange, validate in debug only
4. **Batch parsing**: Parse multiple messages in one pass

## Error Handling

### No Exceptions

**Why?**
- Exceptions are slow (stack unwinding)
- Unpredictable latency
- Not allowed on hot path

**Alternative**: Return `Result` enum
```cpp
enum class Result {
    OK,
    ERROR_INVALID_MESSAGE,
    ERROR_CHECKSUM_FAILED,
    ERROR_INCOMPLETE_MESSAGE
};
```

### Partial Message Recovery

**Problem**: Partial message in buffer, how to recover?

**Solution**: Save to partial buffer, wait for more data
```cpp
if (result == ERROR_INCOMPLETE_MESSAGE) {
    memcpy(partial_buffer_, buffer, length);
    partial_length_ = length;
    return ERROR_INCOMPLETE_MESSAGE;
}
```

### Invalid Message Recovery

**Problem**: Corrupted message, how to resync?

**Solution**: Skip one byte, search for next "8=FIX"
```cpp
if (result == ERROR_INVALID_MESSAGE) {
    consumed_bytes = 1;  // Skip one byte
    return ERROR_INVALID_MESSAGE;
}
```

## Testing

### Unit Tests

- Parse valid execution report
- Handle partial messages
- Validate checksum
- Handle invalid messages

### Fuzz Testing

- Random byte sequences
- Truncated messages
- Invalid checksums
- Malformed tags

### Performance Testing

- Benchmark parsing latency
- Measure throughput
- Profile hot paths

## Production Considerations

### Sequence Gap Detection

**Problem**: Missing messages (sequence numbers have gaps)

**Solution**: Track expected sequence number
```cpp
if (msg_seq != expected_seq) {
    // Request retransmission
    request_resend(expected_seq, msg_seq - 1);
}
```

### Heartbeat Handling

**Problem**: No messages for extended period

**Solution**: Send heartbeat (MsgType=0)
```cpp
if (time_since_last_msg > 30s) {
    send_heartbeat();
}
```

### Session Management

**Problem**: Logon, logout, sequence reset

**Solution**: Implement session state machine
- DISCONNECTED → LOGON_SENT → LOGGED_IN → LOGGED_OUT

## Comparison with Other Parsers

### QuickFIX

- **Pros**: Full-featured, widely used
- **Cons**: Allocates for each message, slower

### Our Parser

- **Pros**: Zero-copy, no allocation, fast
- **Cons**: Limited message types, no session management

## Future Enhancements

1. **SIMD parsing**: AVX2 for tag search
2. **More message types**: Support all FIX 4.4 messages
3. **Session management**: Logon, logout, sequence reset
4. **Retransmission**: Handle sequence gaps
5. **FIX 5.0**: Support newer FIX versions
