# Binary Protocol Parser

## Overview

The binary parser implements zero-copy parsing of a custom binary market data protocol designed for minimal latency.

## Protocol Design

### Why Binary?

**Advantages over FIX**:
- **Smaller messages**: 40-60 bytes vs 150-300 bytes
- **Faster parsing**: No string parsing, just struct overlay
- **Deterministic size**: Fixed-size fields
- **Lower bandwidth**: 3-5x less data

**Disadvantages**:
- **Not human-readable**: Harder to debug
- **Endianness issues**: Must handle byte order
- **Versioning**: Harder to extend

### Wire Format

#### Header (16 bytes)

```cpp
struct Header {
    u8 msg_type;        // Message type (1=OrderBook, 2=Trade, etc.)
    u8 reserved1;       // Reserved for future use
    u16 payload_length; // Length of payload in bytes
    u32 reserved2;      // Reserved for future use
    u64 sequence;       // Sequence number
} __attribute__((packed));
```

**Why 16 bytes?**
- Aligns to cache line boundary (multiple of 16)
- Room for future fields (reserved)
- Fast to parse (single cache line)

#### OrderBookUpdate Payload (40 bytes)

```cpp
struct OrderBookPayload {
    i64 price;          // Fixed-point price (4 decimals)
    i64 quantity;       // Quantity
    u8 side;            // 0=Buy, 1=Sell
    u8 level;           // 0=Top of book, 1=Level 2, etc.
    char symbol[16];    // Symbol (null-terminated)
    u8 _padding[6];     // Pad to 40 bytes
} __attribute__((packed));
```

**Total**: 16 + 40 = 56 bytes

#### Trade Payload (32 bytes)

```cpp
struct TradePayload {
    i64 price;          // Fixed-point price
    i64 quantity;       // Quantity
    u8 side;            // Aggressor side
    char symbol[16];    // Symbol
    u8 _padding[7];     // Pad to 32 bytes
} __attribute__((packed));
```

**Total**: 16 + 32 = 48 bytes

## Parsing Strategy

### Struct Overlay (Zero-Copy)

**Technique**: Cast buffer pointer to struct pointer
```cpp
const Header* hdr = reinterpret_cast<const Header*>(buffer);
```

**Benefit**: No memcpy, no allocation

**Caveat**: Must handle alignment and endianness

### Alignment Considerations

**Problem**: Unaligned access is slow (or crashes on some CPUs)

**Solution**: Use `__attribute__((packed))` and validate alignment
```cpp
static_assert(alignof(Header) == 1, "Header must be packed");
```

**Alternative**: Use memcpy for unaligned fields
```cpp
u64 sequence;
memcpy(&sequence, buffer + 8, sizeof(u64));
```

### Endianness Handling

**Problem**: Network byte order (big-endian) vs host byte order

**Solution**: Explicit conversion
```cpp
u16 payload_length = le16toh(hdr->payload_length);
u64 sequence = le64toh(hdr->sequence);
```

**Why little-endian on wire?**
- Most CPUs are little-endian (x86, ARM)
- Avoids conversion on most platforms

**Tradeoff**: Big-endian CPUs (SPARC, PowerPC) pay conversion cost

## Parsing Algorithm

### Step 1: Check for Complete Message

```cpp
if (length < sizeof(Header)) {
    return ERROR_INCOMPLETE_MESSAGE;
}

const Header* hdr = reinterpret_cast<const Header*>(buffer);
u16 payload_length = le16toh(hdr->payload_length);

if (length < sizeof(Header) + payload_length) {
    return ERROR_INCOMPLETE_MESSAGE;
}
```

### Step 2: Validate Header

```cpp
bool validate_header(const Header* hdr) {
    if (hdr->msg_type == 0 || hdr->msg_type > 10) {
        return false;  // Invalid message type
    }
    
    u16 payload_length = le16toh(hdr->payload_length);
    if (payload_length > 4096) {
        return false;  // Payload too large
    }
    
    return true;
}
```

### Step 3: Parse Payload

```cpp
const char* payload = buffer + sizeof(Header);

switch (hdr->msg_type) {
    case 1:  // OrderBookUpdate
        return parse_order_book_update(payload, payload_length, hdr, out_event);
    case 2:  // Trade
        return parse_trade(payload, payload_length, hdr, out_event);
    default:
        return ERROR_INVALID_MESSAGE;
}
```

### Step 4: Convert Endianness

```cpp
Result parse_order_book_update(const char* payload, size_t len,
                               const Header* hdr, OrderBookUpdate& out) {
    const OrderBookPayload* data = reinterpret_cast<const OrderBookPayload*>(payload);
    
    out.sequence = le64toh(hdr->sequence);
    out.price = le64toh_signed(data->price);
    out.quantity = le64toh_signed(data->quantity);
    out.side = static_cast<Side>(data->side);
    out.level = data->level;
    
    memcpy(out.symbol, data->symbol, sizeof(out.symbol));
    
    return OK;
}
```

## Performance Characteristics

### Latency

- **Header validation**: 10-20ns
- **Endianness conversion**: 5-10ns per field
- **Struct overlay**: 0ns (pointer cast)
- **Total**: 50-100ns per message

**3-5x faster than FIX parsing**

### Throughput

- **Single core**: 10-20M messages/sec
- **Bottleneck**: Endianness conversion

### Optimization Opportunities

1. **Native endianness**: Use host byte order on wire (if all clients are x86)
2. **SIMD conversion**: Use AVX2 for batch endianness conversion
3. **Skip validation**: Trust sender, validate in debug only

## Error Handling

### Incomplete Messages

**Problem**: TCP delivers partial message

**Solution**: Save to partial buffer
```cpp
if (result == ERROR_INCOMPLETE_MESSAGE) {
    memcpy(partial_buffer_, buffer, length);
    partial_length_ = length;
    return ERROR_INCOMPLETE_MESSAGE;
}
```

### Invalid Messages

**Problem**: Corrupted header or payload

**Solution**: Skip message, log error
```cpp
if (!validate_header(hdr)) {
    consumed_bytes = sizeof(Header);
    return ERROR_INVALID_MESSAGE;
}
```

### Sequence Gaps

**Problem**: Missing messages (sequence number gap)

**Solution**: Request retransmission
```cpp
if (sequence != expected_sequence) {
    request_retransmission(expected_sequence, sequence - 1);
}
```

## Protocol Versioning

### Problem

How to extend protocol without breaking existing clients?

### Solution 1: Reserved Fields

Use reserved fields in header for new features
```cpp
struct Header {
    u8 msg_type;
    u8 version;        // Protocol version (was reserved1)
    u16 payload_length;
    u32 flags;         // Feature flags (was reserved2)
    u64 sequence;
};
```

### Solution 2: New Message Types

Add new message types for new features
- Type 1-10: Version 1
- Type 11-20: Version 2

### Solution 3: Length-Prefixed Fields

Use TLV (Type-Length-Value) encoding for extensibility
```cpp
struct TLV {
    u16 type;
    u16 length;
    char value[length];
};
```

**Tradeoff**: More complex parsing, slower

## Comparison with Other Protocols

### FIX

- **Pros**: Human-readable, widely supported
- **Cons**: Slower parsing, larger messages

### Protobuf

- **Pros**: Schema evolution, compression
- **Cons**: Slower parsing, requires code generation

### FlatBuffers

- **Pros**: Zero-copy, fast
- **Cons**: Complex schema, larger messages

### Our Binary Protocol

- **Pros**: Minimal latency, simple
- **Cons**: Not extensible, not human-readable

## Testing

### Unit Tests

- Parse valid messages
- Handle partial messages
- Validate endianness conversion
- Handle invalid messages

### Fuzz Testing

- Random byte sequences
- Truncated messages
- Invalid headers
- Malformed payloads

### Interoperability Testing

- Test on big-endian systems (SPARC, PowerPC)
- Test on ARM (different alignment rules)

## Production Considerations

### Checksum

**Problem**: Detect corrupted messages

**Solution**: Add checksum to header
```cpp
struct Header {
    u8 msg_type;
    u8 reserved1;
    u16 payload_length;
    u32 checksum;      // CRC32 of payload
    u64 sequence;
};
```

**Tradeoff**: Adds 20-50ns latency

### Compression

**Problem**: Reduce bandwidth

**Solution**: Use LZ4 or Snappy for compression
- LZ4: 500 MB/s compression, 2 GB/s decompression
- Snappy: 250 MB/s compression, 500 MB/s decompression

**Tradeoff**: Adds 100-500ns latency

### Encryption

**Problem**: Secure communication

**Solution**: Use TLS or DTLS
- TLS: For TCP connections
- DTLS: For UDP datagrams

**Tradeoff**: Adds 1-10μs latency

## Future Enhancements

1. **Batching**: Send multiple messages in one packet
2. **Compression**: LZ4 or Snappy
3. **Encryption**: TLS or DTLS
4. **Checksums**: CRC32 or xxHash
5. **Versioning**: Protocol version negotiation
