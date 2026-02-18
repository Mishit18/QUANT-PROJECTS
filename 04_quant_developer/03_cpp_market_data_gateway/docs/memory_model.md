# Memory Model

## Design Philosophy

Pre-allocated memory at startup. No heap allocation during message processing.

## Memory Allocation Strategy

### Startup Phase (Cold Path)

- Allocate all buffers and queues
- Allocate thread stacks
- Load configuration

**Allowed**: Dynamic allocation, STL containers, exceptions

### Runtime Phase (Hot Path)

- Process messages
- Parse protocols
- Dispatch events

**Forbidden**: Dynamic allocation, STL containers, exceptions

## Memory Regions

### Ring Buffer (16MB)

**Purpose**: Staging area for raw network data

**Allocation**:
```cpp
posix_memalign((void**)&buffer_, CACHE_LINE_SIZE, capacity_);
```

**Alignment**: 64-byte (cache line)

**Why 16MB?**
- Holds ~100K messages (avg 150 bytes)
- Enough for burst traffic
- Fits in L3 cache on modern CPUs

**Tradeoff**: Larger = more memory, but better burst handling

### Event Queue (64K events)

**Purpose**: Staging area for parsed events

**Allocation**: Stack-allocated array
```cpp
Event buffer_[Capacity];
```

**Size**: 64K × 64 bytes = 4MB

**Why 64K events?**
- Enough for ~10ms of traffic at 5M msg/sec
- Power of 2 for fast modulo

**Tradeoff**: Larger = more memory, but better burst handling

### Event Structs (64 bytes each)

**Purpose**: Normalized market data events

**Alignment**: 64-byte (cache line)

**Why 64 bytes?**
- Fits exactly one cache line
- No false sharing between events
- Fast memcpy (single cache line)

**Layout**:
```cpp
struct CACHE_ALIGNED OrderBookUpdate {
    u64 timestamp_ns;      // 8 bytes
    SequenceNum sequence;  // 8 bytes
    Price price;           // 8 bytes
    Quantity quantity;     // 8 bytes
    Side side;             // 1 byte
    u8 level;              // 1 byte
    char symbol[16];       // 16 bytes
    u8 _padding[22];       // 22 bytes (pad to 64)
};
```

### Partial Message Buffers (4KB each)

**Purpose**: Reassemble partial messages across packets

**Allocation**: Stack-allocated in parser
```cpp
char partial_buffer_[PARTIAL_BUFFER_SIZE];
```

**Why 4KB?**
- Larger than any single message
- Fits in L1 cache

## Cache-Line Alignment

### Why 64-byte alignment?

1. No false sharing: Each structure on separate cache line
2. Fast access: Aligned loads are faster
3. Predictable: No cache-line splits

### What to align?

- Ring buffer pointers (write_pos, read_pos)
- Event queue indices (write_idx, read_idx)
- Event structs (OrderBookUpdate, Trade, etc.)
- Thread-local data

### How to align?

```cpp
#define CACHE_ALIGNED alignas(64)

struct CACHE_ALIGNED MyStruct {
    // ...
};
```

## Minimal Copy Techniques

### Where Minimal Copies Work

1. Binary protocol parsing:
   - Struct overlay on buffer (with alignment checks)
   - No memcpy, just pointer cast
   ```cpp
   Header hdr;
   memcpy(&hdr, buffer, sizeof(Header)); // Safe alignment
   ```

2. FIX tag extraction:
   - StringView points into buffer
   - No string allocation
   ```cpp
   StringView tag_value{buffer + offset, length};
   ```

3. Ring buffer read:
   - Decoder reads directly from buffer
   - No intermediate copy

### Where Copies Are Necessary

1. Partial message reassembly:
   - Must copy to partial buffer
   - Unavoidable with TCP

2. Endianness conversion:
   - Must swap bytes
   - Can't avoid copy

3. Event normalization:
   - Must copy to event struct
   - Necessary for type safety

## Memory Barriers

### Why Memory Barriers?

- Reordering: CPU and compiler may reorder memory operations
- Visibility: Changes on one core may not be visible on another
- Correctness: Lock-free algorithms require proper ordering

### Memory Ordering

```cpp
// Relaxed: No ordering guarantees (for statistics)
counter_.fetch_add(1, std::memory_order_relaxed);

// Acquire: Reads after this can't be reordered before
size_t r = read_pos_.load(std::memory_order_acquire);

// Release: Writes before this can't be reordered after
write_pos_.store(new_pos, std::memory_order_release);
```

### SPSC Queue Pattern

```cpp
// Producer
buffer_[write_idx] = data;  // Write data
write_idx_.store(next_idx, std::memory_order_release);  // Publish

// Consumer
size_t r = read_idx_.load(std::memory_order_relaxed);
size_t w = write_idx_.load(std::memory_order_acquire);  // See published data
if (r != w) {
    data = buffer_[r];  // Read data
    read_idx_.store(next_r, std::memory_order_release);
}
```

## NUMA Considerations

### What is NUMA?

- **Non-Uniform Memory Access**: Memory access time depends on location
- **Local memory**: Fast (same NUMA node as CPU)
- **Remote memory**: Slow (different NUMA node)

### How to Handle NUMA?

1. **Pin threads to cores on same NUMA node**:
   ```bash
   numactl --cpunodebind=0 --membind=0 ./feed_handler
   ```

2. **Allocate memory on same NUMA node as NIC**:
   ```cpp
   numa_alloc_onnode(size, node);
   ```

3. **Check NUMA topology**:
   ```bash
   numactl --hardware
   lscpu
   ```

## Huge Pages

### Why Huge Pages?

- Fewer TLB misses: 2MB pages vs 4KB pages
- Lower overhead: Fewer page table entries
- Better performance: 5-10% improvement

### How to Enable?

```bash
# Allocate 1024 × 2MB = 2GB huge pages
echo 1024 > /proc/sys/vm/nr_hugepages

# Use in application
mmap(..., MAP_HUGETLB, ...);
```

## Memory Profiling

### Tools

- **Valgrind**: Detect leaks and invalid access
- **AddressSanitizer**: Detect buffer overflows
- **perf mem**: Profile memory access patterns
- **Intel VTune**: Advanced memory profiling

### Metrics to Monitor

- **Cache miss rate**: L1, L2, L3
- **TLB miss rate**: Page table lookups
- **Memory bandwidth**: Bytes/sec
- **NUMA remote access**: Cross-node traffic

## Common Pitfalls

### False Sharing

**Problem**: Two threads access different variables on same cache line

**Example**:
```cpp
struct Bad {
    std::atomic<int> counter1;  // Cache line 0
    std::atomic<int> counter2;  // Cache line 0 (same!)
};
```

**Solution**: Pad to separate cache lines
```cpp
struct Good {
    alignas(64) std::atomic<int> counter1;  // Cache line 0
    alignas(64) std::atomic<int> counter2;  // Cache line 1
};
```

### Unaligned Access

**Problem**: Struct fields not aligned to natural boundaries

**Example**:
```cpp
struct Bad {
    u8 a;
    u64 b;  // Unaligned!
} __attribute__((packed));
```

**Solution**: Use natural alignment or explicit padding

### Memory Leaks

**Problem**: Allocate memory but never free

**Solution**: Pre-allocate at startup, no runtime allocation
