#include "engine/decoder_thread.hpp"
#include <pthread.h>
#include <sched.h>
#include <emmintrin.h> // For _mm_pause

namespace hft {

DecoderThread::DecoderThread(RingBuffer& input, EventQueue<65536>& output)
    : input_buffer_(input), output_queue_(output),
      running_(false), messages_parsed_(0), parse_errors_(0) {}

DecoderThread::~DecoderThread() {
    stop();
}

void DecoderThread::start(int cpu_affinity) {
    running_.store(true, std::memory_order_release);
    
    thread_ = std::thread([this, cpu_affinity]() {
        // Set CPU affinity
        if (cpu_affinity >= 0) {
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(cpu_affinity, &cpuset);
            pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
        }
        
        // Set thread priority
        sched_param param;
        param.sched_priority = sched_get_priority_max(SCHED_FIFO);
        pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
        
        run();
    });
}

void DecoderThread::stop() {
    running_.store(false, std::memory_order_release);
    if (thread_.joinable()) {
        thread_.join();
    }
}

DecoderThread::Protocol DecoderThread::detect_protocol(const char* data, size_t length) noexcept {
    if (length < 5) return Protocol::UNKNOWN;
    
    // FIX starts with "8=FIX"
    if (data[0] == '8' && data[1] == '=' && data[2] == 'F' && 
        data[3] == 'I' && data[4] == 'X') {
        return Protocol::FIX;
    }
    
    // Binary has specific message type range
    u8 msg_type = (u8)data[0];
    if (msg_type >= 1 && msg_type <= 10) {
        return Protocol::BINARY;
    }
    
    return Protocol::UNKNOWN;
}

void DecoderThread::run() {
    while (running_.load(std::memory_order_acquire)) {
        // Read from ring buffer
        size_t available = 0;
        const char* data = input_buffer_.peek_read(available);
        
        if (!data || available == 0) {
            // No data, pause CPU to reduce power and avoid busy-wait
            _mm_pause();
            continue;
        }
        
        // Detect protocol
        Protocol proto = detect_protocol(data, available);
        
        Event event;
        size_t consumed = 0;
        Result result = Result::ERROR_INVALID_MESSAGE;
        
        switch (proto) {
            case Protocol::FIX:
                result = fix_parser_.parse(data, available, event, consumed);
                break;
            case Protocol::BINARY:
                result = binary_parser_.parse(data, available, event, consumed);
                break;
            default:
                // Unknown protocol, skip one byte
                consumed = 1;
                result = Result::ERROR_INVALID_MESSAGE;
        }
        
        if (result == Result::OK) {
            // Push to output queue
            if (!output_queue_.try_push(event)) {
                // Queue full, backpressure
                // In production, implement proper backpressure handling
            } else {
                messages_parsed_.fetch_add(1, std::memory_order_relaxed);
            }
            input_buffer_.commit_read(consumed);
        } else if (result == Result::ERROR_INCOMPLETE_MESSAGE) {
            // Need more data, wait
            break;
        } else {
            // Parse error, skip consumed bytes
            parse_errors_.fetch_add(1, std::memory_order_relaxed);
            if (consumed > 0) {
                input_buffer_.commit_read(consumed);
            } else {
                // Skip one byte to avoid infinite loop
                input_buffer_.commit_read(1);
            }
        }
    }
}

} // namespace hft
