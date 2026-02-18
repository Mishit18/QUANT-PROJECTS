#include "network/tcp_listener.hpp"
#include "network/udp_listener.hpp"
#include "buffer/ring_buffer.hpp"
#include "buffer/event_queue.hpp"
#include "engine/io_thread.hpp"
#include "engine/decoder_thread.hpp"
#include "dispatcher/event_dispatcher.hpp"
#include "common/events.hpp"
#include <cstdio>
#include <csignal>
#include <atomic>
#include <thread>
#include <chrono>

using namespace hft;

std::atomic<bool> g_running{true};

void signal_handler(int) {
    g_running.store(false);
}

// Example consumer callback (no allocation on hot path)
void market_data_consumer(const Event& event, void* context) {
    switch (event.type) {
        case MsgType::ORDER_BOOK_UPDATE:
            // Process order book update
            // In production: update order book, calculate signals, etc.
            break;
        case MsgType::TRADE:
            // Process trade
            break;
        case MsgType::EXECUTION_REPORT:
            // Process execution report
            break;
        default:
            break;
    }
}

int main(int argc, char** argv) {
    printf("HFT Feed Handler Starting...\n");
    
    // Configuration
    uint16_t udp_port = 9000;
    uint16_t tcp_port = 9001;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (arg[0] == '-' && arg[1] == '-') {
            if (strcmp(arg, "--udp-port") == 0 && i + 1 < argc) {
                udp_port = (uint16_t)atoi(argv[++i]);
            } else if (strcmp(arg, "--tcp-port") == 0 && i + 1 < argc) {
                tcp_port = (uint16_t)atoi(argv[++i]);
            }
        }
    }
    
    printf("UDP Port: %u\n", udp_port);
    printf("TCP Port: %u\n", tcp_port);
    
    // Allocate buffers (16MB ring buffer, 64K event queue)
    RingBuffer ring_buffer(16 * 1024 * 1024);
    EventQueue<65536> event_queue;
    
    // Create network listeners
    UDPListener udp_listener(udp_port, ring_buffer);
    TCPListener tcp_listener(tcp_port, ring_buffer);
    
    // Initialize
    if (udp_listener.initialize() != Result::OK) {
        fprintf(stderr, "Failed to initialize UDP listener\n");
        return 1;
    }
    
    if (tcp_listener.initialize() != Result::OK) {
        fprintf(stderr, "Failed to initialize TCP listener\n");
        return 1;
    }
    
    printf("Network listeners initialized\n");
    
    // Create threads
    IOThread io_thread(tcp_listener, udp_listener, ring_buffer);
    DecoderThread decoder_thread(ring_buffer, event_queue);
    EventDispatcher dispatcher(event_queue);
    
    // Register consumers (no std::function, direct function pointer)
    dispatcher.register_consumer(market_data_consumer, nullptr);
    
    // Start threads with CPU affinity
    // Core 0: IO
    // Core 1: Decoder
    // Core 2: Dispatcher
    printf("Starting threads...\n");
    io_thread.start(0);
    decoder_thread.start(1);
    dispatcher.start(2);
    
    printf("Feed handler running. Press Ctrl+C to stop.\n");
    
    // Install signal handler
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    
    // Main loop: print statistics
    while (g_running.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        
        printf("\n=== Statistics ===\n");
        printf("IO Thread:\n");
        printf("  Packets received: %llu\n", (unsigned long long)io_thread.get_packets_received());
        printf("Decoder Thread:\n");
        printf("  Messages parsed: %llu\n", (unsigned long long)decoder_thread.get_messages_parsed());
        printf("  Parse errors: %llu\n", (unsigned long long)decoder_thread.get_parse_errors());
        printf("Dispatcher:\n");
        printf("  Events dispatched: %llu\n", (unsigned long long)dispatcher.get_events_dispatched());
        printf("Ring Buffer:\n");
        printf("  Available read: %zu bytes\n", ring_buffer.available_read());
        printf("Event Queue:\n");
        printf("  Size: %zu\n", event_queue.size());
    }
    
    printf("\nShutting down...\n");
    
    // Stop threads
    io_thread.stop();
    decoder_thread.stop();
    dispatcher.stop();
    
    printf("Shutdown complete\n");
    return 0;
}
