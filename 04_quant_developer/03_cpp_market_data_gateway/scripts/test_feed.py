#!/usr/bin/env python3
"""
Test feed generator for HFT feed handler
Sends binary market data messages via UDP
"""

import socket
import struct
import time
import random

def create_order_book_update(sequence, symbol, price, quantity, side, level):
    """Create binary OrderBookUpdate message"""
    # Header: type(u8) reserved(u8) length(u16) reserved(u32) sequence(u64)
    header = struct.pack('<BBHIQ', 1, 0, 40, 0, sequence)
    
    # Payload: price(i64) quantity(i64) side(u8) level(u8) symbol(16) padding(6)
    symbol_bytes = symbol.encode('ascii').ljust(16, b'\x00')
    payload = struct.pack('<qqBB', price, quantity, side, level)
    payload += symbol_bytes
    payload += b'\x00' * 6  # padding
    
    return header + payload

def create_trade(sequence, symbol, price, quantity, side):
    """Create binary Trade message"""
    # Header
    header = struct.pack('<BBHIQ', 2, 0, 32, 0, sequence)
    
    # Payload: price(i64) quantity(i64) side(u8) symbol(16) padding(7)
    symbol_bytes = symbol.encode('ascii').ljust(16, b'\x00')
    payload = struct.pack('<qqB', price, quantity, side)
    payload += symbol_bytes
    payload += b'\x00' * 7  # padding
    
    return header + payload

def main():
    # Create UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    host = '127.0.0.1'
    port = 9000
    
    print(f"Sending test feed to {host}:{port}")
    print("Press Ctrl+C to stop")
    
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    sequence = 1
    
    try:
        while True:
            symbol = random.choice(symbols)
            price = random.randint(100_0000, 200_0000)  # $100-$200 in fixed-point
            quantity = random.randint(100, 10000)
            side = random.randint(0, 1)
            level = random.randint(0, 4)
            
            # Send order book update
            msg = create_order_book_update(sequence, symbol, price, quantity, side, level)
            sock.sendto(msg, (host, port))
            sequence += 1
            
            # Occasionally send trade
            if random.random() < 0.3:
                msg = create_trade(sequence, symbol, price, quantity, side)
                sock.sendto(msg, (host, port))
                sequence += 1
            
            # Rate limit: 10K messages/sec
            time.sleep(0.0001)
            
            if sequence % 10000 == 0:
                print(f"Sent {sequence} messages")
    
    except KeyboardInterrupt:
        print(f"\nSent {sequence} messages total")
        sock.close()

if __name__ == '__main__':
    main()
