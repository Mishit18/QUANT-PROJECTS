#pragma once

#include "../core/Event.h"
#include <vector>
#include <memory>
#include <fstream>
#include <sstream>
#include <string>
#include <random>

namespace simulator {

// Replays market data from file or synthetic generation
class MarketDataReplayer {
public:
    MarketDataReplayer() = default;
    
    // Load events from CSV file
    // Format: timestamp,type,symbol,price,quantity
    bool loadFromCSV(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            return false;
        }
        
        std::string line;
        std::getline(file, line);  // Skip header
        
        while (std::getline(file, line)) {
            auto event = parseCSVLine(line);
            if (event) {
                events_.push_back(event);
            }
        }
        
        return true;
    }
    
    // Generate synthetic order flow
    void generateSyntheticOrders(size_t count, const std::string& symbol,
                                 double base_price, uint64_t time_delta_ns) {
        std::mt19937_64 gen(42);  // Fixed seed for determinism
        std::uniform_real_distribution<> price_dist(-0.01, 0.01);
        std::uniform_int_distribution<> qty_dist(100, 1000);
        std::uniform_int_distribution<> side_dist(0, 1);
        
        uint64_t timestamp = 0;
        for (size_t i = 0; i < count; ++i) {
            timestamp += time_delta_ns;
            
            double price = base_price * (1.0 + price_dist(gen));
            uint64_t quantity = qty_dist(gen);
            bool is_buy = side_dist(gen) == 1;
            
            auto event = std::make_shared<OrderSubmitEvent>(
                timestamp, i, symbol, is_buy, price, quantity);
            events_.push_back(event);
        }
    }
    
    const std::vector<std::shared_ptr<Event>>& getEvents() const {
        return events_;
    }

private:
    std::shared_ptr<Event> parseCSVLine(const std::string& line) {
        // Simple CSV parser - production would be more robust
        return nullptr;  // Placeholder
    }

private:
    std::vector<std::shared_ptr<Event>> events_;
};

} // namespace simulator
