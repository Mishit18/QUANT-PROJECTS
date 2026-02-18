#pragma once

#include "../pricing/option.h"
#include <vector>
#include <string>

namespace risk_engine {
namespace portfolio {

struct Position {
    std::string instrument_id;
    size_t asset_index;
    double quantity;
    pricing::OptionSpec option_spec;
};

class Portfolio {
public:
    void add_position(const Position& pos);
    
    const std::vector<Position>& positions() const { return positions_; }
    
    double compute_pnl(const std::vector<double>& asset_prices) const;
    
    double compute_delta(size_t asset_idx, const std::vector<double>& asset_prices, 
                        double rate, const std::vector<double>& vols) const;
    
    double compute_gamma(size_t asset_idx, const std::vector<double>& asset_prices,
                        double rate, const std::vector<double>& vols) const;
    
    double compute_vega(size_t asset_idx, const std::vector<double>& asset_prices,
                       double rate, const std::vector<double>& vols) const;
    
private:
    std::vector<Position> positions_;
};

} // namespace portfolio
} // namespace risk_engine
