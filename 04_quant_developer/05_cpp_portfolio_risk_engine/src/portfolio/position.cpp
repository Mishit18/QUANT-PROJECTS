#include "position.h"
#include "../pricing/option.h"

namespace risk_engine {
namespace portfolio {

void Portfolio::add_position(const Position& pos) {
    positions_.push_back(pos);
}

double Portfolio::compute_pnl(const std::vector<double>& asset_prices) const {
    double total_pnl = 0.0;
    
    for (const auto& pos : positions_) {
        double spot = asset_prices[pos.asset_index];
        double payoff = pricing::OptionPricer::payoff(pos.option_spec, spot);
        total_pnl += pos.quantity * payoff;
    }
    
    return total_pnl;
}

double Portfolio::compute_delta(size_t asset_idx, const std::vector<double>& asset_prices,
                               double rate, const std::vector<double>& vols) const {
    double total_delta = 0.0;
    
    for (const auto& pos : positions_) {
        if (pos.asset_index == asset_idx) {
            double spot = asset_prices[pos.asset_index];
            double vol = vols[pos.asset_index];
            double delta = pricing::OptionPricer::delta(pos.option_spec, spot, rate, vol);
            total_delta += pos.quantity * delta;
        }
    }
    
    return total_delta;
}

double Portfolio::compute_gamma(size_t asset_idx, const std::vector<double>& asset_prices,
                               double rate, const std::vector<double>& vols) const {
    double total_gamma = 0.0;
    
    for (const auto& pos : positions_) {
        if (pos.asset_index == asset_idx) {
            double spot = asset_prices[pos.asset_index];
            double vol = vols[pos.asset_index];
            double gamma = pricing::OptionPricer::gamma(pos.option_spec, spot, rate, vol);
            total_gamma += pos.quantity * gamma;
        }
    }
    
    return total_gamma;
}

double Portfolio::compute_vega(size_t asset_idx, const std::vector<double>& asset_prices,
                              double rate, const std::vector<double>& vols) const {
    double total_vega = 0.0;
    
    for (const auto& pos : positions_) {
        if (pos.asset_index == asset_idx) {
            double spot = asset_prices[pos.asset_index];
            double vol = vols[pos.asset_index];
            double vega = pricing::OptionPricer::vega(pos.option_spec, spot, rate, vol);
            total_vega += pos.quantity * vega;
        }
    }
    
    return total_vega;
}

} // namespace portfolio
} // namespace risk_engine
