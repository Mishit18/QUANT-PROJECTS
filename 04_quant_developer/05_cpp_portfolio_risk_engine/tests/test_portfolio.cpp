#include "../src/portfolio/position.h"
#include "../src/pricing/option.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace risk_engine;

void test_portfolio_pnl() {
    portfolio::Portfolio port;
    
    port.add_position({
        "CALL_100", 0, 10.0,
        {pricing::OptionType::Call, 100.0, 1.0}
    });
    
    std::vector<double> prices = {110.0};
    double pnl = port.compute_pnl(prices);
    
    assert(std::abs(pnl - 100.0) < 1e-10);
    
    std::cout << "Portfolio PnL test passed (PnL: " << pnl << ")" << std::endl;
}

void test_portfolio_greeks() {
    portfolio::Portfolio port;
    
    port.add_position({
        "CALL_100", 0, 10.0,
        {pricing::OptionType::Call, 100.0, 1.0}
    });
    
    std::vector<double> prices = {100.0};
    std::vector<double> vols = {0.20};
    
    double delta = port.compute_delta(0, prices, 0.05, vols);
    double gamma = port.compute_gamma(0, prices, 0.05, vols);
    double vega = port.compute_vega(0, prices, 0.05, vols);
    
    assert(delta > 0.0);
    assert(gamma > 0.0);
    assert(vega > 0.0);
    
    std::cout << "Portfolio Greeks test passed" << std::endl;
}

int main() {
    test_portfolio_pnl();
    test_portfolio_greeks();
    std::cout << "All portfolio tests passed!" << std::endl;
    return 0;
}
