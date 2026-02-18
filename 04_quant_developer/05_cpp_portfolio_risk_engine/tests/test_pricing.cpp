#include "../src/pricing/option.h"
#include <iostream>
#include <cmath>
#include <cassert>

using namespace risk_engine::pricing;

void test_call_payoff() {
    OptionSpec call{OptionType::Call, 100.0, 1.0};
    
    assert(std::abs(OptionPricer::payoff(call, 110.0) - 10.0) < 1e-10);
    assert(std::abs(OptionPricer::payoff(call, 90.0) - 0.0) < 1e-10);
    
    std::cout << "Call payoff test passed" << std::endl;
}

void test_put_payoff() {
    OptionSpec put{OptionType::Put, 100.0, 1.0};
    
    assert(std::abs(OptionPricer::payoff(put, 90.0) - 10.0) < 1e-10);
    assert(std::abs(OptionPricer::payoff(put, 110.0) - 0.0) < 1e-10);
    
    std::cout << "Put payoff test passed" << std::endl;
}

void test_black_scholes() {
    OptionSpec call{OptionType::Call, 100.0, 1.0};
    double price = OptionPricer::black_scholes(call, 100.0, 0.05, 0.20);
    
    assert(price > 0.0 && price < 100.0);
    
    std::cout << "Black-Scholes test passed (price: " << price << ")" << std::endl;
}

void test_greeks() {
    OptionSpec call{OptionType::Call, 100.0, 1.0};
    
    double delta = OptionPricer::delta(call, 100.0, 0.05, 0.20);
    double gamma = OptionPricer::gamma(call, 100.0, 0.05, 0.20);
    double vega = OptionPricer::vega(call, 100.0, 0.05, 0.20);
    
    assert(delta > 0.0 && delta < 1.0);
    assert(gamma > 0.0);
    assert(vega > 0.0);
    
    std::cout << "Greeks test passed (delta: " << delta << ", gamma: " 
              << gamma << ", vega: " << vega << ")" << std::endl;
}

int main() {
    test_call_payoff();
    test_put_payoff();
    test_black_scholes();
    test_greeks();
    std::cout << "All pricing tests passed!" << std::endl;
    return 0;
}
