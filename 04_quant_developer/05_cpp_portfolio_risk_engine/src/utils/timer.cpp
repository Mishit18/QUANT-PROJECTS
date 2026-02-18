#include "timer.h"
#include <iostream>

namespace risk_engine {
namespace utils {

ScopedTimer::ScopedTimer(const std::string& name) : name_(name) {}

ScopedTimer::~ScopedTimer() {
    std::cout << name_ << ": " << timer_.elapsed_ms() << " ms" << std::endl;
}

} // namespace utils
} // namespace risk_engine
