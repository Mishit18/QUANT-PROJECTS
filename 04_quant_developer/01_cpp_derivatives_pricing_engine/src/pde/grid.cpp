#include "pde/grid.hpp"
#include <algorithm>
#include <cmath>

namespace heston {

Grid1D::Grid1D(double min, double max, size_t num_points, bool log_spacing) {
    points_.resize(num_points);
    
    if (log_spacing && min > 0.0) {
        double log_min = std::log(min);
        double log_max = std::log(max);
        double log_step = (log_max - log_min) / (num_points - 1);
        
        for (size_t i = 0; i < num_points; ++i) {
            points_[i] = std::exp(log_min + i * log_step);
        }
    } else {
        double step = (max - min) / (num_points - 1);
        for (size_t i = 0; i < num_points; ++i) {
            points_[i] = min + i * step;
        }
    }
}

size_t Grid1D::find_index(double x) const {
    if (x <= points_.front()) return 0;
    if (x >= points_.back()) return points_.size() - 2;
    
    auto it = std::lower_bound(points_.begin(), points_.end(), x);
    size_t idx = std::distance(points_.begin(), it);
    return idx > 0 ? idx - 1 : 0;
}

} // namespace heston
