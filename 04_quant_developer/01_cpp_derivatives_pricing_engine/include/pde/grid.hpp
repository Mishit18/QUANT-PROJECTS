#pragma once

#include <vector>
#include <cmath>

namespace heston {

class Grid1D {
public:
    Grid1D(double min, double max, size_t num_points, bool log_spacing = false);
    
    double operator[](size_t i) const { return points_[i]; }
    size_t size() const { return points_.size(); }
    double min() const { return points_.front(); }
    double max() const { return points_.back(); }
    double spacing(size_t i) const { return i < size() - 1 ? points_[i + 1] - points_[i] : 0.0; }
    
    size_t find_index(double x) const;
    
private:
    std::vector<double> points_;
};

class Grid2D {
public:
    Grid2D(const Grid1D& grid_x, const Grid1D& grid_y)
        : grid_x_(grid_x), grid_y_(grid_y) {}
    
    size_t nx() const { return grid_x_.size(); }
    size_t ny() const { return grid_y_.size(); }
    size_t size() const { return nx() * ny(); }
    
    double x(size_t i) const { return grid_x_[i]; }
    double y(size_t j) const { return grid_y_[j]; }
    
    size_t index(size_t i, size_t j) const { return i * ny() + j; }
    void indices(size_t idx, size_t& i, size_t& j) const {
        i = idx / ny();
        j = idx % ny();
    }
    
    const Grid1D& grid_x() const { return grid_x_; }
    const Grid1D& grid_y() const { return grid_y_; }
    
private:
    Grid1D grid_x_;
    Grid1D grid_y_;
};

} // namespace heston
