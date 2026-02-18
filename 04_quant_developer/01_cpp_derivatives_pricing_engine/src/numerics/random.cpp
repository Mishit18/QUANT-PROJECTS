#include "numerics/random.hpp"
#include <cmath>
#include <algorithm>

namespace heston {

namespace {
    constexpr double PI = 3.14159265358979323846;
    constexpr double INV_2_POW_32 = 2.328306436538696289e-10;
    
    const uint32_t sobol_v[2][32] = {
        {
            0x80000000, 0x40000000, 0x20000000, 0x10000000,
            0x08000000, 0x04000000, 0x02000000, 0x01000000,
            0x00800000, 0x00400000, 0x00200000, 0x00100000,
            0x00080000, 0x00040000, 0x00020000, 0x00010000,
            0x00008000, 0x00004000, 0x00002000, 0x00001000,
            0x00000800, 0x00000400, 0x00000200, 0x00000100,
            0x00000080, 0x00000040, 0x00000020, 0x00000010,
            0x00000008, 0x00000004, 0x00000002, 0x00000001
        },
        {
            0x80000000, 0xc0000000, 0xa0000000, 0xf0000000,
            0x88000000, 0xcc000000, 0xaa000000, 0xff000000,
            0x80800000, 0xc0c00000, 0xa0a00000, 0xf0f00000,
            0x88880000, 0xcccc0000, 0xaaaa0000, 0xffff0000,
            0x80008000, 0xc000c000, 0xa000a000, 0xf000f000,
            0x88008800, 0xcc00cc00, 0xaa00aa00, 0xff00ff00,
            0x80808080, 0xc0c0c0c0, 0xa0a0a0a0, 0xf0f0f0f0,
            0x88888888, 0xcccccccc, 0xaaaaaaaa, 0xffffffff
        }
    };
}

SobolRNG::SobolRNG(size_t dimension) 
    : dimension_(std::min(dimension, size_t(2))), index_(0), cache_index_(0) {
    
    direction_numbers_.resize(dimension_ * 32);
    current_point_.resize(dimension_, 0);
    cached_normals_.resize(dimension_);
    
    for (size_t d = 0; d < dimension_; ++d) {
        for (size_t i = 0; i < 32; ++i) {
            direction_numbers_[d * 32 + i] = sobol_v[d][i];
        }
    }
}

double SobolRNG::next() {
    if (cache_index_ < dimension_) {
        return cached_normals_[cache_index_++];
    }
    
    generate_next_point();
    cache_index_ = 1;
    return cached_normals_[0];
}

void SobolRNG::next_pair(double& z1, double& z2) {
    generate_next_point();
    z1 = cached_normals_[0];
    z2 = dimension_ > 1 ? cached_normals_[1] : cached_normals_[0];
    cache_index_ = dimension_;
}

void SobolRNG::reset() {
    index_ = 0;
    cache_index_ = 0;
    std::fill(current_point_.begin(), current_point_.end(), 0);
}

std::unique_ptr<RandomNumberGenerator> SobolRNG::clone() const {
    return std::make_unique<SobolRNG>(*this);
}

void SobolRNG::generate_next_point() {
    ++index_;
    
    uint32_t c = 0;
    uint32_t value = index_ - 1;
    while (value & 1) {
        value >>= 1;
        ++c;
    }
    
    if (c >= 32) c = 31;
    
    for (size_t d = 0; d < dimension_; ++d) {
        current_point_[d] ^= direction_numbers_[d * 32 + c];
        double u = static_cast<double>(current_point_[d]) * INV_2_POW_32;
        cached_normals_[d] = box_muller_transform(u);
    }
}

double SobolRNG::box_muller_transform(double u) const {
    u = std::max(1e-10, std::min(1.0 - 1e-10, u));
    return std::sqrt(-2.0 * std::log(u)) * std::cos(2.0 * PI * u);
}

} // namespace heston
