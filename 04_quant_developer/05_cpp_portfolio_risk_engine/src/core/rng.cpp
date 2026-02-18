#include "rng.h"
#include <stdexcept>
#include <cstring>

namespace risk_engine {
namespace core {

CorrelatedRNG::CorrelatedRNG(const std::vector<std::vector<double>>& correlation_matrix, 
                             uint64_t seed, uint64_t stream_id)
    : dim_(correlation_matrix.size()), rng_(seed, stream_id) {
    
    if (dim_ == 0) {
        throw std::invalid_argument("Correlation matrix cannot be empty");
    }
    
    for (const auto& row : correlation_matrix) {
        if (row.size() != dim_) {
            throw std::invalid_argument("Correlation matrix must be square");
        }
    }
    
    compute_cholesky(correlation_matrix);
}

void CorrelatedRNG::compute_cholesky(const std::vector<std::vector<double>>& corr) {
    cholesky_lower_.resize(dim_ * dim_, 0.0);
    
    for (size_t i = 0; i < dim_; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += cholesky_lower_[i * dim_ + k] * cholesky_lower_[j * dim_ + k];
            }
            
            if (i == j) {
                double val = corr[i][i] - sum;
                if (val <= 0.0) {
                    throw std::runtime_error("Correlation matrix is not positive definite");
                }
                cholesky_lower_[i * dim_ + j] = std::sqrt(val);
            } else {
                cholesky_lower_[i * dim_ + j] = (corr[i][j] - sum) / cholesky_lower_[j * dim_ + j];
            }
        }
    }
}

void CorrelatedRNG::generate_correlated_normals(std::vector<double>& output) {
    output.resize(dim_);
    std::vector<double> independent(dim_);
    
    for (size_t i = 0; i < dim_; ++i) {
        independent[i] = rng_.normal();
    }
    
    for (size_t i = 0; i < dim_; ++i) {
        output[i] = 0.0;
        for (size_t j = 0; j <= i; ++j) {
            output[i] += cholesky_lower_[i * dim_ + j] * independent[j];
        }
    }
}

} // namespace core
} // namespace risk_engine
