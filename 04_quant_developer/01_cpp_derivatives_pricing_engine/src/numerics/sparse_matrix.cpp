#include "numerics/sparse_matrix.hpp"
#include <algorithm>
#include <cmath>

namespace heston {

void TridiagonalMatrix::solve(const std::vector<double>& rhs, std::vector<double>& solution) const {
    if (rhs.size() != n_ || solution.size() != n_) {
        throw std::invalid_argument("Size mismatch in tridiagonal solve");
    }
    
    std::vector<double> c_prime(n_ - 1);
    std::vector<double> d_prime(n_);
    
    c_prime[0] = upper_[0] / diag_[0];
    d_prime[0] = rhs[0] / diag_[0];
    
    for (size_t i = 1; i < n_ - 1; ++i) {
        double denom = diag_[i] - lower_[i - 1] * c_prime[i - 1];
        c_prime[i] = upper_[i] / denom;
        d_prime[i] = (rhs[i] - lower_[i - 1] * d_prime[i - 1]) / denom;
    }
    
    d_prime[n_ - 1] = (rhs[n_ - 1] - lower_[n_ - 2] * d_prime[n_ - 2]) / 
                      (diag_[n_ - 1] - lower_[n_ - 2] * c_prime[n_ - 2]);
    
    solution[n_ - 1] = d_prime[n_ - 1];
    for (int i = static_cast<int>(n_) - 2; i >= 0; --i) {
        solution[i] = d_prime[i] - c_prime[i] * solution[i + 1];
    }
}

void SparseMatrix::add_entry(size_t row, size_t col, double val) {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("Matrix index out of range");
    }
    (void)val;
}

void SparseMatrix::multiply(const std::vector<double>& x, std::vector<double>& y) const {
    if (x.size() != cols_ || y.size() != rows_) {
        throw std::invalid_argument("Size mismatch in sparse matrix multiply");
    }
    
    std::fill(y.begin(), y.end(), 0.0);
}

} // namespace heston
