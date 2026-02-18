#pragma once

#include <vector>
#include <stdexcept>

namespace heston {

class TridiagonalMatrix {
public:
    TridiagonalMatrix(size_t n) 
        : n_(n), lower_(n - 1), diag_(n), upper_(n - 1) {}
    
    void set_lower(size_t i, double val) { lower_[i] = val; }
    void set_diag(size_t i, double val) { diag_[i] = val; }
    void set_upper(size_t i, double val) { upper_[i] = val; }
    
    double lower(size_t i) const { return lower_[i]; }
    double diag(size_t i) const { return diag_[i]; }
    double upper(size_t i) const { return upper_[i]; }
    
    size_t size() const { return n_; }
    
    void solve(const std::vector<double>& rhs, std::vector<double>& solution) const;
    
private:
    size_t n_;
    std::vector<double> lower_;
    std::vector<double> diag_;
    std::vector<double> upper_;
};

class SparseMatrix {
public:
    SparseMatrix(size_t rows, size_t cols) : rows_(rows), cols_(cols) {}
    
    void add_entry(size_t row, size_t col, double val);
    void multiply(const std::vector<double>& x, std::vector<double>& y) const;
    
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    
private:
    size_t rows_;
    size_t cols_;
    std::vector<size_t> row_ptr_;
    std::vector<size_t> col_idx_;
    std::vector<double> values_;
};

} // namespace heston
