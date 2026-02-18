#include "pool_allocator.h"
#include <stdexcept>
#include <cstring>

namespace risk_engine {
namespace memory {

MemoryPool::MemoryPool(size_t block_size, size_t num_blocks)
    : block_size_(block_size), num_blocks_(num_blocks) {
    
    pool_ = new char[block_size_ * num_blocks_];
    std::memset(pool_, 0, block_size_ * num_blocks_);
    
    free_list_.reserve(num_blocks_);
    for (size_t i = 0; i < num_blocks_; ++i) {
        free_list_.push_back(pool_ + i * block_size_);
    }
}

MemoryPool::~MemoryPool() {
    delete[] pool_;
}

void* MemoryPool::allocate() {
    if (free_list_.empty()) {
        throw std::bad_alloc();
    }
    
    void* ptr = free_list_.back();
    free_list_.pop_back();
    return ptr;
}

void MemoryPool::deallocate(void* ptr) {
    free_list_.push_back(ptr);
}

} // namespace memory
} // namespace risk_engine
