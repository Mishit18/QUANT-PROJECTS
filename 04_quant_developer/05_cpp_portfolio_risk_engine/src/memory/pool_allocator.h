#pragma once

#include <vector>
#include <cstddef>
#include <memory>

namespace risk_engine {
namespace memory {

class MemoryPool {
public:
    explicit MemoryPool(size_t block_size, size_t num_blocks);
    ~MemoryPool();
    
    void* allocate();
    void deallocate(void* ptr);
    
    size_t block_size() const { return block_size_; }
    size_t num_blocks() const { return num_blocks_; }
    
private:
    size_t block_size_;
    size_t num_blocks_;
    char* pool_;
    std::vector<void*> free_list_;
};

template<typename T>
class PoolAllocator {
public:
    using value_type = T;
    
    explicit PoolAllocator(MemoryPool& pool) : pool_(pool) {}
    
    template<typename U>
    PoolAllocator(const PoolAllocator<U>& other) : pool_(other.pool_) {}
    
    T* allocate(size_t n) {
        if (n * sizeof(T) > pool_.block_size()) {
            return static_cast<T*>(::operator new(n * sizeof(T)));
        }
        return static_cast<T*>(pool_.allocate());
    }
    
    void deallocate(T* ptr, size_t n) {
        if (n * sizeof(T) > pool_.block_size()) {
            ::operator delete(ptr);
        } else {
            pool_.deallocate(ptr);
        }
    }
    
private:
    MemoryPool& pool_;
    
    template<typename U>
    friend class PoolAllocator;
};

} // namespace memory
} // namespace risk_engine
