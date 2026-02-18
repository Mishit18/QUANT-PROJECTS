#pragma once

#include <cstddef>
#include <new>
#include <cstdlib>

#ifdef _MSC_VER
#include <malloc.h>
#endif

namespace lob {

constexpr size_t MAX_ORDERS = 1'000'000;

// Lock-free memory pool with free-list allocation
// O(1) alloc/dealloc, deterministic latency
template<typename T, size_t N>
class MemoryPool {
public:
    MemoryPool() noexcept {
        // Cross-platform aligned allocation
#ifdef _MSC_VER
        storage_ = static_cast<T*>(_aligned_malloc(sizeof(T) * N, 64));
#else
        storage_ = static_cast<T*>(aligned_alloc(64, sizeof(T) * N));
#endif
        
        // Build free list
        free_list_ = &storage_[0];
        for (size_t i = 0; i < N - 1; ++i) {
            *reinterpret_cast<T**>(&storage_[i]) = &storage_[i + 1];
        }
        *reinterpret_cast<T**>(&storage_[N - 1]) = nullptr;
        
        allocated_ = 0;
    }
    
    ~MemoryPool() noexcept {
#ifdef _MSC_VER
        _aligned_free(storage_);
#else
        free(storage_);
#endif
    }
    
    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;
    
    // HOT PATH: O(1) allocation
    T* allocate() noexcept {
        if (!free_list_) {
            return nullptr;
        }
        
        T* ptr = free_list_;
        free_list_ = *reinterpret_cast<T**>(ptr);
        ++allocated_;
        
        return new (ptr) T();
    }
    
    // HOT PATH: O(1) deallocation
    void deallocate(T* ptr) noexcept {
        if (!ptr) return;
        
        ptr->~T();
        
        *reinterpret_cast<T**>(ptr) = free_list_;
        free_list_ = ptr;
        --allocated_;
    }
    
    size_t allocated_count() const noexcept { return allocated_; }
    size_t capacity() const noexcept { return N; }

private:
    T* storage_;
    T* free_list_;
    size_t allocated_;
};

} // namespace lob
