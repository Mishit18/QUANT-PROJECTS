#pragma once

#include <vector>
#include <memory>
#include <cstdlib>
#include <new>

namespace heston {

constexpr size_t CACHE_LINE_SIZE = 64;
constexpr size_t AVX_ALIGNMENT = 32;
constexpr size_t AVX512_ALIGNMENT = 64;

template<typename T>
class AlignedAllocator {
public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    
    AlignedAllocator(size_t alignment = AVX_ALIGNMENT) noexcept 
        : alignment_(alignment) {}
    
    template<typename U>
    AlignedAllocator(const AlignedAllocator<U>& other) noexcept
        : alignment_(other.alignment_) {}
    
    T* allocate(size_t n) {
        if (n == 0) return nullptr;
        
        size_t size = n * sizeof(T);
        void* ptr = nullptr;
        
#ifdef _WIN32
        ptr = _aligned_malloc(size, alignment_);
#else
        if (posix_memalign(&ptr, alignment_, size) != 0) {
            ptr = nullptr;
        }
#endif
        
        if (!ptr) throw std::bad_alloc();
        return static_cast<T*>(ptr);
    }
    
    void deallocate(T* ptr, size_t) noexcept {
        if (!ptr) return;
#ifdef _WIN32
        _aligned_free(ptr);
#else
        free(ptr);
#endif
    }
    
    size_t alignment_;
};

template<typename T>
using AlignedVector = std::vector<T, AlignedAllocator<T>>;

template<typename T>
class ObjectPool {
public:
    explicit ObjectPool(size_t initial_size = 100) {
        pool_.reserve(initial_size);
        for (size_t i = 0; i < initial_size; ++i) {
            pool_.push_back(std::make_unique<T>());
        }
    }
    
    T* acquire() {
        if (pool_.empty()) {
            return new T();
        }
        T* obj = pool_.back().release();
        pool_.pop_back();
        return obj;
    }
    
    void release(T* obj) {
        pool_.push_back(std::unique_ptr<T>(obj));
    }
    
private:
    std::vector<std::unique_ptr<T>> pool_;
};

} // namespace heston
