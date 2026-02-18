#include "../src/engine/thread_pool.h"
#include <iostream>
#include <cassert>
#include <atomic>
#include <vector>

using namespace risk_engine::engine;

void test_basic_execution() {
    ThreadPool pool(4);
    
    std::atomic<int> counter{0};
    std::vector<std::future<void>> futures;
    
    for (int i = 0; i < 100; ++i) {
        futures.push_back(pool.enqueue([&counter]() {
            counter++;
        }));
    }
    
    for (auto& fut : futures) {
        fut.get();
    }
    
    assert(counter == 100);
    std::cout << "Basic execution test passed" << std::endl;
}

void test_return_values() {
    ThreadPool pool(4);
    
    auto future = pool.enqueue([]() {
        return 42;
    });
    
    assert(future.get() == 42);
    std::cout << "Return values test passed" << std::endl;
}

void test_parallel_computation() {
    ThreadPool pool(4);
    
    std::vector<std::future<int>> futures;
    for (int i = 0; i < 10; ++i) {
        futures.push_back(pool.enqueue([i]() {
            int sum = 0;
            for (int j = 0; j <= i; ++j) {
                sum += j;
            }
            return sum;
        }));
    }
    
    int total = 0;
    for (auto& fut : futures) {
        total += fut.get();
    }
    
    assert(total == 165);
    std::cout << "Parallel computation test passed" << std::endl;
}

int main() {
    test_basic_execution();
    test_return_values();
    test_parallel_computation();
    std::cout << "All thread pool tests passed!" << std::endl;
    return 0;
}
