#include "thread_pool.h"

namespace risk_engine {
namespace engine {

ThreadPool::ThreadPool(size_t num_threads) : stop_(false), active_tasks_(0) {
    for (size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back([this] { worker_thread(); });
    }
}

ThreadPool::~ThreadPool() {
    stop_ = true;
    condition_.notify_all();
    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void ThreadPool::worker_thread() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            condition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
            
            if (stop_ && tasks_.empty()) {
                return;
            }
            
            if (!tasks_.empty()) {
                task = std::move(tasks_.front());
                tasks_.pop();
                active_tasks_++;
            }
        }
        
        if (task) {
            task();
            active_tasks_--;
        }
    }
}

void ThreadPool::wait_all() {
    while (active_tasks_ > 0 || !tasks_.empty()) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}

} // namespace engine
} // namespace risk_engine
