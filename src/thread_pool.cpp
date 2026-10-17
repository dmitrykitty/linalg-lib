#include "thread_pool.hpp"

#include <algorithm>
#include <utility>

#ifdef __linux__
#include <sched.h>

#include <fstream>
#include <set>
#include <string>
#endif

namespace linalg::internal {

ThreadPool::ThreadPool(std::size_t worker_count) {
    workers_.reserve(worker_count);
    for (std::size_t worker = 0; worker < worker_count; ++worker) {
        // Worker w runs task index w + 1; index 0 is the caller.
        workers_.emplace_back(
            [this, worker](std::stop_token stop) { worker_loop(std::move(stop), worker + 1); });
    }
}

bool ThreadPool::try_run_erased(std::size_t count, void* object, Invoke invoke) {
    count = std::clamp<std::size_t>(count, 1, max_threads());
    std::unique_lock region(region_mutex_, std::try_to_lock);
    if (!region.owns_lock()) {
        return false;
    }
    if (count == 1) {
        invoke(object, 0);
        return true;
    }

    {
        const std::scoped_lock lock(state_mutex_);
        task_object_ = object;
        task_invoke_ = invoke;
        active_threads_ = count;
        pending_workers_ = count - 1;
        ++generation_;
    }
    wake_.notify_all();

    invoke(object, 0);

    std::unique_lock lock(state_mutex_);
    finished_.wait(lock, [this] { return pending_workers_ == 0; });
    task_object_ = nullptr;
    task_invoke_ = nullptr;
    return true;
}

void ThreadPool::worker_loop(std::stop_token stop, std::size_t index) {
    std::uint64_t seen_generation = 0;
    while (true) {
        void* object = nullptr;
        Invoke invoke = nullptr;
        {
            std::unique_lock lock(state_mutex_);
            if (!wake_.wait(lock, stop,
                            [&] { return generation_ != seen_generation; })) {
                return; // stop requested by the jthread destructor
            }
            seen_generation = generation_;
            if (index >= active_threads_) {
                continue; // this region needs fewer threads
            }
            object = task_object_;
            invoke = task_invoke_;
        }

        invoke(object, index);

        const std::scoped_lock lock(state_mutex_);
        if (--pending_workers_ == 0) {
            finished_.notify_one();
        }
    }
}

namespace {

// hardware_concurrency() may return 0 when the count is unknown.
std::size_t logical_cpu_count() {
    return std::max(1U, std::thread::hardware_concurrency());
}

} // namespace

std::size_t physical_core_count() {
#ifdef __linux__
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) == 0) {
        // SMT siblings share (package, core) ids, so each pair is one physical core.
        std::set<std::pair<std::string, std::string>> cores;
        for (std::size_t cpu = 0; cpu < static_cast<std::size_t>(CPU_SETSIZE); ++cpu) {
            if (!CPU_ISSET(cpu, &allowed)) {
                continue;
            }
            const std::string topology =
                "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
            std::ifstream package_file(topology + "physical_package_id");
            std::ifstream core_file(topology + "core_id");
            std::string package;
            std::string core;
            if (!(package_file >> package) || !(core_file >> core)) {
                return static_cast<std::size_t>(std::max(1, CPU_COUNT(&allowed)));
            }
            cores.emplace(std::move(package), std::move(core));
        }
        if (!cores.empty()) {
            return cores.size();
        }
    }
#endif
    return logical_cpu_count();
}

ThreadPool& shared_thread_pool() {
    static ThreadPool pool(logical_cpu_count() - 1);
    return pool;
}

} // namespace linalg::internal
