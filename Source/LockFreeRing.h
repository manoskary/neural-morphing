#pragma once

#include <atomic>
#include <optional>
#include <stdexcept>
#include <vector>

template <typename T>
class LockFreeRing
{
public:
    explicit LockFreeRing(size_t capacity)
        : buffer_(capacity)
    {
        if (capacity < 2)
            throw std::invalid_argument("LockFreeRing capacity must be >= 2");
    }

    bool push(T&& value)
    {
        auto head = head_.load(std::memory_order_relaxed);
        auto next = increment(head);
        if (next == tail_.load(std::memory_order_acquire))
            return false;

        buffer_[head] = std::move(value);
        head_.store(next, std::memory_order_release);
        return true;
    }

    bool push(const T& value)
    {
        auto head = head_.load(std::memory_order_relaxed);
        auto next = increment(head);
        if (next == tail_.load(std::memory_order_acquire))
            return false;

        buffer_[head] = value;
        head_.store(next, std::memory_order_release);
        return true;
    }

    bool pop(T& out)
    {
        auto tail = tail_.load(std::memory_order_relaxed);
        if (tail == head_.load(std::memory_order_acquire))
            return false;

        out = std::move(buffer_[tail]);
        tail_.store(increment(tail), std::memory_order_release);
        return true;
    }

    std::optional<T> pop()
    {
        T value{};
        if (!pop(value))
            return std::nullopt;
        return value;
    }

    void clear()
    {
        tail_.store(head_.load(std::memory_order_acquire), std::memory_order_release);
    }

    size_t capacity() const noexcept { return buffer_.size(); }

private:
    size_t increment(size_t idx) const noexcept
    {
        return (idx + 1) % buffer_.size();
    }

    std::vector<T> buffer_;
    std::atomic<size_t> head_{0};
    std::atomic<size_t> tail_{0};
};
