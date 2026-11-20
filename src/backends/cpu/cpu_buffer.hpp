#pragma once 
#include <cstddef>


template <typename T>
class CpuBuffer 
{
    public:
        explicit CpuBuffer(std::size_t count); 

        std::size_t size() const;
        std::span<T> data();
        std::span<const T> data() const; 
};