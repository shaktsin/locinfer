#pragma once

#include <vector>
#include <cstddef>


enum class DType 
{
    FP16,
};

template <typename T>
struct TensorView
{
    T* data = nullptr;
    std::vector<std::size_t> shape; 
    std::vector<std::ptrdiff_t> strides; 

};
