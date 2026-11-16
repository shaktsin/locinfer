#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct TensorInfo {
    std::string dtype;
    std::vector<std::uint64_t> shape;
    std::uint64_t begin{};
    std::uint64_t end{};
};

struct ModelConfig {
    std::string architecture;
    int vocab_size{};
    int hidden_size{};
    int num_layers{};
};

struct Model {
    ModelConfig config;
    std::unordered_map<std::string, TensorInfo> tensors;
};
