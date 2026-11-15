#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

struct TensorInfo {
    std::string dtype;
    std::vector<std::uint64_t> shape;
    std::uint64_t begin{};
    std::uint64_t end{};
};

class SafetensorFile {
public:
    explicit SafetensorFile(const std::filesystem::path& file_path);

    const TensorInfo& tensor_info(const std::string& name) const;

private:
    std::ifstream file_;
    std::uint64_t data_start_{};
    std::unordered_map<std::string, TensorInfo> tensors_;
};
