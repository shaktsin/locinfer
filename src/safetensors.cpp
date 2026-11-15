#include "safetensors.hpp"

#include <array>
#include <iostream>
#include <stdexcept>
#include <nlohmann/json.hpp>

std::uint64_t read_header_size(std::istream& file) {
    std::array<char, 8> bytes{};

    if (!file.read(bytes.data(), bytes.size())) {
        throw std::runtime_error("Could not read the 8-byte header length");
    }

    std::uint64_t size = 0;
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        const auto byte = static_cast<unsigned char>(bytes[i]);
        size |= static_cast<std::uint64_t>(byte) << (8 * i);
    }

    return size;
}

SafetensorFile::SafetensorFile(const std::filesystem::path& file_path) : file_(file_path, std::ios::binary) {
    if (!file_.is_open()) {
        throw std::runtime_error("Failed to open safetensor file: " + file_path.string());
    }

    // file size
    file_.seekg(0, std::ios::end);
    std::uint64_t file_size = file_.tellg();
    file_.seekg(0, std::ios::beg);

    // read header size
    std::uint64_t header_size = read_header_size(file_);
    if (header_size <= 0 || header_size > file_size) {
        throw std::runtime_error("Header size is invalid in safetensor file: " + file_path.string());
    }

    // read header 
    std::string header_json(header_size, '\0');
    if (!file_.read(header_json.data(), header_size)) {
        throw std::runtime_error("Failed to read header from safetensor file: " + file_path.string());
    }

    // parse header
    nlohmann::json header = nlohmann::json::parse(header_json);
    for (const auto& [name, entry]: header.items()) {
        if (name == "__metadata__") {
            continue; // skip metadata
        }

        // SET DATA START POSITION
        if (data_start_ == 0) {
            data_start_ = 8 + header_size; // 8 bytes for header size + header size  
        }


        TensorInfo tensor_info;
        tensor_info.dtype = entry["dtype"].get<std::string>();
        tensor_info.shape = entry["shape"].get<std::vector<std::uint64_t>>();
        tensor_info.begin = data_start_ + entry["data_offsets"][0].get<std::uint64_t>();
        tensor_info.end = data_start_ + entry["data_offsets"][1].get<std::uint64_t>();

        tensors_.emplace(name, std::move(tensor_info));
    }
    
}
