#include "safetensor_checkpoint.hpp"
#include "model.hpp"
#include <array>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <utility>

namespace
{
    std::uint64_t read_header_size(std::istream &file)
    {
        std::array<char, 8> bytes{};

        if (!file.read(bytes.data(), bytes.size()))
        {
            throw std::runtime_error("Could not read the 8-byte header length");
        }

        std::uint64_t size = 0;
        for (std::size_t i = 0; i < bytes.size(); ++i)
        {
            const auto byte = static_cast<unsigned char>(bytes[i]);
            size |= static_cast<std::uint64_t>(byte) << (8 * i);
        }

        return size;
    }
}

ModelConfig SafetensorsCheckpoint::read_config(const std::filesystem::path &file_path) const
{

    ModelConfig config;
    nlohmann::json json_config;
    std::ifstream file(file_path);

    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open model config file: " + std::string(file_path));
    }

    try
    {
        json_config = nlohmann::json::parse(file);
    }
    catch (const nlohmann::json::parse_error &e)
    {
        throw std::runtime_error("Invalid Json file: " + std::string(file_path));
    }

    config.architecture = json_config.at("architectures").at(0).get<std::string>();
    config.vocab_size = json_config.at("vocab_size").get<int>();
    config.hidden_size = json_config.at("hidden_size").get<int>();
    config.num_layers = json_config.at("num_hidden_layers").get<int>();

    return config;
}

std::unordered_map<std::string, TensorInfo> SafetensorsCheckpoint::read_tensorinfo(const std::filesystem::path &file_path) const
{
    std::unordered_map<std::string, TensorInfo> tensors;
    std::ifstream file_(file_path, std::ios::binary);
    std::uint64_t data_start_{0};

    if (!file_.is_open())
    {
        throw std::runtime_error("Failed to open safetensor file: " + file_path.string());
    }

    // file size
    file_.seekg(0, std::ios::end);
    std::uint64_t file_size = file_.tellg();
    file_.seekg(0, std::ios::beg);

    // read header size
    std::uint64_t header_size = read_header_size(file_);
    if (header_size == 0 || header_size > file_size - 8)
    {
        throw std::runtime_error("Header size is invalid in safetensor file: " + file_path.string());
    }

    // read header
    std::string header_json(header_size, '\0');
    if (!file_.read(header_json.data(), header_size))
    {
        throw std::runtime_error("Failed to read header from safetensor file: " + file_path.string());
    }

    // parse header
    nlohmann::json header = nlohmann::json::parse(header_json);
    for (const auto &[name, entry] : header.items())
    {
        if (name == "__metadata__")
        {
            continue; // skip metadata
        }

        // SET DATA START POSITION
        if (data_start_ == 0)
        {
            data_start_ = 8 + header_size; // 8 bytes for header size + header size
        }

        TensorInfo tensor_info;
        tensor_info.dtype = entry["dtype"].get<std::string>();
        tensor_info.shape = entry["shape"].get<std::vector<std::uint64_t>>();
        tensor_info.begin = data_start_ + entry["data_offsets"][0].get<std::uint64_t>();
        tensor_info.end = data_start_ + entry["data_offsets"][1].get<std::uint64_t>();

        tensors.emplace(name, std::move(tensor_info));
    }
    return tensors;
}

Model SafetensorsCheckpoint::load() const
{
    return Model{
        .config = read_config(model_path_ / "config.json"),
        .tensors = read_tensorinfo(model_path_ / "model.safetensors")};
}
