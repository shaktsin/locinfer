#include "model_loader.hpp"

#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>


QwenConfig load_qwen_config(const char* model_file_path) {
    std::ifstream file(model_file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open model config file: " + std::string(model_file_path));
    }

    nlohmann::json json_config;
    file >> json_config;

    QwenConfig config;
    config.hidden_size = json_config["hidden_size"].get<int>();
    return config;
}