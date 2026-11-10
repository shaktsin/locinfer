#include "model_loader.hpp"

#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>


ModelConfig load_model_config(const char* model_file_path) {
    std::ifstream file(model_file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open model config file: " + std::string(model_file_path));
    }

    nlohmann::json json_config;
    file >> json_config;

    ModelConfig config;
    config.hidden_size = json_config["hidden_size"].get<int>();
    config.intermediate_size = json_config["intermediate_size"].get<int>();
    config.num_hidden_layers = json_config["num_hidden_layers"].get<int>();
    config.num_attention_heads = json_config["num_attention_heads"].get<int>();
    config.num_key_value_heads = json_config["num_key_value_heads"].get<int>();
    config.vocab_size = json_config["vocab_size"].get<int>();
    config.max_position_embeddings = json_config["max_position_embeddings"].get<int>();

    return config;
}