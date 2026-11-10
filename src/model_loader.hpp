#pragma once


struct ModelConfig {

    int hidden_size{};
    int intermediate_size{};
    int num_hidden_layers{};
    int num_attention_heads{};
    int num_key_value_heads{};
    int vocab_size{};
    int max_position_embeddings{};
};

ModelConfig load_model_config(const char* model_file_path);

