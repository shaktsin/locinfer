#pragma once

#include <string>


struct QwenConfig {

    std::string architecture;
    int vocab_size = 0;
    int hidden_size = 0;
    int num_layers = 0;
};



QwenConfig load_qwen_config(const char* model_file_path);

