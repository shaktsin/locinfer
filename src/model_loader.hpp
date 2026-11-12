#pragma once


struct QwenConfig {

    std::string architecture;
    int vocab_size = 0;
    int hidden_size = 0;
    int num_layers = 0;
};

struct TensorInfo {
    std::string dtype;
    std::vector<int64_t> shape;
    size_t data_begin = 0;
    size_t data_end = 0;
};

QwenConfig load_qwen_config(const char* model_file_path);

