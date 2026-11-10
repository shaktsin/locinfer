#include "model_loader.hpp"
#include <iostream>
#include <fstream>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: locinf <model_file_path>" << std::endl;
        return 1;
    }

    const char* file_path = argv[1];


    try {
        auto model_config = load_model_config(file_path);

        std::cout << "Model Configuration:" << std::endl;
        std::cout << "Hidden Size: " << model_config.hidden_size << std::endl;
        std::cout << "Intermediate Size: " << model_config.intermediate_size << std::endl;
        std::cout << "Number of Hidden Layers: " << model_config.num_hidden_layers << std::endl;
        std::cout << "Number of Attention Heads: " << model_config.num_attention_heads << std::endl;
        std::cout << "Number of Key-Value Heads: " << model_config.num_key_value_heads << std::endl;
        std::cout << "Vocabulary Size: " << model_config.vocab_size << std::endl;
        std::cout << "Maximum Position Embeddings: " << model_config.max_position_embeddings << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error loading model configuration: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}