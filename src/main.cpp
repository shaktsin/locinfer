#include "checkpoint.hpp"
#include "safetensor_checkpoint.hpp"
#include <filesystem>
#include <iostream>
#include <memory>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: locinf <model_folder_path>" << std::endl;
        return 1;
    }

    const std::filesystem::path file_path = argv[1];
    try {
        std::unique_ptr<Checkpoint> checkpoint =
            std::make_unique<SafetensorsCheckpoint>(file_path);
        Model model = checkpoint->load();
        std::cout << "Loaded model from: " << file_path << '\n'
                  << "Architecture: " << model.config.architecture << '\n'
                  << "Vocabulary size: " << model.config.vocab_size << '\n'
                  << "Hidden size: " << model.config.hidden_size << '\n'
                  << "Layers: " << model.config.num_layers << '\n'
                  << "Tensors: " << model.tensors.size() << '\n';

    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return 1;
    }


    return 0;
}
