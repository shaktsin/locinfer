#include "safetensors.hpp"
#include <filesystem>
#include <iostream>
#include <fstream>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: locinf <model_file_path>" << std::endl;
        return 1;
    }

    const std::filesystem::path file_path = argv[1];
    try {
        SafetensorFile safetensor_file(file_path / "model.safetensors");

    } catch (const std::exception& e) {
        std::cerr << "Error loading model configuration: " << e.what() << std::endl;
        return 1;
    }


    return 0;
}
